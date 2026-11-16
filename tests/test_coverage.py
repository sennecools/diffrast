"""Which pixels a triangle covers: the fill rule on edges, cracks, degenerate and non-finite
triangles, and size limits. Every rule is checked on both backends, and the Triton backend is
compared with the reference backend pixel by pixel."""

import importlib

import pytest
import torch

import diffrast as dr
from diffrast.backends import reference
from diffrast.ops import check_rasterize_inputs
from helpers import screen_space_rasterize

QUAD = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
RESOLUTIONS = [(100, 100), (255, 255), (1000, 1000), (600, 800)]
# Directions of the two infinitesimal steps in exact_owner; the first is the fill rule.
STEPS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
SHEETS = {
    "facing": ((6, 5), (6, 0), (0, 6)),
    "rotated 45 degrees": ((32, 4), (4, 4), (-4, 4)),
    "generic angle": ((20, 3), (6, 2), (-2, 6)),
}


def load_triton():
    pytest.importorskip("triton")
    return importlib.import_module("diffrast.backends.triton")


@pytest.fixture(params=["reference", pytest.param("triton", marks=pytest.mark.gpu)])
def backend(request):
    """A backend module and the device it is run on."""
    if request.param == "triton":
        return load_triton(), "cuda"
    return reference, "cpu"


def peel(backend, pos, tri, resolution, count=1):
    """Triangle index per pixel (-1 for background) of the first ``count`` depth layers."""
    module, device = backend
    pos, tri = pos.to(device), tri.to(device)
    state, layers = None, []
    for _ in range(count):
        rast, _, state = module.rasterize_layer(pos, tri, resolution, True, state)
        layers.append(rast[..., 3].long().cpu() - 1)
    return layers


def owner(backend, pos, tri, resolution):
    return peel(backend, pos, tri, resolution)[0]


def lattice(cols, rows, resolution, w=1.0, z=0.0, device="cpu", dtype=torch.float32):
    """Clip-space vertices that project onto the centres of the pixels ``(rows, cols)``.

    The centres are computed like the rasterizer's own and on the same device, where a division
    by a scalar may round differently, so that the edge functions through a vertex are exactly
    zero at its pixel. Indices outside the image give lattice points off screen.
    """
    height, width = resolution
    x = (cols.to(device, dtype) * 2 + 1) / width - 1
    y = (rows.to(device, dtype) * 2 + 1) / height - 1
    w = torch.as_tensor(w, dtype=dtype, device=device).expand(x.shape)
    return torch.stack([x * w, y * w, z * w, w], -1)[None]


def exact_owner(cols, rows, tri, resolution, step=(1, 1)):
    """Coverage in integer arithmetic of triangles with vertices on pixel centres.

    Each pixel is sampled at its centre moved by an infinitesimal step along x and a much smaller
    one along y, in the directions given by ``step``; ``(1, 1)`` is the fill rule. Returns the
    index of a covering triangle per pixel (-1 for none) and the number of covering triangles.
    """
    height, width = resolution
    v = torch.stack([cols, rows], -1).long()[tri.long()][:, :, :, None, None]
    px, py = torch.arange(width).view(1, 1, width), torch.arange(height).view(1, height, 1)
    # Coordinates are far below 2**13, so with this scale neither step can outweigh a nonzero
    # value of the term before it.
    scale = 1 << 13
    area, shifted = 0, []
    for k in range(3):
        (ax, ay), (bx, by) = v[:, (k + 1) % 3].unbind(1), v[:, (k + 2) % 3].unbind(1)
        e = (ax - px) * (by - py) - (bx - px) * (ay - py)
        area = area + e
        shifted.append((e * scale + step[0] * (ay - by)) * scale + step[1] * (bx - ax))
    inside = torch.stack([value * area.sign() > 0 for value in shifted]).all(0)
    count = inside.sum(0)
    return torch.where(count > 0, inside.long().argmax(0), -1), count


def shuffle_vertices(tri, gen):
    """List the vertices of each triangle in a random order, which also randomizes winding."""
    order = torch.stack([torch.randperm(3, generator=gen) for _ in range(tri.shape[0])])
    return tri.gather(1, order)


def grid_triangles(nx, ny, gen=None):
    """Triangles of a grid of ``nx`` by ``ny`` quads over ``(ny + 1) * (nx + 1)`` vertices.

    With ``gen``, each quad is split along a random diagonal and the vertices of each triangle
    are shuffled, so neighbours differ in winding.
    """
    idx = torch.arange((nx + 1) * (ny + 1)).view(ny + 1, nx + 1)
    a, b, c, d = (t.flatten() for t in (idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]))
    first, second = torch.stack([a, b, c], -1), torch.stack([a, c, d], -1)
    if gen is None:
        return torch.cat([first, second]).int()
    other = (torch.rand(nx * ny, generator=gen) < 0.5)[:, None]
    first = torch.where(other, torch.stack([a, b, d], -1), first)
    second = torch.where(other, torch.stack([b, c, d], -1), second)
    return shuffle_vertices(torch.cat([first, second]), gen).int()


def sheet(origin, u, v, n, gen):
    """Lattice coordinates and triangles of an ``n`` by ``n`` grid of quads with sides u and v."""
    j, i = torch.meshgrid(torch.arange(n + 1), torch.arange(n + 1), indexing="ij")
    cols = (origin[0] + i * u[0] + j * v[0]).flatten()
    rows = (origin[1] + i * u[1] + j * v[1]).flatten()
    return cols, rows, grid_triangles(n, n, gen)


def screen_quad(scale=1.0, w=1.0):
    corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    return torch.tensor([[[x * scale * w, y * scale * w, 0.0, w] for x, y in corners]])


def corner_grid(nx, ny, w=1.0):
    """A grid of quads filling the screen exactly, so its diagonals pass through pixel centres."""
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, ny + 1), torch.linspace(-1, 1, nx + 1), indexing="ij"
    )
    xy = torch.stack([x, y], -1).reshape(-1, 2)
    return torch.cat([xy * w, torch.zeros(len(xy), 1), torch.full((len(xy), 1), w)], -1)[None]


def covering_lattice(resolution, cell, gen, device, w=None):
    """A lattice mesh of ``cell``-pixel quads that extends past every side of the image."""
    height, width = resolution
    nx, ny = (width + 1) // cell + 1, (height + 1) // cell + 1
    rows, cols = torch.meshgrid(
        torch.arange(ny + 1) * cell - 1, torch.arange(nx + 1) * cell - 1, indexing="ij"
    )
    if w is None:
        w = torch.rand(rows.numel(), generator=gen) + 0.5
    pos = lattice(cols.flatten(), rows.flatten(), resolution, w=w, device=device)
    return pos, grid_triangles(nx, ny, gen)


def full_screen_scenes(resolution, gen, cell, device):
    """Meshes that cover every pixel exactly once, with many edges through pixel centres."""
    nx, ny = resolution[1] // cell, resolution[0] // cell
    return {
        "quad": (screen_quad(), QUAD),
        "oversized quad": (screen_quad(1.3, w=0.37), QUAD),
        "corner grid": (corner_grid(nx, ny), grid_triangles(nx, ny, gen)),
        "corner grid, w = 2.5": (corner_grid(4, 3, w=2.5), grid_triangles(4, 3, gen)),
        "lattice": covering_lattice(resolution, cell, gen, device, w=1.0),
        "lattice, w per vertex": covering_lattice(resolution, cell, gen, device),
    }


def small_triangles(count, size, gen, perspective=True):
    """Triangles about ``size`` NDC units across, scattered over and just past the screen."""
    centre = torch.rand(count, 1, 2, generator=gen) * 2.2 - 1.1
    xy = (centre + (torch.rand(count, 3, 2, generator=gen) * 2 - 1) * size).reshape(1, -1, 2)
    z = torch.rand(1, 3 * count, 1, generator=gen) * 1.8 - 0.9
    w = torch.rand(1, 3 * count, 1, generator=gen) + 1.0 if perspective else torch.ones_like(z)
    return torch.cat([xy * w, z * w, w], -1), torch.arange(3 * count).view(count, 3).int()


def collinear_triangles(count, gen, perspective):
    """Triangles ``(a, b, midpoint of a and b)`` with ``a`` and ``b`` on a lattice of 1 / 10."""
    ends = torch.randint(-9, 10, (count, 2, 2), generator=gen).float() / 10
    xy = torch.cat([ends, ends.mean(1, keepdim=True)], 1).reshape(1, -1, 2)
    w = torch.ones(1, 3 * count, 1)
    if perspective:
        w = w + torch.rand(1, 3 * count, 1, generator=gen)
    pos = torch.cat([xy * w, torch.zeros_like(w), w], -1)
    return pos, torch.arange(3 * count, dtype=torch.int32).view(count, 3)


def integer_homography_grid(gen):
    """A planar grid seen through a random integer homography, as integer ``(x, y, w)`` per vertex.

    Many of these grids cross the camera plane, and many pixel centres lie exactly on their edges.
    """
    while True:
        h = torch.randint(-4, 5, (3, 3), generator=gen)
        minors = h[1, [1, 2, 0]] * h[2, [2, 0, 1]] - h[1, [2, 0, 1]] * h[2, [1, 2, 0]]
        if (h[0] * minors).sum() != 0:
            break
    n = int(torch.randint(2, 6, (1,), generator=gen))
    line = torch.arange(n + 1) - int(torch.randint(1, 4, (1,), generator=gen))
    v, u = torch.meshgrid(line, line, indexing="ij")
    grid = torch.stack([u.flatten(), v.flatten(), torch.ones(u.numel(), dtype=torch.long)], -1)
    return grid @ h.T, grid_triangles(n, n, gen)


def exact_homogeneous_owner(clip, tri, size):
    """Coverage in integer arithmetic of triangles with integer clip coordinates ``(x, y, w)``.

    Each pixel of a ``size`` by ``size`` image is sampled at its centre moved by an infinitesimal
    step along +x and a much smaller one along +y: where an edge function is zero, the sign of
    its derivative along x decides, and that along y where this is zero as well. Returns the
    covering triangle per pixel (-1 for none), the number of covering triangles, the pixels
    whose owner has such a zero, those of them whose owner crosses the camera plane, and the
    pixels left undecided because a triangle's area there is zero or close to the tolerance.
    """
    x, y, w = (clip[tri.long()][:, :, k, None, None] for k in range(3))
    centre = torch.arange(size) * 2 + 1 - size
    a = x * size - centre.view(1, 1, 1, size) * w
    c = y * size - centre.view(1, 1, size, 1) * w
    area, ew, zero, shifted = 0, 0, False, []
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        e = a[:, i] * c[:, j] - a[:, j] * c[:, i]
        along_x = y[:, i] * w[:, j] - y[:, j] * w[:, i]
        along_y = x[:, j] * w[:, i] - x[:, i] * w[:, j]
        area, ew, zero = area + e, ew + e * w[:, k], zero | (e == 0)
        shifted.append(torch.where(e != 0, e, torch.where(along_x != 0, along_x, along_y)))
    inside = torch.stack([value * area.sign() > 0 for value in shifted]).all(0) & (ew * area > 0)
    count = inside.sum(0)

    reach_x, reach_y = ((t.abs() + w.abs()).amax(1).double() for t in (x, y))
    span_x, span_y = (t.abs().sum(1).double() / size for t in (a, c))
    noise = (reach_x * span_y + span_x * (reach_y + span_y)) * torch.finfo(torch.float32).eps
    undecided = (area.abs().double() / size**2 < 4 * noise).any(0)
    tie = (inside & zero).any(0)
    crossing = (inside & zero & (w.amin(1) <= 0)).any(0)
    return torch.where(count > 0, inside.long().argmax(0), -1), count, tie, crossing, undecided


def sliver(half_width, steps):
    """A triangle around the centre of pixel ``(8, 8)`` of a 16 by 16 image.

    It is ``2 * half_width`` wide and ``2 * steps`` float32 steps of its y coordinates high. Its
    coordinates, and everything the coverage test computes from them at that pixel, are exact.
    """
    centre, step = 0.0625, 2.0**-27
    xy = [[-half_width, -steps * step], [half_width, -steps * step], [0.0, steps * step]]
    return torch.tensor([[[centre + x, centre + y, 0.0, 1.0] for x, y in xy]])


# Fill rule


@pytest.mark.parametrize("resolution", [(8, 8), (100, 100), (255, 255), (1000, 1000)])
@pytest.mark.parametrize("scale", [1.0, 0.9])
def test_quad_diagonal_goes_to_the_triangle_on_its_right(backend, resolution, scale):
    # The diagonal passes through the centres of the pixels on it, where its edge function is
    # exactly zero in both triangles: triangle 0 lies in the +x direction and takes them all.
    size = resolution[0]
    ids = owner(backend, screen_quad(scale), QUAD, resolution)[0]
    centre = (torch.arange(size, dtype=torch.float64) * 2 + 1) / size - 1
    inside = (centre.abs() < scale)[:, None] & (centre.abs() < scale)[None, :]
    row, col = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    assert torch.equal(ids, torch.where(inside, torch.where(col >= row, 0, 1), -1))


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0]])
@pytest.mark.parametrize("w", [1.0, 2.5])
def test_rectangle_owns_its_left_and_top_edges_only(backend, order, w):
    # All four sides pass through pixel centres. Row 0 is at y = -1, so the side whose interior
    # lies in the +y direction is the one with the lowest row.
    resolution = (20, 24)
    cols, rows = torch.tensor([5, 17, 17, 5])[order], torch.tensor([3, 3, 14, 14])[order]
    pos = lattice(cols, rows, resolution, w=w, device=backend[1])
    ids = owner(backend, pos, QUAD, resolution)[0]
    expected = torch.zeros(resolution, dtype=torch.bool)
    expected[3:14, 5:17] = True
    assert torch.equal(ids >= 0, expected)


def test_fill_rule_matches_shifted_sampling(backend):
    # Right triangles with two axis-aligned sides, whose edge functions are exactly zero on the
    # pixel centres they pass, and legs of coprime length, so the third side passes none. Every
    # pixel then has to match the integer computation.
    resolution = (40, 48)
    gen = torch.Generator().manual_seed(0)
    for _ in range(40):
        col, row = torch.randint(18, 24, (2,), generator=gen).tolist()
        dx = (3, 5, 7, 9, 11)[int(torch.randint(5, (1,), generator=gen))]
        dy = (2, 4, 8, 16)[int(torch.randint(4, (1,), generator=gen))]
        sx, sy = (torch.randint(0, 2, (2,), generator=gen) * 2 - 1).tolist()
        cols = torch.tensor([col, col + sx * dx, col])
        rows = torch.tensor([row, row, row + sy * dy])
        tri = shuffle_vertices(torch.tensor([[0, 1, 2]]), gen).int()
        w = torch.rand(3, generator=gen) + 0.5
        pos = lattice(cols, rows, resolution, w=w, device=backend[1])
        expected, _ = exact_owner(cols, rows, tri, resolution)
        assert torch.equal(owner(backend, pos, tri, resolution)[0], expected)


def assert_partition(layers, cols, rows, tri, resolution):
    """The first layer covers the mesh without holes or overlaps and the second is empty.

    A pixel whose centre lies on an edge may go to any triangle touching it there, since an edge
    function that is zero in exact arithmetic can round to either side; every other pixel has to
    match the integer computation.
    """
    first, second = (layer[0] for layer in layers)
    owners = [exact_owner(cols, rows, tri, resolution, step) for step in STEPS]
    assert all(int(count.max()) == 1 for _, count in owners), "the mesh overlaps itself"
    candidates = torch.stack([ids for ids, _ in owners])
    clear = (candidates == candidates[0]).all(0)
    interior = (candidates >= 0).all(0)
    assert (clear & interior).sum() > 100 and (interior & ~clear).sum() > 10
    assert torch.equal(first[clear], candidates[0][clear])
    assert (first[interior][None] == candidates[:, interior]).any(0).all()
    assert (second == -1).all()


@pytest.mark.parametrize("name", SHEETS)
@pytest.mark.parametrize("w", ["one", "uniform", "per vertex"])
def test_flat_sheet_has_a_single_layer(backend, name, w):
    origin, u, v = SHEETS[name]
    resolution = (60, 64)
    gen = torch.Generator().manual_seed(1)
    cols, rows, tri = sheet(origin, u, v, 6, gen)
    w = {"one": 1.0, "uniform": 2.5, "per vertex": torch.rand(49, generator=gen) + 0.5}[w]
    pos = lattice(cols, rows, resolution, w=w, z=0.25, device=backend[1])
    assert_partition(peel(backend, pos, tri, resolution, 2), cols, rows, tri, resolution)


@pytest.mark.parametrize(("u", "v"), [((3, 0), (0, 3)), ((2, 2), (-2, 2)), ((3, 1), (-1, 3))])
def test_sheet_of_triangles_a_few_pixels_across(backend, u, v):
    # Small enough for the Triton backend to rasterize them directly, without binning into tiles.
    resolution = (40, 44)
    gen = torch.Generator().manual_seed(4)
    cols, rows, tri = sheet((16, 4), u, v, 8, gen)
    w = torch.rand(81, generator=gen) + 0.5
    pos = lattice(cols, rows, resolution, w=w, z=-0.5, device=backend[1])
    assert_partition(peel(backend, pos, tri, resolution, 2), cols, rows, tri, resolution)


def test_depth_peeler_finds_no_second_layer_in_a_flat_sheet(ctx):
    resolution = (60, 64)
    cols, rows, tri = sheet(*SHEETS["facing"], 6, torch.Generator().manual_seed(1))
    with dr.DepthPeeler(ctx, lattice(cols, rows, resolution), tri, resolution) as peeler:
        first, _ = peeler.rasterize_next_layer()
        second, second_db = peeler.rasterize_next_layer()
    assert (first[..., 3] > 0).sum() == 36 * 36
    assert (second == 0).all() and (second_db == 0).all()


@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_meshes_through_pixel_centres_cover_every_pixel_once(backend, resolution):
    gen = torch.Generator().manual_seed(2)
    cell = 20 if backend[1] == "cuda" else resolution[1] // 4
    for name, (pos, tri) in full_screen_scenes(resolution, gen, cell, backend[1]).items():
        first, second = peel(backend, pos, tri, resolution, 2)
        assert (first >= 0).all(), f"{name}: holes"
        assert (second == -1).all(), f"{name}: pixels covered twice"


@pytest.mark.parametrize("seed", range(4))
def test_triangles_sharing_an_edge_cover_their_union_once(backend, seed):
    # The shared edge runs between lattice points, so it passes through pixel centres. The second
    # triangle lies on the other side of it and is listed with either winding.
    resolution = (36, 40)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(10):
        a, b = torch.randint(8, 28, (2, 2), generator=gen) // 4 * 4 + 2
        along = b - a
        if not along.any():
            continue
        normal = torch.stack([-along[1], along[0]])
        c = a + normal + torch.randint(-3, 4, (2,), generator=gen)
        d = b - normal + torch.randint(-3, 4, (2,), generator=gen)
        cols, rows = torch.stack([a, b, c, d]).unbind(-1)
        second = [[1, 0, 3], [0, 1, 3], [3, 0, 1]][int(torch.randint(3, (1,), generator=gen))]
        tri = torch.tensor([[0, 1, 2], second], dtype=torch.int32)
        w = torch.rand(4, generator=gen) + 0.5
        pos = lattice(cols, rows, resolution, w=w, device=backend[1])

        alone = [owner(backend, pos, tri[k : k + 1], resolution)[0] >= 0 for k in (0, 1)]
        assert not (alone[0] & alone[1]).any()
        assert torch.equal(alone[0] | alone[1], owner(backend, pos, tri, resolution)[0] >= 0)
        steps = int(torch.gcd(along[0], along[1]))
        on_edge = a + along // steps * torch.arange(1, steps)[:, None]
        assert (alone[0] | alone[1])[on_edge[:, 1], on_edge[:, 0]].all()


@pytest.mark.parametrize("second", [[1, 0, 3], [0, 1, 3]])
def test_shared_edge_of_a_triangle_crossing_the_camera_plane(backend, second):
    # Vertex 3 projects onto the same side of the shared edge as vertex 2, but lies behind the
    # camera, so the visible part of its triangle is on the opposite side of the edge.
    resolution = (32, 32)
    cols, rows = torch.tensor([6, 26, 20, 14]), torch.tensor([6, 26, 8, 4])
    w = torch.tensor([1.0, 1.5, 0.75, 1.0])
    pos = lattice(cols, rows, resolution, w=w, z=0.1, device=backend[1])
    pos[0, 3] = -pos[0, 3]
    tri = torch.tensor([[0, 1, 2], second], dtype=torch.int32)

    alone = [owner(backend, pos, tri[k : k + 1], resolution)[0] >= 0 for k in (0, 1)]
    assert alone[0].sum() > 20 and alone[1].sum() > 20
    assert not (alone[0] & alone[1]).any()
    assert torch.equal(alone[0] | alone[1], owner(backend, pos, tri, resolution)[0] >= 0)
    assert (alone[0] | alone[1]).diagonal()[7:26].all()


@pytest.mark.parametrize("count", [3, 4, 7])
@pytest.mark.parametrize("on_lattice", [True, False])
def test_vertex_of_a_closed_fan_is_covered_once(backend, count, on_lattice):
    # The hub lies on a pixel centre, where two edge functions of every triangle in the fan are
    # exactly zero.
    resolution = (24, 24)
    gen = torch.Generator().manual_seed(count)
    for _ in range(6):
        angle = (torch.arange(count) + torch.rand(count, generator=gen) * 0.4) / count
        angle = (angle + torch.rand(1, generator=gen)) * 6.2832
        rim = torch.stack([angle.cos(), angle.sin()], -1) * 9
        hub = torch.tensor([[12.0, 11.0]])
        cols, rows = torch.cat([hub, hub + (rim.round() if on_lattice else rim)]).unbind(-1)
        w = torch.cat([torch.ones(1), torch.rand(count, generator=gen) + 0.5])
        pos = lattice(cols, rows, resolution, w=w, device=backend[1])
        spokes = torch.arange(count)
        tri = torch.stack([torch.zeros_like(spokes), spokes + 1, (spokes + 1) % count + 1], -1)
        tri = shuffle_vertices(tri, gen).int()

        alone = [owner(backend, pos, tri[k : k + 1], resolution)[0, 11, 12] >= 0 for k in spokes]
        assert sum(int(covers) for covers in alone) == 1
        first, second = peel(backend, pos, tri, resolution, 2)
        assert first[0, 11, 12] >= 0 and (second == -1).all()


@pytest.mark.parametrize("seed", range(6))
def test_fan_with_nearly_horizontal_spokes_covers_its_hub_once(backend, seed):
    # Several spokes leave the hub a few float32 steps off its row, at different lengths, with
    # slivers between them. In order of angle from the +x axis towards +y: a vertex on the -y
    # side of the hub, spokes along +x tilted slightly towards +y, a vertex on the +y side, and
    # spokes along -x tilted slightly towards -y. The first triangle contains the direction just
    # past +x and owns the hub. All coordinates are exact.
    resolution = (16, 16)
    gen = torch.Generator().manual_seed(seed)
    step = 2.0**-24
    for _ in range(8):
        slopes = {}
        while len(slopes) < 6:
            steps, length = (int(torch.randint(1, high, (1,), generator=gen)) for high in (9, 600))
            slopes[steps / length] = (length / 1024, steps * step)
        spokes = [slopes[slope] for slope in sorted(slopes)]
        rim = [(0.125, -0.375), *spokes[::2], (-0.0625, 0.375)]
        rim += [(-length, -rise) for length, rise in spokes[1::2]]
        xy = torch.tensor([(0.0, 0.0), *rim], dtype=torch.float64) + torch.tensor([0.0625, 0.5625])
        pos = torch.cat([xy, torch.zeros(len(xy), 1), torch.ones(len(xy), 1)], -1).float()[None]
        assert torch.equal(pos[0, :, :2].double(), xy)
        index = torch.arange(len(rim))
        tri = torch.stack([torch.zeros_like(index), index + 1, (index + 1) % len(rim) + 1], -1)
        tri = shuffle_vertices(tri, gen).int()

        alone = [int(owner(backend, pos, tri[k : k + 1], resolution)[0, 12, 8] >= 0) for k in index]
        assert alone == [1] + [0] * (len(rim) - 1)
        first, second = peel(backend, pos, tri, resolution, 2)
        assert first[0, 12, 8] == 0 and (second == -1).all()


def test_owner_of_pixels_on_edges_matches_exact_arithmetic(backend):
    # With small integer clip coordinates and a power-of-two resolution every edge function is
    # exact in float32, so the triangle each pixel goes to is known, also on the edges of
    # triangles that cross the camera plane.
    size = 16
    gen = torch.Generator().manual_seed(10)
    ties = crossing_ties = decided = 0
    for _ in range(150):
        clip, tri = integer_homography_grid(gen)
        expected, count, tie, crossing, undecided = exact_homogeneous_owner(clip, tri, size)
        assert count.max() <= 1
        x, y, w = clip.float().unbind(-1)
        pos = torch.stack([x, y, torch.zeros_like(x), w], -1)[None]
        ids = owner(backend, pos, tri, (size, size))[0]
        assert torch.equal(ids[~undecided], expected[~undecided])
        ties += int((tie & ~undecided).sum())
        crossing_ties += int((crossing & ~undecided).sum())
        decided += int((~undecided).sum())
    assert decided > 0.95 * 150 * size * size
    assert ties > 300 and crossing_ties > 100


@pytest.mark.gpu
@pytest.mark.parametrize("resolution", RESOLUTIONS)
def test_triton_picks_the_reference_triangle_on_shared_edges(resolution):
    triton_backend = load_triton()
    gen = torch.Generator().manual_seed(3)
    scenes = full_screen_scenes(resolution, gen, max(20, resolution[1] // 16), "cuda")
    scenes["quad, 0.9"] = (screen_quad(0.9), QUAD)
    for name, (origin, u, v) in SHEETS.items():
        cols, rows, tri = sheet(origin, u, v, 6, gen)
        scenes[name] = (lattice(cols, rows, resolution, w=1.5, device="cuda"), tri)
    for name, (pos, tri) in scenes.items():
        pos, tri = pos.cuda(), tri.cuda()
        ref, _ = reference.rasterize(pos, tri, resolution, True)
        out, _ = triton_backend.rasterize(pos, tri, resolution, True)
        assert torch.equal(ref[..., 3], out[..., 3]), name


def test_gradients_on_a_shared_vertex_are_those_of_its_owner():
    # The fill rule only selects the triangle. With the selection held fixed, the outputs are
    # smooth in the vertex positions, also at a pixel centre that is a vertex of the mesh.
    resolution = (6, 6)
    cols, rows = torch.tensor([3.0, 0.4, 5.3, 4.6, 0.7]), torch.tensor([2.0, 0.6, 0.2, 5.4, 4.9])
    w = torch.tensor([1.0, 1.3, 0.8, 1.1, 1.6], dtype=torch.float64)
    pos = lattice(cols, rows, resolution, w=w, z=0.2, dtype=torch.float64).requires_grad_(True)
    tri = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]], dtype=torch.int32)
    rast, _ = reference.rasterize(pos, tri, resolution, True)
    tri_id = rast[..., 3].long() - 1
    assert tri_id[0, 2, 3] >= 0 and (tri_id >= 0).sum() > 12
    torch.testing.assert_close(rast[0, 2, 3, :3], pos.new_tensor([1.0, 0.0, 0.2]))
    assert torch.autograd.gradcheck(
        lambda p: reference.rasterize_from_ids(p, tri, tri_id, True), (pos,)
    )


# Degenerate triangles


@pytest.mark.parametrize("perspective", [False, True])
def test_collinear_triangles_are_not_drawn(backend, perspective):
    # The lattice lines pass through pixel centres, where the edge functions of these triangles
    # are rounding residue whose signs would otherwise agree on thousands of pixels.
    pos, tri = collinear_triangles(20000, torch.Generator().manual_seed(0), perspective)
    assert (owner(backend, pos, tri, (100, 100)) == -1).all()


def test_repeated_index_triangles_are_not_drawn(backend):
    gen = torch.Generator().manual_seed(1)
    xy = torch.randint(-14, 15, (1, 500, 2), generator=gen).float() / 16
    pos = torch.cat([xy, torch.zeros(1, 500, 1), torch.ones(1, 500, 1)], -1)
    i, j = torch.randint(0, 500, (2, 5000), generator=gen)
    tri = torch.cat([torch.stack([i, i, j], -1), torch.stack([i, j, i], -1)]).int()
    assert (owner(backend, pos, tri, (100, 100)) == -1).all()


def test_zero_area_triangles_do_not_hide_the_mesh_behind_them(backend):
    quad = screen_quad()
    quad[..., 2] = 0.5
    line = torch.tensor([[[-0.9, -0.9, 0.0, 1.0], [0.3, 0.3, 0.0, 1.0], [0.9, 0.9, 0.0, 1.0]]])
    tri = torch.cat([torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.int32), QUAD + 3])
    for resolution in ((100, 100), (255, 255)):
        assert (owner(backend, torch.cat([line, quad], 1), tri, resolution) >= 2).all()


@pytest.mark.parametrize("thickness", [1e-2, 1e-3, 1e-4])
def test_thin_triangles_are_drawn(backend, thickness):
    # Slivers up to 20000 times longer than wide.
    resolution = (128, 128)
    gen = torch.Generator().manual_seed(2)
    start = torch.rand(300, 2, generator=gen) * 1.6 - 0.8
    angle = torch.rand(300, generator=gen) * 6.2832
    along = torch.stack([angle.cos(), angle.sin()], -1)
    across = torch.stack([-angle.sin(), angle.cos()], -1) * thickness
    xy = torch.stack([start - along, start + along, start + along + across], 1).reshape(1, -1, 2)
    z = (torch.arange(300.0) / 300 - 0.5).repeat_interleave(3).view(1, -1, 1) * 1.6
    pos = torch.cat([xy, z, torch.ones(1, 900, 1)], -1)
    tri = torch.arange(900, dtype=torch.int32).view(300, 3)

    ids = owner(backend, pos, tri, resolution)
    # The thinner the triangle, the larger the float32 error of its barycentrics.
    expected, _, ambiguous = screen_space_rasterize(pos, tri, *resolution, margin=2e-7 / thickness)
    assert (expected[~ambiguous] >= 0).sum() > 3e5 * thickness
    assert torch.equal(ids[~ambiguous], expected[~ambiguous])


@pytest.mark.parametrize(("size", "count"), [(0.3, 100), (0.03, 1000), (0.01, 2000)])
def test_small_valid_triangles_match_screen_space_rasterizer(backend, size, count):
    resolution = (64, 56)
    pos, tri = small_triangles(count, size, torch.Generator().manual_seed(3))
    ids = owner(backend, pos, tri, resolution)
    expected, _, ambiguous = screen_space_rasterize(pos, tri, *resolution)
    assert (~ambiguous).float().mean() > 0.9 and (expected[~ambiguous] >= 0).sum() > 20
    assert torch.equal(ids[~ambiguous], expected[~ambiguous])


def test_triangles_below_float_resolution_are_not_drawn(backend):
    # Triangles a few float32 steps across have no meaningful edge functions anywhere on screen.
    pos, tri = small_triangles(5000, 3e-7, torch.Generator().manual_seed(4), perspective=False)
    assert (owner(backend, pos, tri, (64, 64)) == -1).all()


@pytest.mark.parametrize("half_width", [2.0**-5, 0.5])
@pytest.mark.parametrize(("steps", "drawn"), [(4, False), (16, False), (18, True), (64, True)])
def test_area_tolerance_is_two_epsilons(backend, half_width, steps, drawn):
    # The centre of pixel (8, 8) lies strictly inside the triangle, whose area there is
    # 2 * steps / 17 epsilons of the magnitude of the terms it is computed from. Below two
    # epsilons the triangle counts as degenerate, although its exact area is positive.
    ids = owner(backend, sliver(half_width, steps), torch.tensor([[0, 1, 2]]).int(), (16, 16))
    assert (ids[0, 8, 8] == 0) == drawn
    assert drawn or (ids == -1).all()


@pytest.mark.gpu
def test_triton_matches_reference_near_the_area_tolerance():
    # One sliver per image, around a pixel centre, with an area there between half an epsilon
    # and five epsilons of the magnitude of its terms: the backends have to draw the same ones.
    triton_backend = load_triton()
    gen = torch.Generator().manual_seed(12)
    count, eps = 3000, torch.finfo(torch.float32).eps
    centre = (torch.randint(0, 16, (count, 1, 2), generator=gen).double() * 2 + 1) / 16 - 1
    half_width = 2.0 ** -torch.randint(1, 6, (count,), generator=gen).double()
    ratio = torch.rand(count, generator=gen, dtype=torch.float64) * 4.5 + 0.5
    height = ratio * eps * (1 + centre[:, 0, 1].abs()) / 2
    zero = torch.zeros(count, dtype=torch.float64)
    offset = torch.stack([-half_width, -height, half_width, -height, zero, height], -1)
    xy = (centre + offset.view(count, 3, 2)).float()
    pos = torch.cat([xy, torch.zeros(count, 3, 1), torch.ones(count, 3, 1)], -1).cuda()
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32).cuda()

    ref, _ = reference.rasterize(pos, tri, (16, 16), True)
    out, _ = triton_backend.rasterize(pos, tri, (16, 16), True)
    assert torch.equal(ref[..., 3], out[..., 3])
    drawn = (ref[..., 3] > 0).flatten(1).any(1).float().mean()
    assert 0.3 < drawn < 0.85


@pytest.mark.parametrize("exponent", [-40, -20, -10, 10, 20, 40])
def test_coverage_does_not_depend_on_clip_space_scale(backend, exponent):
    # A power of two scales every intermediate value exactly, so nothing may change at all while
    # the products of three coordinates fit in float32: twelve orders of magnitude either way.
    gen = torch.Generator().manual_seed(5)
    resolution = (48, 48)
    cols, rows, sheet_tri = sheet(*SHEETS["facing"], 6, gen)
    scenes = [
        small_triangles(60, 0.5, gen),
        (lattice(cols, rows, resolution, device=backend[1]), sheet_tri),
        collinear_triangles(500, gen, True),
    ]
    for pos, tri in scenes:
        scaled = owner(backend, pos * 2.0**exponent, tri, resolution)
        assert torch.equal(scaled, owner(backend, pos, tri, resolution))
    # The last scene has no area at any scale.
    assert (scaled == -1).all()


@pytest.mark.gpu
def test_triton_matches_reference_on_degenerate_and_tiny_triangles():
    triton_backend = load_triton()
    gen = torch.Generator().manual_seed(6)
    scenes = [collinear_triangles(5000, gen, perspective) for perspective in (False, True)]
    scenes += [small_triangles(20000, size, gen) for size in (3e-2, 3e-4, 3e-6)]
    for pos, tri in scenes:
        pos, tri = pos.cuda(), tri.cuda()
        for resolution in ((100, 100), (37, 61)):
            ref, _ = reference.rasterize(pos, tri, resolution, True)
            out, _ = triton_backend.rasterize(pos, tri, resolution, True)
            assert torch.equal(ref[..., 3], out[..., 3])


@pytest.mark.gpu
@pytest.mark.parametrize("w_range", [(1.0, 2.0), (-1.0, 2.0)])
def test_triton_matches_reference_on_every_pixel_of_every_layer(w_range):
    # Meshes in general position, also across the camera plane and the depth range: no pixel is
    # left to rounding, because the visibility kernels perform the reference's operations.
    triton_backend = load_triton()
    gen = torch.Generator().manual_seed(9)
    xy = torch.rand(2, 900, 2, generator=gen) * 3 - 1.5
    z = torch.rand(2, 900, 1, generator=gen) * 2.4 - 1.2
    w = torch.rand(2, 900, 1, generator=gen) * (w_range[1] - w_range[0]) + w_range[0]
    pos = torch.cat([xy * w, z * w, w], -1)
    tri = torch.arange(900, dtype=torch.int32).view(300, 3)
    for resolution in ((64, 64), (37, 61), (255, 257)):
        ref = peel((reference, "cuda"), pos, tri, resolution, 4)
        out = peel((triton_backend, "cuda"), pos, tri, resolution, 4)
        assert (ref[3] >= 0).any()
        for expected, layer in zip(ref, out, strict=True):
            assert torch.equal(layer, expected)


def test_gradcheck_ignores_a_zero_area_triangle():
    pos = torch.tensor(
        [
            [
                [-0.8, -0.7, 0.2, 1.0],
                [0.9, -0.6, 0.5, 1.4],
                [0.1, 0.8, -0.3, 1.2],
                [-0.7, -0.33, -0.5, 1.0],
                [0.1, 0.07, -0.5, 1.0],
                [0.5, 0.27, -0.5, 1.0],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    tri = torch.tensor([[0, 1, 2], [3, 4, 5], [3, 3, 5]], dtype=torch.int32)
    rast, _ = reference.rasterize(pos, tri, (8, 8), True)
    assert set(rast[..., 3].unique().tolist()) == {0.0, 1.0}
    assert torch.autograd.gradcheck(lambda p: reference.rasterize(p, tri, (8, 8), True), (pos,))


# Non-finite vertices


@pytest.mark.parametrize("value", [torch.nan, torch.inf, -torch.inf])
@pytest.mark.parametrize("coordinate", range(4))
def test_triangles_with_non_finite_vertices_are_culled(backend, value, coordinate):
    module, device = backend
    resolution = (24, 24)
    pos = torch.cat([screen_quad(0.8), screen_quad(0.5)], 1)
    pos[0, 4:, 2] = -0.5
    tri = torch.cat([QUAD, QUAD + 4])
    behind = owner(backend, pos, tri[:2], resolution)
    for vertex in (4, 5, 6):
        broken = pos.clone()
        broken[0, vertex, coordinate] = value
        rast, rast_db = module.rasterize(broken.to(device), tri.to(device), resolution, True)
        assert torch.isfinite(rast).all() and torch.isfinite(rast_db).all()
        ids = rast[..., 3].long().cpu() - 1
        # Vertex 5 is part of triangle 2 only, so triangle 3 stays; the others are in both.
        intact = ids == 3
        assert intact.any() == (vertex == 5)
        assert torch.equal(ids[~intact], behind[~intact])


@pytest.mark.gpu
def test_non_finite_triangles_are_not_binned():
    load_triton()
    from diffrast.backends import _triton_rasterize

    gen = torch.Generator().manual_seed(7)
    pos = (torch.rand(1, 600, 4, generator=gen) * 2 - 1).cuda()
    tri = torch.arange(600, dtype=torch.int32).view(200, 3).cuda()
    for value in (torch.nan, torch.inf):
        broken = pos.clone()
        broken[0, ::3, 3] = value
        state = _triton_rasterize.prepare(broken, tri, (128, 128))
        _triton_rasterize.setup_stage(state)
        assert _triton_rasterize.bin_stage(state) is None
        assert (state.keys == _triton_rasterize.EMPTY).all()


# Limits
#
# The 64-bit pixel offsets and triangle lists of the Triton tile kernel have no test: they only
# matter from 2**31 pixels or triangle instances per call, whose buffers alone exceed 16 GiB.


def test_batch_of_zero_gives_empty_outputs(backend):
    module, device = backend
    pos = torch.zeros(0, 3, 4, device=device)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    rast, rast_db, state = module.rasterize_layer(pos, tri, (6, 8), True, None)
    assert rast.shape == rast_db.shape == (0, 6, 8, 4)
    second, _, _ = module.rasterize_layer(pos, tri, (6, 8), True, state)
    assert second.shape == (0, 6, 8, 4)


def test_batch_of_zero_through_the_public_api(ctx):
    pos = torch.zeros(0, 3, 4)
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    rast, rast_db = dr.rasterize(ctx, pos, tri, (6, 8))
    assert rast.shape == rast_db.shape == (0, 6, 8, 4)
    with dr.DepthPeeler(ctx, pos, tri, (6, 8)) as peeler:
        assert peeler.rasterize_next_layer()[0].shape == (0, 6, 8, 4)


def test_more_triangles_than_float32_can_index_are_rejected(ctx, triangle):
    pos, tri = triangle
    limit = 1 << 24
    assert check_rasterize_inputs(ctx, pos, tri.expand(limit, 3), (4, 4), None) == (4, 4)
    with pytest.raises(ValueError, match=r"at most 2\*\*24"):
        dr.rasterize(ctx, pos, tri.expand(limit + 1, 3), (4, 4))
    with pytest.raises(ValueError, match=r"at most 2\*\*24"):
        dr.DepthPeeler(ctx, pos, tri.expand(limit + 1, 3), (4, 4))


@pytest.mark.gpu
@pytest.mark.parametrize("size", [80001, 200003])
@pytest.mark.parametrize("tall", [False, True])
def test_very_wide_and_very_tall_images(size, tall):
    # Two opposite sides of every quad lie exactly on pixel-centre columns, or rows of a tall
    # image. The bounding box of a triangle has to keep them despite float32 rounding, which
    # grows with the image size.
    triton_backend = load_triton()
    gen = torch.Generator().manual_seed(8)
    first = torch.randint(0, size - 8, (3000,), generator=gen)
    last = first + torch.randint(1, 4, (3000,), generator=gen)
    along = torch.stack([first, last, last, first], -1).flatten()
    across = torch.tensor([-2, -2, 4, 4]).repeat(3000)
    resolution = (size, 2) if tall else (2, size)
    cols, rows = (across, along) if tall else (along, across)
    pos = lattice(cols, rows, resolution, device="cuda")
    tri = (QUAD[None] + torch.arange(3000)[:, None, None] * 4).view(-1, 3).int().cuda()

    edges = torch.zeros(size + 1)
    edges.index_add_(0, first, torch.ones(3000)).index_add_(0, last, -torch.ones(3000))
    expected = (edges.cumsum(0)[:size] > 0).expand(2, size)
    rast, _ = triton_backend.rasterize(pos, tri, resolution, True)
    covered = (rast[0, ..., 3] > 0).cpu()
    assert torch.equal(covered.T if tall else covered, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("resolution", [(64, 96), (100, 100)])
def test_float64_resolve_is_accurate_to_float64(resolution, perspective_triangle):
    # Sizes that are not powers of two, whose pixel size 2 / size is inexact in float32.
    triton_backend = load_triton()
    pos, tri = (t.cuda() for t in perspective_triangle)
    ref_pos, our_pos = (pos.double().requires_grad_(True) for _ in range(2))
    ref, ref_db = reference.rasterize(ref_pos, tri, resolution, True)
    ours, our_db = triton_backend.rasterize(our_pos, tri, resolution, True)
    # Visibility runs in float32 on Triton, so a pixel on the outline may differ.
    same = ref[..., 3] == ours[..., 3]
    assert same.float().mean() > 0.999
    ref_db[same].square().sum().backward()
    our_db[same].square().sum().backward()
    for value, expected in ((ours, ref), (our_db, ref_db)):
        torch.testing.assert_close(value[same], expected[same], rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(our_pos.grad, ref_pos.grad, rtol=1e-12, atol=1e-13)


@pytest.mark.gpu
def test_triton_double_backward_raises(perspective_triangle):
    triton_backend = load_triton()
    pos, tri = (t.cuda() for t in perspective_triangle)
    pos.requires_grad_(True)
    rast, _ = triton_backend.rasterize(pos, tri, (16, 16), True)
    with pytest.raises(RuntimeError, match="no second-order gradients"):
        torch.autograd.grad(rast[..., :3].square().sum(), pos, create_graph=True)
