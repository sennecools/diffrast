"""Cube-map lookups, checked against a brute-force sampler written from the OpenGL face table."""

import itertools
import math

import pytest
import torch

import diffrast as dr
from diffrast import backends
from diffrast.backends import reference
from diffrast.ops import texture_construct_mip

FILTERS = ("nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear")
MIP_FILTERS = FILTERS[2:]
CORNERS = list(itertools.product((-1.0, 1.0), repeat=3))


def face_coords(direction, face=None):
    """``(face, s, t)`` of a direction: the major-axis table of the OpenGL specification.

    With ``face`` given, the direction is projected onto the plane of that face instead.
    """
    x, y, z = direction
    ax, ay, az = abs(x), abs(y), abs(z)
    if face is None:
        if ax >= ay and ax >= az:
            face = 0 if x > 0 else 1
        elif ay >= az:
            face = 2 if y > 0 else 3
        else:
            face = 4 if z > 0 else 5
    sc, tc, ma = [(-z, -y, ax), (z, -y, ax), (x, z, ay), (x, -z, ay), (x, -y, az), (-x, -y, az)][
        face
    ]
    return face, (sc / ma + 1) / 2, (tc / ma + 1) / 2


def face_direction(face, s, t):
    """A direction that projects to ``(s, t)`` on ``face``; the inverse of the table."""
    sc, tc = 2 * s - 1, 2 * t - 1
    return [
        (1.0, -tc, -sc),
        (-1.0, -tc, sc),
        (sc, 1.0, tc),
        (sc, -1.0, -tc),
        (sc, -tc, 1.0),
        (-sc, -tc, -1.0),
    ][face]


def texel(tex, face, row, col):
    """Texel of a ``[6, size, size, channels]`` map, continued across one edge.

    A texel one step outside a face is looked up by projecting its centre back onto the cube.
    Returns ``None`` beyond a corner, where no face continues.
    """
    size = tex.shape[1]
    inside_row, inside_col = 0 <= row < size, 0 <= col < size
    if not inside_row and not inside_col:
        return None
    if not (inside_row and inside_col):
        centre = face_direction(face, (col + 0.5) / size, (row + 0.5) / size)
        face, s, t = face_coords(centre)
        row, col = min(int(t * size), size - 1), min(int(s * size), size - 1)
    return tex[face, row, col]


def brute_nearest(tex, direction):
    size = tex.shape[1]
    face, s, t = face_coords(direction)
    return tex[face, min(int(t * size), size - 1), min(int(s * size), size - 1)]


def brute_linear(tex, direction):
    """Seamless bilinear lookup; a tap missing beyond a corner is the mean of the other three."""
    size = tex.shape[1]
    face, s, t = face_coords(direction)
    x, y = s * size - 0.5, t * size - 0.5
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0
    taps = [
        (texel(tex, face, y0 + j, x0 + i), (fx if i else 1 - fx) * (fy if j else 1 - fy))
        for j in (0, 1)
        for i in (0, 1)
    ]
    present = [value for value, _ in taps if value is not None]
    fill = sum(present) / len(present)
    return sum((fill if value is None else value) * weight for value, weight in taps)


def brute_level_of_detail(direction, derivs, size, step=1e-5):
    """Level of detail from finite differences of the face coordinates along screen x and y."""
    face = face_coords(direction)[0]
    longest = 0.0
    for axis in (0, 1):
        ahead = [d + step * da[axis] for d, da in zip(direction, derivs, strict=True)]
        behind = [d - step * da[axis] for d, da in zip(direction, derivs, strict=True)]
        _, s1, t1 = face_coords(ahead, face)
        _, s0, t0 = face_coords(behind, face)
        longest = max(longest, math.hypot(s1 - s0, t1 - t0) * size / (2 * step))
    return math.log2(longest)


def brute_lods(dirs, uv_da, size):
    return torch.tensor(
        [
            brute_level_of_detail(d, da, size)
            for d, da in zip(
                dirs.detach().view(-1, 3).tolist(),
                uv_da.detach().view(-1, 3, 2).tolist(),
                strict=True,
            )
        ],
        dtype=torch.float64,
    )


def box_levels(tex):
    """Mip levels of a ``[6, size, size, channels]`` map with a power-of-two size."""
    levels = [tex]
    while levels[-1].shape[1] > 1:
        half = levels[-1].shape[1] // 2
        levels.append(levels[-1].view(6, half, 2, half, 2, -1).mean((2, 4)))
    return levels


def brute_mipmapped(levels, direction, lod, blend):
    lod = min(max(lod, 0.0), len(levels) - 1.0)
    if not blend:
        return brute_linear(levels[math.floor(lod + 0.5)], direction)
    lower = min(math.floor(lod), len(levels) - 2) if len(levels) > 1 else 0
    upper = min(lower + 1, len(levels) - 1)
    frac = lod - lower
    return (1 - frac) * brute_linear(levels[lower], direction) + frac * brute_linear(
        levels[upper], direction
    )


def random_cube(size, channels=2, batch=1, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 6, size, size, channels, generator=gen, dtype=dtype)


def random_directions(count, seed=0, dtype=torch.float64):
    """Unnormalised directions: half anywhere, half close to the edges and corners of the cube."""
    gen = torch.Generator().manual_seed(seed)
    anywhere = torch.randn(count // 2, 3, generator=gen, dtype=dtype)
    anywhere = anywhere * (torch.rand(count // 2, 1, generator=gen, dtype=dtype) * 5 + 0.1)
    near = torch.randint(0, 2, (count - count // 2, 3), generator=gen).to(dtype) * 2 - 1
    near = near + 0.2 * torch.randn(near.shape, generator=gen, dtype=dtype)
    near[::2, 0] *= torch.rand(near[::2].shape[0], generator=gen, dtype=dtype)
    return torch.cat([anywhere, near]).view(1, 1, count, 3)


def sample(tex, dirs, filter_mode="linear", uv_da=None, **kwargs):
    """The reference backend, which also accepts float64."""
    kwargs.setdefault("max_mip_level", None)
    return reference.texture(
        tex, dirs, uv_da, filter_mode, "cube", kwargs.pop("max_mip_level"), **kwargs
    )


def edge_points(steps=(-0.97, -0.5, 0.0, 0.3, 0.8, 0.999)):
    """Points on the twelve edges of the cube, each with the axis whose face it leaves."""
    points = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        free = 3 - a - b
        for sign_a, sign_b, along in itertools.product((-1.0, 1.0), (-1.0, 1.0), steps):
            point = [0.0, 0.0, 0.0]
            point[a], point[b], point[free] = sign_a, sign_b, along
            points.append((point, a))
    return points


@pytest.mark.parametrize("filter_mode", FILTERS)
def test_constant_map_returns_the_constant(filter_mode):
    tex = torch.full((1, 6, 8, 8, 3), 0.375)
    dirs = random_directions(200, dtype=torch.float32)
    uv_da = torch.randn(1, 1, 200, 6, generator=torch.Generator().manual_seed(1))
    out = dr.texture(
        tex,
        dirs,
        uv_da if "mipmap" in filter_mode else None,
        filter_mode=filter_mode,
        boundary_mode="cube",
    )
    torch.testing.assert_close(out, torch.full_like(out, 0.375), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
def test_axis_directions_return_face_centres(filter_mode):
    tex = random_cube(5, dtype=torch.float32)
    axes = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    dirs = torch.tensor(axes, dtype=torch.float32).view(1, 1, 6, 3) * 2.5
    out = dr.texture(tex, dirs, filter_mode=filter_mode, boundary_mode="cube")
    torch.testing.assert_close(out[0, 0], tex[0, :, 2, 2])


@pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
@pytest.mark.parametrize("size", [1, 4, 7])
def test_face_orientation_texel_by_texel(filter_mode, size):
    tex = random_cube(size, dtype=torch.float32)
    centres = [
        face_direction(face, (col + 0.5) / size, (row + 0.5) / size)
        for face, row, col in itertools.product(range(6), range(size), range(size))
    ]
    dirs = torch.tensor(centres, dtype=torch.float32).view(1, 6, size * size, 3) * 0.3
    out = dr.texture(tex, dirs, filter_mode=filter_mode, boundary_mode="cube")
    torch.testing.assert_close(out.view(tex.shape), tex, atol=1e-5, rtol=1e-5)


def test_faces_are_ordered_and_oriented_as_in_opengl():
    # Spot checks written out by hand: on +x, s runs towards -z and t towards -y.
    tex = torch.arange(6 * 4 * 4, dtype=torch.float32).view(1, 6, 4, 4, 1)
    dirs = torch.tensor(
        [[1.0, 0.75, 0.75], [1.0, -0.75, -0.75], [-0.25, 1.0, 0.75], [0.25, 0.75, -1.0]]
    ).view(1, 1, 4, 3)
    out = dr.texture(tex, dirs, filter_mode="nearest", boundary_mode="cube")
    expected = [tex[0, 0, 0, 0], tex[0, 0, 3, 3], tex[0, 2, 3, 1], tex[0, 5, 0, 1]]
    assert out.flatten().tolist() == [value.item() for value in expected]


@pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
@pytest.mark.parametrize("size", [1, 2, 5, 8])
def test_matches_brute_force(filter_mode, size):
    tex = random_cube(size)
    dirs = random_directions(600)
    out = sample(tex, dirs, filter_mode)
    brute = brute_nearest if filter_mode == "nearest" else brute_linear
    expected = torch.stack([brute(tex[0], d) for d in dirs[0, 0].tolist()])
    torch.testing.assert_close(out[0, 0], expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("size", [1, 3, 8])
def test_continuous_across_every_edge(size):
    # Any map is continuous under bilinear filtering, not only a smooth one, so a random map
    # and a tiny step make the test strict.
    tex = random_cube(size, seed=2)
    step = 1e-9
    inside, outside = [], []
    for point, axis in edge_points():
        nudged = list(point)
        nudged[axis] *= 1 + step
        inside.append(nudged)
        nudged = list(point)
        nudged[axis] *= 1 - step
        outside.append(nudged)
    dirs = torch.tensor([inside, outside], dtype=torch.float64).view(1, 2, -1, 3)
    faces = [face_coords(d)[0] for d in dirs.view(-1, 3).tolist()]
    half = len(faces) // 2
    assert all(a != b for a, b in zip(faces[:half], faces[half:], strict=True))
    out = sample(tex, dirs)
    torch.testing.assert_close(out[0, 0], out[0, 1], atol=1e-6, rtol=0)


@pytest.mark.parametrize("filter_mode", MIP_FILTERS)
@pytest.mark.parametrize("level", [0.0, 1.3, 2.0])
def test_mip_levels_are_continuous_across_every_edge(filter_mode, level):
    # With the level set by a bias it is the same on both sides of an edge, and every level is
    # filtered across the edge like the base level. A level derived from uv_da is the footprint
    # on one face, which is not the same on the next.
    tex = random_cube(8, seed=49)
    step = 1e-9
    sides = []
    for scale in (1 + step, 1 - step):
        nudged = [list(point) for point, _ in edge_points()]
        for direction, (_, axis) in zip(nudged, edge_points(), strict=True):
            direction[axis] *= scale
        sides.append(nudged)
    dirs = torch.tensor(sides, dtype=torch.float64).view(1, 2, -1, 3)
    bias = torch.full(dirs.shape[:3], level, dtype=torch.float64)
    out = sample(tex, dirs, filter_mode, mip_level_bias=bias)
    torch.testing.assert_close(out[0, 0], out[0, 1], atol=1e-6, rtol=0)


@pytest.mark.parametrize("size", [1, 3, 8])
def test_continuous_around_every_corner(size):
    tex = random_cube(size, seed=3)
    step = 1e-9
    dirs = []
    for corner in CORNERS:
        for axis in range(3):
            nudged = [c * (1 - step) for c in corner]
            nudged[axis] = corner[axis]
            dirs.append(nudged)
    dirs = torch.tensor(dirs, dtype=torch.float64).view(1, 8, 3, 3)
    faces = [face_coords(d)[0] for d in dirs.view(-1, 3).tolist()]
    assert all(len(set(faces[i : i + 3])) == 3 for i in range(0, 24, 3))
    out = sample(tex, dirs)
    torch.testing.assert_close(out[0, :, 0], out[0, :, 1], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[0, :, 0], out[0, :, 2], atol=1e-6, rtol=0)


def test_corner_direction_is_the_mean_of_the_three_corner_texels():
    size = 4
    tex = random_cube(size, seed=4)
    dirs = torch.tensor(CORNERS, dtype=torch.float64).view(1, 1, 8, 3)
    out = sample(tex, dirs)
    for index, corner in enumerate(CORNERS):
        texels = []
        for axis in range(3):
            # Step just inside each of the three faces that meet at the corner.
            inner = [c * 0.999 for c in corner]
            inner[axis] = corner[axis]
            texels.append(brute_nearest(tex[0], inner))
        torch.testing.assert_close(out[0, 0, index], sum(texels) / 3)


@pytest.mark.parametrize("filter_mode", MIP_FILTERS)
@pytest.mark.parametrize("with_bias", [False, True])
def test_mipmapped_matches_brute_force(filter_mode, with_bias):
    size, count = 8, 300
    tex = random_cube(size, seed=5)
    dirs = random_directions(count, seed=6)
    gen = torch.Generator().manual_seed(7)
    scale = 10 ** (torch.rand(1, 1, count, 1, generator=gen, dtype=torch.float64) * 2 - 1.5)
    uv_da = torch.randn(1, 1, count, 6, generator=gen, dtype=torch.float64) * scale
    uv_da = uv_da * dirs.abs().amax(-1, keepdim=True)
    bias = torch.rand(1, 1, count, generator=gen, dtype=torch.float64) * 3 - 1.5
    out = sample(tex, dirs, filter_mode, uv_da, mip_level_bias=bias if with_bias else None)

    levels = box_levels(tex[0])
    expected, lods = [], []
    for i, direction in enumerate(dirs[0, 0].tolist()):
        lod = brute_level_of_detail(direction, uv_da[0, 0, i].view(3, 2).tolist(), size)
        lod += bias[0, 0, i].item() if with_bias else 0.0
        # Where the level is rounded, a lookup at the rounding point may go either way.
        close_call = filter_mode == "linear-mipmap-nearest" and abs(lod % 1 - 0.5) < 1e-6
        lods.append(lod)
        blend = filter_mode == "linear-mipmap-linear"
        expected.append(
            out[0, 0, i] if close_call else brute_mipmapped(levels, direction, lod, blend)
        )
    assert min(lods) < 0 and max(lods) > 3 and sum(0 < lod < 3 for lod in lods) > count // 3
    torch.testing.assert_close(out[0, 0], torch.stack(expected), atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("scale", [1.0, 0.01, 300.0])
def test_footprint_of_four_texels_uses_level_two(scale):
    # At the centre of +z, s = (x / z + 1) / 2, so a footprint of four of the sixteen texels
    # along a face takes a step of 8 / 16 in x / z per pixel.
    size = 16
    tex = random_cube(size, seed=8)
    dirs = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).view(1, 1, 1, 3) * scale
    uv_da = torch.tensor([0.5, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=torch.float64).view(1, 1, 1, 6)
    out = sample(tex, dirs, "linear-mipmap-linear", uv_da * scale)
    # The centre of level two is the mean of its four middle texels: the middle 8 x 8 texels.
    torch.testing.assert_close(out[0, 0, 0], tex[0, 4, 4:12, 4:12].mean((0, 1)))
    by_bias = sample(tex, dirs, "linear-mipmap-linear", mip_level_bias=torch.full((1, 1, 1), 2.0))
    torch.testing.assert_close(out, by_bias.to(out.dtype))


def test_max_mip_level_and_bias():
    tex = random_cube(8, seed=9, dtype=torch.float32)
    dirs = random_directions(60, dtype=torch.float32)
    far = torch.full((1, 1, 60), 9.0)
    kwargs = {"boundary_mode": "cube"}
    coarsest = dr.texture(tex, dirs, mip_level_bias=far, **kwargs)
    one_texel_faces = box_levels(tex[0])[-1]
    expected = torch.stack([brute_linear(one_texel_faces, d) for d in dirs[0, 0].tolist()])
    torch.testing.assert_close(coarsest[0, 0], expected)
    limited = dr.texture(tex, dirs, mip_level_bias=far, max_mip_level=1, **kwargs)
    level_one = dr.texture(tex, dirs, mip_level_bias=torch.full((1, 1, 60), 1.0), **kwargs)
    assert torch.equal(limited, level_one)
    disabled = dr.texture(tex, dirs, mip_level_bias=far, max_mip_level=0, **kwargs)
    assert torch.equal(disabled, dr.texture(tex, dirs, **kwargs))


class TestMipStacks:
    def lookup(self, tex, **kwargs):
        dirs = random_directions(80, seed=10, dtype=torch.float32)
        gen = torch.Generator().manual_seed(11)
        uv_da = torch.randn(1, 1, 80, 6, generator=gen) * 0.3
        return dr.texture(tex, dirs, uv_da, boundary_mode="cube", **kwargs)

    @pytest.mark.parametrize("size", [8, 6, 5])
    def test_constructed_stack_matches_building_internally(self, size):
        tex = random_cube(size, batch=1, seed=12, dtype=torch.float32)
        mip = texture_construct_mip(tex, cube_mode=True)
        assert [level.shape[2] for level in mip.levels][:2] == [(size + 1) // 2, (size + 3) // 4]
        assert all(level.shape[:2] == (1, 6) for level in mip.levels)
        assert torch.equal(self.lookup(tex, mip=mip), self.lookup(tex))
        short = texture_construct_mip(tex, max_mip_level=1, cube_mode=True)
        assert torch.equal(self.lookup(tex, mip=short), self.lookup(tex, max_mip_level=1))

    def test_faces_are_filtered_one_by_one(self):
        tex = random_cube(8, seed=13, dtype=torch.float32)
        levels = texture_construct_mip(tex, cube_mode=True).levels
        for level, expected in zip(levels, box_levels(tex[0])[1:], strict=True):
            torch.testing.assert_close(level[0], expected)

    def test_custom_list_of_the_same_levels(self):
        tex = random_cube(8, seed=14, dtype=torch.float32)
        levels = [level[None] for level in box_levels(tex[0])[1:]]
        torch.testing.assert_close(self.lookup(tex, mip=levels), self.lookup(tex))
        torch.testing.assert_close(
            self.lookup(tex, mip=levels[:1]), self.lookup(tex, max_mip_level=1)
        )

    def test_prefiltered_specular_lookup_as_in_pipelines(self):
        # A list of independently trained cube maps, one per roughness, selected by a bias.
        gen = torch.Generator().manual_seed(15)
        spec = [
            torch.rand(16 >> i, 16 >> i, 6, 3, generator=gen).permute(2, 0, 1, 3) for i in range(4)
        ]
        spec = [level.contiguous().requires_grad_() for level in spec]
        dirs = random_directions(2 * 5 * 7, seed=16, dtype=torch.float32).view(2, 5, 7, 3)
        bias = torch.rand(2, 5, 7, generator=gen) * 3
        out = dr.texture(
            spec[0][None],
            dirs,
            mip=[level[None] for level in spec[1:]],
            mip_level_bias=bias,
            filter_mode="linear-mipmap-linear",
            boundary_mode="cube",
        )
        assert out.shape == (2, 5, 7, 3)
        out.sum().backward()
        assert all(level.grad.abs().sum() > 0 for level in spec)
        expected = [
            brute_mipmapped([lv.detach().double() for lv in spec], d, b, True)
            for d, b in zip(dirs.view(-1, 3).tolist(), bias.flatten().tolist(), strict=True)
        ]
        expected = torch.stack(expected).view(2, 5, 7, 3).float()
        torch.testing.assert_close(out.detach(), expected, atol=1e-5, rtol=1e-5)

    def test_environment_lookup_as_in_pipelines(self):
        cubemap = random_cube(8, channels=3, seed=17, dtype=torch.float32)[0]
        dirs = random_directions(64, seed=18, dtype=torch.float32).view(8, 8, 3)
        plain = dr.texture(cubemap[None], dirs[None], boundary_mode="cube")
        linear = dr.texture(
            cubemap[None] * 0.25,
            dirs[None].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )
        assert plain.shape == (1, 8, 8, 3)
        torch.testing.assert_close(linear, plain * 0.25)


class TestGradients:
    def inputs(self, seed, size=4, level_offset=(0.25, 0.75)):
        """Inputs whose level of detail is a whole level in 0 to 2 plus ``level_offset``.

        That keeps it away from the levels where the lookup has a kink or a jump.
        """
        gen = torch.Generator().manual_seed(seed)
        tex = torch.randn(1, 6, size, size, 2, generator=gen, dtype=torch.float64)
        dirs = random_directions(12, seed=seed).view(1, 3, 4, 3)
        length = dirs.abs().amax(-1, keepdim=True)
        uv_da = (torch.rand(1, 3, 4, 6, generator=gen, dtype=torch.float64) * 0.4 + 0.2) * length
        low, high = level_offset
        offset = torch.rand(1, 3, 4, generator=gen, dtype=torch.float64) * (high - low) + low
        level = torch.randint(0, 3, (1, 3, 4), generator=gen) + offset
        bias = level - brute_lods(dirs, uv_da, size).view(1, 3, 4)
        return [t.requires_grad_() for t in (tex, dirs, uv_da, bias)]

    def test_linear(self):
        tex, dirs, _, _ = self.inputs(19)
        assert torch.autograd.gradcheck(lambda t, d: sample(t, d), (tex, dirs))

    def test_nearest_texture_gradient(self):
        tex, dirs, _, _ = self.inputs(20)
        assert torch.autograd.gradcheck(lambda t: sample(t, dirs.detach(), "nearest"), (tex,))

    @pytest.mark.parametrize("seed", [21, 22])
    def test_trilinear(self, seed):
        tex, dirs, uv_da, bias = self.inputs(seed, size=8)
        assert torch.autograd.gradcheck(
            lambda t, d, a, b: sample(t, d, "linear-mipmap-linear", a, mip_level_bias=b),
            (tex, dirs, uv_da, bias),
        )

    def test_mipmap_nearest(self):
        tex, dirs, uv_da, bias = self.inputs(23, size=8, level_offset=(0.6, 1.4))
        uv_da, bias = uv_da.detach(), bias.detach()
        assert torch.autograd.gradcheck(
            lambda t, d: sample(t, d, "linear-mipmap-nearest", uv_da, mip_level_bias=bias),
            (tex, dirs),
        )

    def test_custom_mip_levels(self):
        tex, dirs, _, _ = self.inputs(24)
        level = random_cube(2, seed=25).requires_grad_()
        gen = torch.Generator().manual_seed(46)
        bias = (
            torch.rand(1, 3, 4, generator=gen, dtype=torch.float64) * 0.5 + 0.25
        ).requires_grad_()
        assert torch.autograd.gradcheck(
            lambda t, m, d, b: sample(t, d, "linear-mipmap-linear", mip_level_bias=b, mip=[m]),
            (tex, level, dirs, bias),
        )

    def test_constructed_stack_carries_gradients_to_the_texture(self):
        tex = random_cube(8, seed=26, dtype=torch.float32).requires_grad_()
        dirs = random_directions(40, seed=27, dtype=torch.float32)
        bias = torch.rand(1, 1, 40, generator=torch.Generator().manual_seed(28)) * 3
        kwargs = {"mip_level_bias": bias, "boundary_mode": "cube"}
        (expected,) = torch.autograd.grad(dr.texture(tex, dirs, **kwargs).sum(), tex)
        mip = texture_construct_mip(tex, cube_mode=True)
        (grad,) = torch.autograd.grad(dr.texture(tex, dirs, mip=mip, **kwargs).sum(), tex)
        torch.testing.assert_close(grad, expected)


class TestInvalidDirections:
    BAD = (
        (0.0, 0.0, 0.0),
        (float("nan"), 1.0, 0.0),
        (float("inf"), 0.0, 1.0),
        (1.0, -float("inf"), 2.0),
    )

    @pytest.mark.parametrize("filter_mode", FILTERS)
    def test_zeros_and_no_gradient(self, filter_mode):
        tex = random_cube(4, seed=29, dtype=torch.float32).abs() + 1
        good = random_directions(4, seed=30, dtype=torch.float32).view(4, 3)
        dirs = torch.cat([torch.tensor(self.BAD), good]).view(1, 2, 4, 3)
        uv_da = torch.randn(1, 2, 4, 6, generator=torch.Generator().manual_seed(31)) * 0.2
        for leaf in (tex, dirs, uv_da):
            leaf.requires_grad_()
        mip = "mipmap" in filter_mode
        out = dr.texture(
            tex, dirs, uv_da if mip else None, filter_mode=filter_mode, boundary_mode="cube"
        )
        assert (out[0, 0] == 0).all()
        assert (out[0, 1] > 0.99).all()

        out[0, 0].sum().backward(retain_graph=True)
        assert (tex.grad == 0).all()
        assert dirs.grad is None or (dirs.grad == 0).all()
        assert uv_da.grad is None or (uv_da.grad == 0).all()
        out.sum().backward()
        assert tex.grad.abs().sum() > 0
        for grad in (tex.grad, dirs.grad, uv_da.grad):
            assert grad is None or torch.isfinite(grad).all()
        assert dirs.grad is None or (dirs.grad[0, 0] == 0).all()

    def lookup_with_derivatives(self, first, filter_mode="linear-mipmap-linear", bias=None):
        """Four valid directions; the first has the derivatives ``first``, the others 0.05."""
        tex = random_cube(8, seed=32, dtype=torch.float32)
        dirs = torch.tensor(
            [[0.2, 0.3, 1.0], [1.0, 0.1, -0.4], [-0.3, 1.0, 0.2], [0.1, -0.2, -1.0]]
        ).view(1, 1, 4, 3)
        uv_da = torch.full((1, 1, 4, 6), 0.05)
        uv_da[0, 0, 0] = torch.tensor(first)
        leaves = [t.requires_grad_() for t in (tex, dirs, uv_da)]
        out = dr.texture(
            *leaves, mip_level_bias=bias, filter_mode=filter_mode, boundary_mode="cube"
        )
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(47))
        grads = torch.autograd.grad((out * weights).nansum(), leaves, allow_unused=True)
        return out, [
            torch.zeros_like(t) if g is None else g for g, t in zip(grads, leaves, strict=True)
        ]

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    @pytest.mark.parametrize("component", range(6))
    def test_nan_derivatives_of_a_valid_direction_give_nan_and_no_gradient(
        self, filter_mode, component
    ):
        first = [0.05] * 6
        first[component] = float("nan")
        out, (grad_tex, grad_dirs, grad_da) = self.lookup_with_derivatives(first, filter_mode)
        assert out[0, 0, 0].isnan().all() and not out[0, 0, 1:].isnan().any()
        for grad in (grad_tex, grad_dirs, grad_da):
            assert torch.isfinite(grad).all()
        assert (grad_dirs[0, 0, 0] == 0).all() and (grad_da[0, 0, 0] == 0).all()
        assert grad_dirs[0, 0, 1:].abs().sum() > 0 and grad_tex.abs().sum() > 0

    def test_nan_bias_gives_nan_and_no_gradient(self):
        bias = torch.tensor([float("nan"), 0.5, 0.5, 0.5]).view(1, 1, 4)
        out, grads = self.lookup_with_derivatives([0.05] * 6, bias=bias)
        assert out[0, 0, 0].isnan().all() and not out[0, 0, 1:].isnan().any()
        assert all(torch.isfinite(grad).all() for grad in grads)
        assert (grads[1][0, 0, 0] == 0).all()

    @pytest.mark.parametrize(
        "first",
        [
            [float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -float("inf"), 0.0, 0.0],
            [float("inf")] * 6,
            [float("inf"), -float("inf")] * 3,
            [3e38, 3e38, -3e38, 3e38, 3e38, 3e38],
            [1e30, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    def test_huge_derivatives_select_the_coarsest_level(self, first):
        out, (grad_tex, grad_dirs, grad_da) = self.lookup_with_derivatives(first)
        far = torch.full((1, 1, 4), 50.0)
        coarsest, (_, coarsest_grad_dirs, _) = self.lookup_with_derivatives([0.05] * 6, bias=far)
        torch.testing.assert_close(out[0, 0, 0], coarsest[0, 0, 0])
        for grad in (grad_tex, grad_dirs, grad_da):
            assert torch.isfinite(grad).all()
        assert (grad_da[0, 0, 0] == 0).all()
        torch.testing.assert_close(grad_dirs[0, 0, 0], coarsest_grad_dirs[0, 0, 0])
        assert grad_dirs[0, 0, 0].abs().sum() > 0


class TestDirectionLength:
    """Only the direction of ``uv`` matters, whatever its length."""

    DIRECTION = (0.375, -1.0, 0.625)

    def lookup(self, length, filter_mode="linear"):
        tex = random_cube(4, seed=48, dtype=torch.float32)
        dirs = (torch.tensor(self.DIRECTION) * length).view(1, 1, 1, 3).requires_grad_()
        uv_da = torch.tensor([0.9, 0.3, -0.6, 0.75, 0.45, -0.3]).view(1, 1, 1, 6) * length
        mip = "mipmap" in filter_mode
        out = dr.texture(
            tex, dirs, uv_da if mip else None, filter_mode=filter_mode, boundary_mode="cube"
        )
        (grad,) = torch.autograd.grad(out.sum(), dirs)
        return out, grad

    @pytest.mark.parametrize("filter_mode", ["linear", "linear-mipmap-linear"])
    @pytest.mark.parametrize("length", [2.0**-120, 1e-30, 1e-20, 1e-6, 1e6, 1e20, 1e30])
    def test_value_and_gradient_at_any_length(self, filter_mode, length):
        expected, expected_grad = self.lookup(1.0, filter_mode)
        out, grad = self.lookup(length, filter_mode)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)
        assert expected_grad.abs().sum() > 0
        torch.testing.assert_close(grad * length, expected_grad, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("filter_mode", ["linear", "linear-mipmap-linear"])
    @pytest.mark.parametrize("length", [2.0**-127, 2.0**-130, 2.0**-146])
    def test_subnormal_length_counts_as_the_zero_vector(self, filter_mode, length):
        # A subnormal vector has lost the bits that give it a direction, and the gradient of a
        # lookup, which grows with the reciprocal of the length, would overflow.
        out, grad = self.lookup(length, filter_mode)
        assert (out == 0).all() and (grad == 0).all()
        out, grad = self.lookup(2.0**-126, filter_mode)
        torch.testing.assert_close(out, self.lookup(1.0, filter_mode)[0], atol=1e-6, rtol=1e-5)
        assert not grad.isnan().any() and grad.abs().sum() > 0


class TestBatches:
    def test_shared_map_matches_per_view_maps(self):
        tex = random_cube(4, seed=33, dtype=torch.float32)
        dirs = random_directions(36, seed=34, dtype=torch.float32).view(3, 3, 4, 3)
        shared = dr.texture(tex, dirs, boundary_mode="cube")
        per_view = dr.texture(
            tex.expand(3, -1, -1, -1, -1).contiguous(), dirs, boundary_mode="cube"
        )
        assert torch.equal(shared, per_view)

    @pytest.mark.parametrize("filter_mode", FILTERS)
    def test_per_view_maps_are_independent(self, filter_mode):
        tex = random_cube(4, batch=3, seed=35, dtype=torch.float32)
        dirs = random_directions(36, seed=36, dtype=torch.float32).view(3, 3, 4, 3)
        bias = torch.rand(3, 3, 4, generator=torch.Generator().manual_seed(37)) * 2
        kwargs = {"filter_mode": filter_mode, "boundary_mode": "cube"}
        if "mipmap" in filter_mode:
            kwargs["mip_level_bias"] = bias
        out = dr.texture(tex, dirs, **kwargs)
        for view in range(3):
            if "mipmap" in filter_mode:
                kwargs["mip_level_bias"] = bias[view : view + 1]
            single = dr.texture(tex[view : view + 1], dirs[view : view + 1], **kwargs)
            torch.testing.assert_close(out[view : view + 1], single)

    def test_custom_level_shared_across_a_batch_of_maps(self):
        tex = random_cube(4, batch=2, seed=38, dtype=torch.float32)
        level = random_cube(2, seed=39, dtype=torch.float32)
        dirs = random_directions(24, seed=40, dtype=torch.float32).view(2, 3, 4, 3)
        kwargs = {"mip_level_bias": torch.full((2, 3, 4), 0.6), "boundary_mode": "cube"}
        shared = dr.texture(tex, dirs, mip=[level], **kwargs)
        repeated = dr.texture(tex, dirs, mip=[level.expand(2, -1, -1, -1, -1)], **kwargs)
        assert torch.equal(shared, repeated)


class TestValidation:
    def test_cube_map_needs_the_cube_boundary_mode(self):
        tex, dirs = torch.zeros(1, 6, 4, 4, 3), torch.zeros(1, 2, 2, 3)
        with pytest.raises(ValueError, match=r'shape of a cube map.*boundary_mode="cube"'):
            dr.texture(tex, dirs)
        with pytest.raises(ValueError, match=r"shape of a cube map.*cube_mode=True"):
            texture_construct_mip(tex)

    def test_cube_boundary_mode_needs_a_cube_map(self):
        with pytest.raises(ValueError, match="needs tex to be a cube map"):
            dr.texture(torch.zeros(1, 4, 4, 3), torch.zeros(1, 2, 2, 3), boundary_mode="cube")
        with pytest.raises(ValueError, match="needs tex to be a cube map"):
            texture_construct_mip(torch.zeros(1, 4, 4, 3), cube_mode=True)

    @pytest.mark.parametrize("shape", [(1, 5, 4, 4, 3), (1, 6, 4, 5, 3)])
    def test_six_square_faces(self, shape):
        with pytest.raises(ValueError, match="six square faces"):
            dr.texture(torch.zeros(shape), torch.zeros(1, 2, 2, 3), boundary_mode="cube")

    def test_coordinates_are_directions(self):
        tex = torch.zeros(1, 6, 4, 4, 3)
        with pytest.raises(ValueError, match="uv must have shape"):
            dr.texture(tex, torch.zeros(1, 2, 2, 2), boundary_mode="cube")
        with pytest.raises(ValueError, match="uv_da must have shape"):
            dr.texture(tex, torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, 2, 4), boundary_mode="cube")
        with pytest.raises(ValueError, match="uv must have shape"):
            dr.texture(torch.zeros(1, 4, 4, 3), torch.zeros(1, 2, 2, 3))

    @pytest.mark.parametrize(
        "level, match",
        [
            (torch.zeros(1, 2, 2, 3), "needs mip level 1 to be a cube map"),
            (torch.zeros(1, 6, 2, 3, 3), "mip level 1 must have six square faces"),
            (torch.zeros(1, 6, 2, 2, 1), "mip level 1 has 1 channels"),
            (torch.zeros(2, 6, 2, 2, 3), "mip level 1 has batch size 2"),
        ],
    )
    def test_mip_levels_are_cube_maps(self, level, match):
        tex, dirs = torch.zeros(1, 6, 4, 4, 3), torch.zeros(1, 2, 2, 3)
        with pytest.raises(ValueError, match=match):
            dr.texture(tex, dirs, mip=[level], boundary_mode="cube")

    def test_stack_of_another_kind(self):
        tex, dirs = torch.zeros(1, 6, 4, 4, 3), torch.zeros(1, 2, 2, 3)
        stack = texture_construct_mip(torch.zeros(1, 4, 4, 3))
        with pytest.raises(ValueError, match="cube_mode=False"):
            dr.texture(tex, dirs, mip=stack, boundary_mode="cube")

    def test_stack_of_another_map(self):
        dirs = torch.zeros(1, 2, 2, 3)
        stack = texture_construct_mip(torch.zeros(1, 6, 8, 8, 3), cube_mode=True)
        with pytest.raises(ValueError, match="constructed for a texture of shape"):
            dr.texture(torch.zeros(1, 6, 4, 4, 3), dirs, mip=stack, boundary_mode="cube")


@pytest.mark.gpu
class TestDevices:
    """Cube maps run the same PyTorch code on every device and backend."""

    def scene(self):
        tex = random_cube(8, channels=3, batch=2, seed=41, dtype=torch.float32)
        dirs = random_directions(2 * 6 * 20, seed=42, dtype=torch.float32).view(2, 6, 20, 3)
        dirs[0, 0, 0] = 0.0
        gen = torch.Generator().manual_seed(43)
        uv_da = torch.randn(2, 6, 20, 6, generator=gen) * 0.3
        bias = torch.rand(2, 6, 20, generator=gen) * 2 - 0.5
        return tex, dirs, uv_da, bias

    def run(self, inputs, filter_mode):
        tex, dirs, uv_da, bias = (t.clone().requires_grad_() for t in inputs)
        mip = "mipmap" in filter_mode
        out = dr.texture(
            tex,
            dirs,
            uv_da if mip else None,
            bias if mip else None,
            filter_mode=filter_mode,
            boundary_mode="cube",
        )
        weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(44))
        (out * weights.to(out.device)).sum().backward()
        grads = [
            torch.zeros_like(t) if t.grad is None else t.grad for t in (tex, dirs, uv_da, bias)
        ]
        return [out.detach().cpu(), *(grad.cpu() for grad in grads)]

    @pytest.mark.parametrize("filter_mode", FILTERS)
    @pytest.mark.parametrize("backend", ["auto", "triton", "reference"])
    def test_same_on_cpu_and_gpu(self, filter_mode, backend, monkeypatch):
        pytest.importorskip("triton")
        inputs = self.scene()
        monkeypatch.setattr(backends, "_selected", "reference")
        on_cpu = self.run(inputs, filter_mode)
        monkeypatch.setattr(backends, "_selected", backend)
        assert dr.get_backend_name("cuda", "texture") == (
            "reference" if backend == "reference" else "triton"
        )
        on_gpu = self.run([t.cuda() for t in inputs], filter_mode)
        torch.testing.assert_close(on_gpu[0], on_cpu[0], atol=1e-5, rtol=1e-5)
        for gpu_grad, cpu_grad in zip(on_gpu[1:], on_cpu[1:], strict=True):
            torch.testing.assert_close(gpu_grad, cpu_grad, atol=1e-4, rtol=1e-3)

    def test_set_backend_triton(self):
        pytest.importorskip("triton")
        tex, dirs, uv_da, _ = (t.cuda() for t in self.scene())
        selected = backends._selected
        try:
            dr.set_backend("reference")
            expected = dr.texture(tex, dirs, uv_da, boundary_mode="cube")
            dr.set_backend("triton")
            out = dr.texture(tex, dirs, uv_da, boundary_mode="cube")
            mip = texture_construct_mip(tex, cube_mode=True)
            with_stack = dr.texture(tex, dirs, uv_da, mip=mip, boundary_mode="cube")
        finally:
            dr.set_backend(selected)
        assert out.device.type == "cuda"
        assert torch.equal(out, expected)
        assert torch.equal(with_stack, expected)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -3e38])
    def test_derivatives_that_are_not_finite(self, bad, monkeypatch):
        pytest.importorskip("triton")
        inputs = self.scene()
        inputs[1][0, 0, 0] = 1.0
        inputs[2][0, 0, :4, :3] = bad
        inputs[2][0, 1, :4, 3:] = -bad
        monkeypatch.setattr(backends, "_selected", "reference")
        on_cpu = self.run(inputs, "linear-mipmap-linear")
        monkeypatch.setattr(backends, "_selected", "triton")
        on_gpu = self.run([t.cuda() for t in inputs], "linear-mipmap-linear")
        undefined = on_cpu[0].isnan().all(-1)
        assert undefined[0, :2, :4].all() if math.isnan(bad) else not undefined.any()
        assert not undefined[0, 2:].any() and not undefined[1].any()
        torch.testing.assert_close(on_gpu[0], on_cpu[0], atol=1e-5, rtol=1e-5, equal_nan=True)
        for gpu_grad, cpu_grad in zip(on_gpu[1:], on_cpu[1:], strict=True):
            assert torch.isfinite(cpu_grad).all()
            torch.testing.assert_close(gpu_grad, cpu_grad, atol=1e-4, rtol=1e-3)

    def test_large_lookup(self):
        # A 512 x 512 lookup into 256 x 256 faces, all in one vectorised pass.
        gen = torch.Generator().manual_seed(45)
        tex = torch.rand(1, 6, 256, 256, 3, generator=gen).cuda()
        dirs = torch.randn(1, 512, 512, 3, generator=gen).cuda()
        uv_da = (torch.randn(1, 512, 512, 6, generator=gen) * 0.02).cuda()
        out = dr.texture(tex, dirs, uv_da, boundary_mode="cube")
        assert out.shape == (1, 512, 512, 3)
        assert torch.isfinite(out).all() and 0.3 < out.mean() < 0.7
