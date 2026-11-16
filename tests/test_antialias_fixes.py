"""Antialias gradients on axis-aligned geometry, shared positions and edges that tie."""

import math

import pytest
import torch

import diffrast as dr
from diffrast import backends
from diffrast.backends import reference
from diffrast.backends._autograd import first_order

# Not square, so that the diagonals of the cube's faces miss the pixel centres. Which of the
# two triangles of a face a pixel belongs to would otherwise change with the smallest step.
RESOLUTION = (20, 24)


@pytest.fixture
def triton_backend():
    pytest.importorskip("triton")
    from diffrast.backends import triton

    return triton


def quad_grid(columns, rows, dtype=torch.float64):
    """An axis-aligned grid of quads at ``w = 1`` with shared vertices.

    The outline lies between pixel centres and pixel borders at ``RESOLUTION``, so the image is
    differentiable in the vertex positions.
    """
    xs = torch.linspace(-0.57, 0.63, columns + 1, dtype=dtype)
    ys = torch.linspace(-0.43, 0.52, rows + 1, dtype=dtype)
    x, y = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([x, y], -1).reshape(-1, 2)
    pos = torch.cat([xy, torch.zeros_like(xy[:, :1]), torch.ones_like(xy[:, :1])], -1)[None]
    tri = []
    for row in range(rows):
        for column in range(columns):
            a = row * (columns + 1) + column
            b, c, d = a + 1, a + columns + 2, a + columns + 1
            tri += [[a, b, c], [a, c, d]]
    return pos, torch.tensor(tri, dtype=torch.int32)


def upright_cube(angle, dtype=torch.float64):
    """A cube turned about the vertical axis and seen in perspective.

    Its vertical edges project to exactly vertical lines, and without a turn the edges of the
    front face are exactly horizontal too.
    """
    signs = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    x, y, z = (torch.tensor(signs, dtype=dtype) * 0.45).unbind(-1)
    x, z = math.cos(angle) * x + math.sin(angle) * z, math.cos(angle) * z - math.sin(angle) * x
    pos = torch.stack([1.8 * x, 1.8 * y, z, z + 2.5], -1)[None]
    quads = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    tri = [[a, b, c] for a, b, c, _ in quads] + [[a, c, d] for a, _, c, d in quads]
    return pos, torch.tensor(tri, dtype=torch.int32)


SCENES = {
    "strip": lambda: quad_grid(2, 1),
    "grid": lambda: quad_grid(3, 2),
    "cube": lambda: upright_cube(0.0),
    "turned cube": lambda: upright_cube(0.4),
}


def random(*shape, like, seed=0):
    values = torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return values.to(like)


def shade(rast, tri):
    """Smooth vertex colours, which jump at silhouettes but not at edges inside a surface."""
    colours = random(int(tri.max()) + 1, 3, like=rast)
    return reference.interpolate(colours, rast, tri, None, [])[0]


def render(backend, pos, tri, resolution):
    rast, _ = reference.rasterize(pos, tri, resolution, True)
    return backend.antialias(shade(rast, tri), rast, pos, tri, 1.0)


def position_gradient(backend, pos, tri, resolution):
    pos = pos.clone().requires_grad_(True)
    out = render(backend, pos, tri, resolution)
    (out * random(*out.shape, like=out, seed=1)).sum().backward()
    return out.detach(), pos.grad


@pytest.mark.parametrize("scene", SCENES)
def test_axis_aligned_edges_give_finite_gradients(ctx, scene):
    pos, tri = SCENES[scene]()
    pos = pos.float().requires_grad_(True)
    rast, _ = dr.rasterize(ctx, pos, tri, RESOLUTION)
    out = dr.antialias(shade(rast, tri), rast, pos, tri)
    out.square().sum().backward()

    assert out.isfinite().all()
    assert pos.grad.isfinite().all()
    assert pos.grad.abs().sum() > 0


@pytest.mark.parametrize("scene", SCENES)
def test_axis_aligned_gradients_match_finite_differences(scene):
    pos, tri = SCENES[scene]()
    w = random(1, *RESOLUTION, 3, like=pos, seed=1)

    def loss(p):
        return (render(reference, p, tri, RESOLUTION) * w).sum()

    _, grad = position_gradient(reference, pos, tri, RESOLUTION)
    eps = 1e-6
    numeric = torch.zeros_like(pos)
    for vertex in range(pos.shape[1]):
        for axis in (0, 1, 3):
            step = torch.zeros_like(pos)
            step[0, vertex, axis] = eps
            numeric[0, vertex, axis] = (loss(pos + step) - loss(pos - step)) / (2 * eps)

    assert numeric.abs().sum() > 0
    torch.testing.assert_close(grad[..., [0, 1, 3]], numeric[..., [0, 1, 3]], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("scene", ["strip", "cube"])
def test_axis_aligned_gradcheck(scene):
    pos, tri = SCENES[scene]()
    rast, _ = reference.rasterize(pos, tri, RESOLUTION, True)
    color = shade(rast, tri)[..., :2].contiguous()
    pos.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: reference.antialias(color, rast, p, tri, 1.0), (pos,))


@pytest.mark.gpu
@pytest.mark.parametrize("scene", SCENES)
def test_axis_aligned_gradients_match_triton(triton_backend, scene):
    pos, tri = (t.cuda() for t in SCENES[scene]())
    ref_out, ref_grad = position_gradient(reference, pos, tri, RESOLUTION)
    out, grad = position_gradient(triton_backend, pos, tri, RESOLUTION)
    torch.testing.assert_close(out, ref_out, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(grad, ref_grad, atol=1e-9, rtol=1e-9)

    out32, grad32 = position_gradient(triton_backend, pos.float(), tri, RESOLUTION)
    assert grad32.isfinite().all()
    torch.testing.assert_close(out32.double(), ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad32.double(), ref_grad, atol=1e-3, rtol=1e-3)


class TestSharedPositions:
    """A 2D ``pos`` is shared by all batch items, as in nvdiffrast's range mode."""

    def scene(self, dtype=torch.float64, device="cpu"):
        pos, tri = upright_cube(0.4, dtype)
        pos, tri = pos.to(device), tri.to(device)
        rast, _ = reference.rasterize(pos.expand(3, -1, -1), tri, (16, 20), True)
        return pos[0], tri, rast, random(3, 16, 20, 2, like=rast)

    def check(self, antialias, pos, tri, rast, color):
        w = random(*color.shape, like=color, seed=2)
        shared = pos.clone().requires_grad_(True)
        out_shared = antialias(color, rast, shared, tri)
        (out_shared * w).sum().backward()
        batched = pos.expand(3, -1, -1).clone().requires_grad_(True)
        out_batched = antialias(color, rast, batched, tri)
        (out_batched * w).sum().backward()

        assert (out_shared != color).any()
        tolerance = {"atol": 1e-12, "rtol": 1e-12} if pos.dtype == torch.float64 else {}
        torch.testing.assert_close(out_shared, out_batched, **tolerance)
        assert shared.grad.shape == pos.shape
        torch.testing.assert_close(shared.grad, batched.grad.sum(0), **tolerance)
        return out_shared.detach(), shared.grad

    def test_matches_expanded_positions(self):
        pos, tri, rast, color = self.scene(torch.float32)
        self.check(dr.antialias, pos, tri, rast, color)

    def test_gradcheck(self):
        pos, tri, rast, color = self.scene()
        pos.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda p, c: reference.antialias(c, rast, p, tri, 1.0), (pos, color.requires_grad_())
        )

    def test_rejects_other_ranks(self):
        pos, tri, rast, color = self.scene(torch.float32)
        with pytest.raises(ValueError, match=r"pos must have shape \[batch, vertices, 4\] or"):
            dr.antialias(color, rast, pos[0], tri)
        with pytest.raises(ValueError, match="pos batch size 2"):
            dr.antialias(color, rast, pos.expand(2, -1, -1), tri)

    @pytest.mark.gpu
    def test_triton_matches_reference(self, triton_backend):
        pos, tri, rast, color = self.scene(device="cuda")
        ref = self.check(lambda *args: reference.antialias(*args, 1.0), pos, tri, rast, color)
        out = self.check(lambda *args: triton_backend.antialias(*args, 1.0), pos, tri, rast, color)
        for ours, theirs in zip(out, ref, strict=True):
            torch.testing.assert_close(ours, theirs, atol=1e-9, rtol=1e-9)

    @pytest.mark.gpu
    def test_triton_gradcheck(self, triton_backend):
        pos, tri, rast, color = self.scene(device="cuda")
        pos.requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda p, c: triton_backend.antialias(c, rast, p, tri, 1.0),
            (pos, color.requires_grad_()),
        )


def quad(corners, scale, device="cpu"):
    """Two triangles through four ``(x, y)`` corners, with all clip coordinates scaled."""
    pos = torch.tensor([[[x, y, 0.1, 1.0] for x, y in corners]], device=device) * scale
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32, device=device)
    return pos, tri


def diamond(scale, device="cpu", radius=0.7):
    """A square standing on a corner, whose four edges run at 45 degrees on a square image."""
    corners = [[0.0, -radius], [radius, 0.0], [0.0, radius], [-radius, 0.0]]
    return quad(corners, scale, device)


def antialiased_coverage(backend, pos, tri, resolution):
    rast, _ = reference.rasterize(pos, tri, resolution, True)
    return backend.antialias((rast[..., 3:] > 0).float(), rast, pos, tri, 1.0)


@pytest.mark.parametrize("scale", [2.3, 0.37, 7.1])
def test_diagonal_edges_do_not_depend_on_clip_space_scale(scale):
    # Scaling all four clip coordinates leaves the geometry unchanged, and with a scale of one
    # the arithmetic is exact. Rounding at other scales must not move an edge between the
    # horizontal and the vertical pixel pairs from one pixel to the next.
    resolution = (256, 256)
    exact = antialiased_coverage(reference, *diamond(1.0), resolution)
    scaled = antialiased_coverage(reference, *diamond(scale), resolution)
    torch.testing.assert_close(scaled, exact, atol=1e-4, rtol=0)
    area = 2 * 0.7**2 * resolution[0] * resolution[1] / 4
    torch.testing.assert_close(scaled.sum(), torch.tensor(area), rtol=1e-4, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("resolution", [(256, 256), (100, 100), (64, 96)])
@pytest.mark.parametrize("scale", [1.0, 2.3, 0.37, 7.1])
def test_diagonal_edges_match_between_backends(triton_backend, resolution, scale):
    # At 100 x 100 the edges also pass exactly through pixel centres.
    pos, tri = diamond(scale, "cuda")
    grads, images = [], []
    for backend in (reference, triton_backend):
        p = pos.clone().requires_grad_(True)
        out = antialiased_coverage(backend, p, tri, resolution)
        out.sum().backward()
        images.append(out.detach())
        grads.append(p.grad)
    torch.testing.assert_close(images[1], images[0], atol=1e-6, rtol=0)
    assert (grads[1] - grads[0]).abs().max() < 1e-4 * grads[0].abs().max()


# Outlines whose edges all pass through pixel centres, while the diagonal shared by their two
# triangles passes through none inside the outline: corners, resolution and area in pixels.
CENTRED_SHAPES = {
    "diamond": ([[0.0, -0.7], [0.7, 0.0], [0.0, 0.7], [-0.7, 0.0]], (100, 100), 2450.0),
    "small diamond": ([[0.0, -0.5], [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]], (64, 64), 512.0),
    "rectangle": ([[-0.51, -0.31], [0.51, -0.31], [0.51, 0.31], [-0.51, 0.31]], (100, 100), 1581.0),
}
CLIP_SPACE_SCALES = (1.0, 2.3, 0.37)


class TestEdgesThroughPixelCentres:
    """A pixel whose centre lies on a silhouette is half covered, whichever side of the edge the
    rasterizer gives it to and however the rasterizer rounds."""

    @pytest.fixture(
        params=[
            ("cpu", "reference"),
            pytest.param(("cuda", "reference"), marks=pytest.mark.gpu),
            pytest.param(("cuda", "triton"), marks=pytest.mark.gpu),
        ],
        ids=["reference", "reference on gpu", "triton"],
    )
    def device(self, request):
        """A device, with one backend selected for all ops."""
        name, backend = request.param
        if backend == "triton":
            pytest.importorskip("triton")
        previous = backends._selected
        dr.set_backend(backend)
        yield torch.device(name)
        backends._selected = previous

    def expected(self, corners, resolution, device):
        """The coverage of each pixel, a half on the outline, and the pixels to compare: all but
        those at a vertex, where two silhouette edges meet and the blend is only approximate."""
        height, width = resolution
        x = ((torch.arange(width, dtype=torch.float64) * 2 + 1) / width - 1)[None, :]
        y = ((torch.arange(height, dtype=torch.float64) * 2 + 1) / height - 1)[:, None]
        corners = torch.tensor(corners, dtype=torch.float64)
        # Counter-clockwise outlines: the interior is on the left of every edge.
        sides = torch.stack(
            [
                (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
                for (x0, y0), (x1, y1) in zip(corners, corners.roll(-1, 0), strict=True)
            ]
        )
        inside = (sides > 1e-9).all(0)
        outline = (sides > -1e-9).all(0) & ~inside
        coverage = torch.where(outline, 0.5, inside.double())
        pixels = torch.stack(
            [
                torch.maximum((x - cx).abs() * width / 2, (y - cy).abs() * height / 2)
                for cx, cy in corners
            ]
        )
        away = pixels.min(0).values > 1.5
        assert (outline & away).sum() > 20
        return coverage.float().to(device), away.to(device)

    def check(self, image, corners, resolution, area):
        coverage, away = self.expected(corners, resolution, image.device)
        torch.testing.assert_close(image[away], coverage[away], atol=1e-3, rtol=0)
        assert abs(float(image.sum()) - area) <= 2.5

    @pytest.mark.parametrize("scale", CLIP_SPACE_SCALES)
    @pytest.mark.parametrize("shape", CENTRED_SHAPES)
    def test_rasterize_then_antialias(self, device, shape, scale):
        corners, resolution, area = CENTRED_SHAPES[shape]
        pos, tri = quad(corners, scale, device)
        pos.requires_grad_(True)
        rast, _ = dr.rasterize(dr.RasterizeContext(device), pos, tri, resolution)
        image = dr.antialias((rast[..., 3:] > 0).float(), rast, pos, tri)
        image.sum().backward()
        self.check(image[0, ..., 0].detach(), corners, resolution, area)
        assert pos.grad.isfinite().all() and (pos.grad != 0).any()

    @pytest.mark.parametrize("scale", CLIP_SPACE_SCALES)
    @pytest.mark.parametrize("shape", CENTRED_SHAPES)
    def test_rast_from_a_rasterizer_that_rounds_differently(self, device, shape, scale):
        # In float64 the edge functions round differently, so pixels on an edge are given to
        # the triangle or to the background in another pattern than float32 would.
        corners, resolution, area = CENTRED_SHAPES[shape]
        pos, tri = quad(corners, scale, device)
        rast, _ = reference.rasterize(pos.double(), tri, resolution, True)
        rast = rast.float()
        image = dr.antialias((rast[..., 3:] > 0).float(), rast, pos, tri)
        self.check(image[0, ..., 0], corners, resolution, area)


class TestFirstOrderGuard:
    """``first_order`` refuses a backward pass that is asked for a differentiable result."""

    class Double(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        @first_order("double")
        def backward(ctx, grad):
            return grad * 2

    def test_first_order_gradients_pass(self):
        x = torch.ones(3, requires_grad=True)
        self.Double.apply(x).sum().backward(retain_graph=True)
        with torch.enable_grad():
            (grad,) = torch.autograd.grad(self.Double.apply(x).square().sum(), x)
        assert torch.equal(x.grad, torch.full((3,), 2.0))
        assert torch.equal(grad, torch.full((3,), 8.0)) and not grad.requires_grad

    def test_differentiable_gradient_is_refused(self):
        x = torch.ones(3, requires_grad=True)
        with pytest.raises(RuntimeError, match="double on the Triton backend has no second-order"):
            torch.autograd.grad(self.Double.apply(x).sum(), x, create_graph=True)


def squared(out, leaf):
    return out.square().sum()


def mixed(out, leaf):
    return out.square().sum() + leaf.pow(3).sum()


def linear(out, leaf):
    return out.sum()


@pytest.mark.gpu
class TestDoubleBackward:
    """The Triton backward passes are kernels, which autograd cannot differentiate again.

    Every way of asking for a gradient of a gradient has to raise, also the ones where the
    second pass could reach its inputs without going through the op again and would return a
    gradient that lacks the op's share.
    """

    def scene(self):
        pos, tri = (t.cuda() for t in upright_cube(0.4, torch.float32))
        rast, rast_db = reference.rasterize(pos, tri, (16, 16), True)
        return pos, tri, rast, rast_db

    def antialias(self, backend):
        pos, tri, rast, _ = self.scene()
        pos.requires_grad_(True)
        color = shade(rast, tri).requires_grad_(True)
        return backend.antialias(color, rast, pos, tri, 1.0), pos

    def interpolate(self, backend):
        _, tri, rast, rast_db = self.scene()
        attr = random(8, 3, like=rast).requires_grad_(True)
        out, out_da = backend.interpolate(attr, rast, tri, rast_db, [0, 1])
        return torch.cat([out, out_da], -1), attr

    @pytest.mark.parametrize("op", ["antialias", "interpolate"])
    @pytest.mark.parametrize("loss", [squared, mixed, linear], ids=lambda loss: loss.__name__)
    def test_differentiable_gradient_is_refused(self, triton_backend, op, loss):
        out, leaf = getattr(self, op)(triton_backend)
        with pytest.raises(RuntimeError, match=f"{op} on the Triton backend has no second-order"):
            torch.autograd.grad(loss(out, leaf), leaf, create_graph=True)

    @pytest.mark.filterwarnings("ignore:Using backward.. with create_graph=True")
    @pytest.mark.parametrize("op", ["antialias", "interpolate"])
    def test_backward_with_create_graph_is_refused(self, triton_backend, op):
        out, leaf = getattr(self, op)(triton_backend)
        with pytest.raises(RuntimeError, match="set_backend\\('reference'\\)"):
            squared(out, leaf).backward(create_graph=True)

    @pytest.mark.parametrize("op", ["antialias", "interpolate"])
    def test_first_order_gradients_are_unaffected(self, triton_backend, op):
        out, leaf = getattr(self, op)(triton_backend)
        with torch.enable_grad():
            (grad,) = torch.autograd.grad(mixed(out, leaf), leaf, retain_graph=True)
        ref_out, ref_leaf = getattr(self, op)(reference)
        (expected,) = torch.autograd.grad(mixed(ref_out, ref_leaf), ref_leaf)
        assert not grad.requires_grad
        torch.testing.assert_close(grad, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("op", ["antialias", "interpolate"])
    def test_reference_backend_gives_second_order_gradients(self, op):
        out, leaf = getattr(self, op)(reference)
        (grad,) = torch.autograd.grad(mixed(out, leaf), leaf, create_graph=True)
        (second,) = torch.autograd.grad(grad.square().sum(), leaf)
        assert second.isfinite().all() and (second != 0).any()
