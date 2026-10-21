import pytest
import torch

import diffrast as dr
from diffrast.backends import reference

QUAD = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)


def half_plane(edge, axis, dtype=torch.float32):
    """A quad covering clip space up to ``edge`` along ``axis`` (0 for x, 1 for y).

    Its other sides lie off screen, so the only silhouette on screen is one straight edge.
    """
    corners = [[-2.0, -2.0], [edge, -2.0], [edge, 2.0], [-2.0, 2.0]]
    if axis == 1:
        corners = [[y, x] for x, y in corners][::-1]
    return torch.tensor([[[x, y, 0.0, 1.0] for x, y in corners]], dtype=dtype)


def render(pos, tri, resolution, color_fn, backend=reference, boost=1.0):
    rast, _ = backend.rasterize(pos, tri, resolution, True)
    return reference.antialias(color_fn(rast), rast, pos, tri, boost)


def coverage(rast):
    return (rast[..., 3:] > 0).to(rast.dtype)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("edge", [-0.5, 0.0625, 0.3, 0.71])
def test_straight_edge_coverage_is_exact(ctx, axis, edge):
    # With an axis-aligned edge, the box-filter blend gives each pixel its exact covered area.
    height, width = 16, 24
    pos = half_plane(edge, axis)
    rast, _ = dr.rasterize(ctx, pos, QUAD, (height, width))
    out = dr.antialias(coverage(rast), rast, pos, QUAD)

    size = width if axis == 0 else height
    across = height if axis == 0 else width
    torch.testing.assert_close(out.sum(), torch.tensor(across * (edge + 1) * size / 2))


@pytest.mark.parametrize("top", [-0.2, 0.25, 0.9])
def test_slanted_edge_coverage_is_exact(ctx, top):
    # Each row recovers where the edge crosses its centre line, and summing those is exact for a
    # straight edge, so the total area is exact even when the edge is not axis aligned.
    height, width = 16, 24
    bottom = 0.3
    pos = half_plane(0.0, 0)
    pos[0, 1, 0], pos[0, 2, 0] = bottom - (top - bottom) / 2, top + (top - bottom) / 2
    rast, _ = dr.rasterize(ctx, pos, QUAD, (height, width))
    out = dr.antialias(coverage(rast), rast, pos, QUAD)
    middle = (bottom + top) / 2
    torch.testing.assert_close(out.sum(), torch.tensor(height * (middle + 1) * width / 2))


@pytest.mark.parametrize("axis", [0, 1])
def test_straight_edge_gradient_is_edge_length(axis):
    # Moving the edge by one NDC unit changes the covered area by its length times size / 2.
    height, width = 16, 24
    pos = half_plane(0.3, axis, torch.float64).requires_grad_(True)
    render(pos, QUAD, (height, width), coverage).sum().backward()

    size, across = (width, height) if axis == 0 else (height, width)
    grad = pos.grad[0, [1, 2], axis].sum()
    torch.testing.assert_close(grad, torch.tensor(across * size / 2, dtype=torch.float64))


@pytest.mark.parametrize("axis", [0, 1])
def test_continuous_when_edge_crosses_pixel_centre(axis):
    # Pixel ownership flips as the edge passes a pixel centre, but the blended image must not jump.
    size = 16
    centre = (2 * 5 + 1) / size - 1
    images = [
        render(half_plane(centre + offset, axis, torch.float64), QUAD, (size, size), coverage)
        for offset in (-1e-9, 1e-9)
    ]
    torch.testing.assert_close(images[0], images[1], atol=1e-6, rtol=0)
    aliased = [
        coverage(
            reference.rasterize(
                half_plane(centre + o, axis, torch.float64), QUAD, (size, size), True
            )[0]
        )
        for o in (-1e-9, 1e-9)
    ]
    assert not torch.equal(aliased[0], aliased[1])


def test_interior_edges_are_left_alone(ctx):
    # The two halves of the quad have different colours, but their shared diagonal lies inside a
    # continuous surface and is not a silhouette.
    pos = half_plane(2.0, 0)
    rast, _ = dr.rasterize(ctx, pos, QUAD, (12, 12))
    color = torch.stack([rast[..., 3] == 1, rast[..., 3] == 2], -1).float()
    torch.testing.assert_close(dr.antialias(color, rast, pos, QUAD), color)


def test_image_without_edges_is_unchanged(ctx, triangle):
    pos, tri = triangle
    rast, _ = dr.rasterize(ctx, pos * torch.tensor([0.0, 0.0, 1.0, 1.0]), tri, (8, 8))
    color = torch.rand(1, 8, 8, 3)
    torch.testing.assert_close(dr.antialias(color, rast, pos, tri), color)


class TestSilhouetteEdges:
    def test_flat_quad(self):
        sil = reference._silhouette_edges(half_plane(0.3, 0).double(), QUAD.long())
        # Edge k is opposite vertex k, so the diagonal (0, 2) is edge 1 of [0, 1, 2] and edge 2
        # of [0, 2, 3].
        diagonal = torch.zeros(2, 3, dtype=torch.bool)
        diagonal[0, 1] = diagonal[1, 2] = True
        assert torch.equal(sil[0], ~diagonal)

    def test_folded_quad(self):
        pos = half_plane(0.3, 0).double()
        # Move vertex 3 across the diagonal so the second triangle folds over the first.
        pos[0, 3, :2] = torch.tensor([0.5, -1.5])
        sil = reference._silhouette_edges(pos, QUAD.long())
        assert sil[0, 0, 1] and sil[0, 1, 2]

    def test_non_manifold_edge(self):
        tri = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 2, 4]])
        pos = torch.cat([half_plane(0.3, 0), torch.tensor([[[0.0, 1.0, 0.0, 1.0]]])], 1)
        sil = reference._silhouette_edges(pos.double(), tri)
        assert sil[0, 0, 1] and sil[0, 1, 2] and sil[0, 2, 2]


def test_front_triangle_owns_the_occlusion_edge(ctx):
    # A small triangle in front of a background quad: only the front triangle's vertices should
    # receive silhouette gradients from the boundary between them.
    front = [[-0.4, -0.3, -0.5, 1.0], [0.5, -0.4, -0.5, 1.0], [0.1, 0.6, -0.5, 1.0]]
    back = half_plane(2.0, 0)[0].tolist()
    pos = torch.tensor([front + [[x, y, 0.5, w] for x, y, _, w in back]], requires_grad=True)
    tri = torch.tensor([[0, 1, 2], [3, 4, 5], [3, 5, 6]], dtype=torch.int32)
    rast, _ = dr.rasterize(ctx, pos, tri, (32, 32))
    color = torch.where(rast[..., 3:] == 1, 1.0, 0.0)
    dr.antialias(color, rast, pos, tri).sum().backward()

    assert pos.grad[0, :3].abs().sum() > 0
    assert (pos.grad[0, 3:] == 0).all()


def test_pos_gradient_boost_scales_only_the_gradient(perspective_triangle):
    pos, tri = (t.double() for t in perspective_triangle)
    grads, images = [], []
    for boost in (1.0, 3.0):
        p = pos.clone().requires_grad_(True)
        out = render(p, tri, (16, 16), coverage, boost=boost)
        out.sum().backward()
        grads.append(p.grad)
        images.append(out.detach())
    torch.testing.assert_close(images[0], images[1])
    torch.testing.assert_close(grads[1], 3 * grads[0])


def test_color_gradcheck(perspective_triangle):
    pos, tri = (t.double() for t in perspective_triangle)
    rast, _ = reference.rasterize(pos, tri, (12, 12), True)
    color = torch.rand(1, 12, 12, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda c: reference.antialias(c, rast, pos, tri, 1.0), (color,))


class TestFiniteDifferences:
    """The antialiased image is continuous in the vertex positions; its gradient must match."""

    def check(self, pos, tri, resolution, color_fn, eps=1e-6, rtol=1e-4):
        weights = torch.rand(
            1, *resolution, 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )

        def loss(p):
            return (render(p, tri, resolution, color_fn) * weights).sum()

        p = pos.clone().requires_grad_(True)
        loss(p).backward()
        numeric = torch.zeros_like(pos)
        for index in torch.cartesian_prod(*(torch.arange(n) for n in pos.shape)):
            index = tuple(index.tolist())
            if index[-1] == 2:
                continue  # z only selects visibility; it has no silhouette gradient.
            step = torch.zeros_like(pos)
            step[index] = eps
            numeric[index] = (loss(pos + step) - loss(pos - step)) / (2 * eps)
        torch.testing.assert_close(
            p.grad[..., [0, 1, 3]], numeric[..., [0, 1, 3]], rtol=rtol, atol=1e-6
        )
        assert numeric.abs().sum() > 0

    def test_constant_colour_triangle(self, perspective_triangle):
        pos, tri = (t.double() for t in perspective_triangle)
        self.check(pos, tri, (16, 16), lambda rast: coverage(rast).expand(-1, -1, -1, 3))

    def test_interpolated_colours(self, perspective_triangle):
        pos, tri = (t.double() for t in perspective_triangle)
        colours = torch.tensor(
            [[1.0, 0.0, 0.2], [0.1, 0.9, 0.3], [0.4, 0.2, 1.0]], dtype=torch.float64
        )

        def color_fn(rast):
            return reference.interpolate(colours, rast, tri, None, [])[0]

        self.check(pos, tri, (16, 16), color_fn)

    def test_overlapping_triangles(self):
        pos = torch.tensor(
            [[[-0.8, -0.7, 0.3, 1.0], [0.9, -0.5, 0.2, 1.2], [-0.2, 0.8, 0.4, 1.1],
              [-0.5, 0.6, -0.2, 1.0], [0.1, -0.9, -0.1, 1.3], [0.8, 0.7, -0.3, 0.9]]],
            dtype=torch.float64,
        )  # fmt: skip
        tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        shades = torch.tensor([[0.2, 0.5, 0.9], [0.9, 0.4, 0.1]], dtype=torch.float64)

        def color_fn(rast):
            ids = rast[..., 3].long()
            return torch.where((ids > 0)[..., None], shades[(ids - 1).clamp(min=0)], 0.0)

        self.check(pos, tri, (20, 20), color_fn)


@pytest.mark.gpu
def test_gpu_matches_cpu(perspective_triangle):
    pos, tri = perspective_triangle
    cpu_ctx, gpu_ctx = dr.RasterizeContext("cpu"), dr.RasterizeContext("cuda")
    rast, _ = dr.rasterize(cpu_ctx, pos, tri, (32, 32))
    color = torch.rand(1, 32, 32, 3, generator=torch.Generator().manual_seed(1))
    cpu = dr.antialias(color, rast, pos, tri)
    gpu = dr.antialias(*(t.to(gpu_ctx.device) for t in (color, rast, pos, tri)))
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-5, rtol=1e-5)
