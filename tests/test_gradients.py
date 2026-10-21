"""Gradient checks against finite differences, in float64 on the reference backend."""

import pytest
import torch

import diffrast as dr
from diffrast.backends import reference

RES = (8, 8)


@pytest.fixture
def scene(perspective_triangle):
    pos, tri = perspective_triangle
    # A second, overlapping triangle so that gradients route to the visible one only.
    other = torch.tensor([[[-0.7, 0.8, -0.2, 1.2], [0.9, 0.6, 0.4, 1.0], [-0.1, -0.9, 0.1, 1.1]]])
    pos = torch.cat([pos, other], 1).double()
    tri = torch.cat([tri, tri + 3])
    return pos, tri


def test_rasterize_gradcheck(scene):
    pos, tri = scene
    pos.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: reference.rasterize(p, tri, RES, True), (pos,))


def test_rasterize_without_grad_db(scene):
    pos, tri = scene
    pos.requires_grad_(True)
    rast, rast_db = reference.rasterize(pos, tri, RES, False)
    assert rast.requires_grad
    assert not rast_db.requires_grad


def test_interpolate_gradcheck(scene):
    pos, tri = scene
    rast, rast_db = reference.rasterize(pos, tri, RES, True)
    attr = torch.randn(1, 6, 3, dtype=torch.float64, requires_grad=True)
    rast = rast.detach().requires_grad_(True)
    rast_db = rast_db.detach().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda a, r, d: reference.interpolate(a, r, tri, d, [0, 2]),
        (attr, rast, rast_db),
    )


def test_rasterize_interpolate_chain_gradcheck(scene):
    pos, tri = scene
    pos.requires_grad_(True)
    attr = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)

    def render(p, a):
        rast, rast_db = reference.rasterize(p, tri, RES, True)
        return reference.interpolate(a, rast, tri, rast_db, [0, 1])

    assert torch.autograd.gradcheck(render, (pos, attr))


def test_rast_db_matches_subpixel_shift(perspective_triangle):
    pos, tri = perspective_triangle
    pos = pos.double()
    height, width = 16, 16
    _, rast_db = reference.rasterize(pos, tri, (height, width), True)

    # Moving the triangle by +d in NDC is equivalent to sampling it at -d.
    eps = 1e-5
    for axis, size, cols in ((0, width, [0, 2]), (1, height, [1, 3])):
        shift = torch.zeros(4, dtype=torch.float64)
        shift[axis] = eps
        plus, _ = reference.rasterize(pos - shift * pos[..., 3:], tri, (height, width), True)
        minus, _ = reference.rasterize(pos + shift * pos[..., 3:], tri, (height, width), True)
        both = (plus[..., 3] > 0) & (minus[..., 3] > 0)
        fd = (plus[..., :2] - minus[..., :2]) / (2 * eps) * (2 / size)
        torch.testing.assert_close(rast_db[..., cols][both], fd[both], atol=1e-6, rtol=1e-5)


def test_gradient_reaches_only_visible_vertices(ctx):
    pos = torch.tensor(
        [
            [
                [-0.9, -0.9, -0.5, 1.0],
                [0.9, -0.9, -0.5, 1.0],
                [0.0, 0.9, -0.5, 1.0],
                [-0.9, -0.9, 0.5, 1.0],
                [0.9, -0.9, 0.5, 1.0],
                [0.0, 0.9, 0.5, 1.0],
            ]
        ],
        requires_grad=True,
    )
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    rast, _ = dr.rasterize(ctx, pos, tri, (16, 16))
    rast[..., :3].sum().backward()
    assert pos.grad[0, :3].abs().sum() > 0
    assert (pos.grad[0, 3:] == 0).all()
