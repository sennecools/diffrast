import pytest
import torch

import diffrast as dr
from helpers import screen_space_layers


def peel(ctx, pos, tri, resolution, layers):
    with dr.DepthPeeler(ctx, pos, tri, resolution) as peeler:
        return [peeler.rasterize_next_layer() for _ in range(layers)]


def flat_triangle(z, w=1.0):
    return [[-0.9 * w, -0.9 * w, z * w, w], [0.9 * w, -0.9 * w, z * w, w], [0.0, 0.9 * w, z * w, w]]


def test_layers_go_from_front_to_back(ctx):
    pos = torch.tensor([flat_triangle(0.5) + flat_triangle(-0.3) + flat_triangle(0.1)])
    tri = torch.arange(9, dtype=torch.int32).view(3, 3)
    layers = peel(ctx, pos, tri, (16, 16), 4)

    centre = [rast[0, 8, 8] for rast, _ in layers]
    assert [c[3].item() for c in centre] == [2, 3, 1, 0]
    assert [c[2].item() for c in centre[:3]] == pytest.approx([-0.3, 0.1, 0.5])


def test_first_layer_matches_rasterize(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    (first, first_db), _ = peel(ctx, pos, tri, (16, 16), 2)
    rast, rast_db = dr.rasterize(ctx, pos, tri, (16, 16))
    torch.testing.assert_close(first, rast)
    torch.testing.assert_close(first_db, rast_db)


def test_equal_depths_are_returned_in_index_order(ctx):
    pos = torch.tensor([flat_triangle(0.25) * 3])
    tri = torch.arange(9, dtype=torch.int32).view(3, 3)[[2, 0, 1]]
    layers = peel(ctx, pos, tri, (8, 8), 4)
    covered = layers[0][0][..., 3] > 0
    for k, expected in enumerate([1, 2, 3, 0]):
        assert (layers[k][0][..., 3][covered] == expected).all()


def test_layers_match_sorted_coverage(ctx):
    gen = torch.Generator().manual_seed(0)
    n = 30
    xy = torch.rand(2, 3 * n, 2, generator=gen) * 2.4 - 1.2
    z = torch.rand(2, 3 * n, 1, generator=gen) * 1.8 - 0.9
    w = torch.rand(2, 3 * n, 1, generator=gen) + 1.0
    pos = torch.cat([xy * w, z * w, w], -1)
    tri = torch.arange(3 * n, dtype=torch.int32).view(n, 3)

    expected, ambiguous = screen_space_layers(pos, tri, 20, 24)
    depth_layers = int((expected >= 0).sum(dim=1).max())
    layers = peel(ctx, pos, tri, (20, 24), depth_layers + 1)

    ok = ~ambiguous
    assert ok.float().mean() > 0.8
    for k, (rast, _) in enumerate(layers[:depth_layers]):
        assert torch.equal((rast[..., 3].long() - 1)[ok], expected[:, k][ok])
    assert (layers[-1][0] == 0).all()


def test_later_layers_reach_hidden_vertices(ctx):
    # The far triangle is hidden by the near one in the first layer; its vertices only get a
    # gradient through the second layer.
    pos = torch.tensor([flat_triangle(-0.5) + flat_triangle(0.5)], requires_grad=True)
    tri = torch.arange(6, dtype=torch.int32).view(2, 3)
    first, second = peel(ctx, pos, tri, (16, 16), 2)

    first[0][..., :3].sum().backward(retain_graph=True)
    assert (pos.grad[0, 3:] == 0).all()
    pos.grad = None
    second[0][..., :3].sum().backward()
    assert pos.grad[0, 3:].abs().sum() > 0
    assert (pos.grad[0, :3] == 0).all()


def test_peeler_validates_inputs(ctx, triangle):
    pos, tri = triangle
    with pytest.raises(ValueError, match="pos must have shape"):
        dr.DepthPeeler(ctx, pos[..., :3], tri, (8, 8))
    with pytest.raises(NotImplementedError, match="Range mode"):
        dr.DepthPeeler(ctx, pos, tri, (8, 8), ranges=torch.zeros(1, 2, dtype=torch.int32))
