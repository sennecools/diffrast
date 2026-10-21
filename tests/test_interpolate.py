import torch

import diffrast as dr


def test_vertex_attributes_at_vertices(ctx):
    pos = torch.tensor([[[-1.0, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]]])
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    colour = torch.eye(3)
    rast, _ = dr.rasterize(ctx, pos, tri, (8, 8))
    out, _ = dr.interpolate(colour, rast, tri)

    assert (rast[..., 3] == 1).all()
    torch.testing.assert_close(out.sum(-1), torch.ones(1, 8, 8))
    torch.testing.assert_close(out[..., 0], rast[..., 0])
    torch.testing.assert_close(out[..., 1], rast[..., 1])


def test_shared_and_batched_attributes_agree(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    views = torch.cat([pos, pos])
    rast, _ = dr.rasterize(ctx, views, tri, (16, 16))
    attr = torch.randn(3, 5)
    shared, _ = dr.interpolate(attr, rast, tri)
    batched, _ = dr.interpolate(attr.expand(2, -1, -1), rast, tri)
    single, _ = dr.interpolate(attr[None], rast, tri)
    torch.testing.assert_close(shared, batched)
    torch.testing.assert_close(shared, single)


def test_attribute_derivatives_match_neighbour_differences(ctx):
    # With w = 1 everywhere, attributes are affine in screen space, so differences are exact.
    pos = torch.tensor([[[-1.0, -1.0, 0.0, 1.0], [3.0, -1.0, 0.0, 1.0], [-1.0, 3.0, 0.0, 1.0]]])
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    attr = torch.tensor([[0.0, 1.0], [4.0, -2.0], [1.0, 3.0]])
    rast, rast_db = dr.rasterize(ctx, pos, tri, (8, 8))
    out, out_da = dr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs="all")

    assert out_da.shape == (1, 8, 8, 4)
    for c in range(2):
        dx = out[0, :, 1:, c] - out[0, :, :-1, c]
        dy = out[0, 1:, :, c] - out[0, :-1, :, c]
        torch.testing.assert_close(out_da[0, :, :-1, 2 * c], dx)
        torch.testing.assert_close(out_da[0, :-1, :, 2 * c + 1], dy)


def test_selected_derivative_channels(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    rast, rast_db = dr.rasterize(ctx, pos, tri, (16, 16))
    attr = torch.randn(3, 4)
    _, all_da = dr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs="all")
    _, some_da = dr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs=[3, 1])
    torch.testing.assert_close(some_da, all_da[..., [6, 7, 2, 3]])


def test_no_derivatives_requested(ctx, triangle):
    pos, tri = triangle
    rast, rast_db = dr.rasterize(ctx, pos, tri, (8, 8))
    _, out_da = dr.interpolate(torch.ones(3, 2), rast, tri, rast_db=rast_db)
    assert out_da is None
