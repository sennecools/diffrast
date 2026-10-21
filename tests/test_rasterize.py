import pytest
import torch

import diffrast as dr
from diffrast.backends import reference
from helpers import pixel_centres, screen_space_rasterize

TRI = torch.tensor([[0, 1, 2]], dtype=torch.int32)


def covered(rast):
    return rast[..., 3] > 0


def test_coverage_matches_projected_triangle(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    rast, _ = dr.rasterize(ctx, pos, tri, (32, 32))
    expected, _, ambiguous = screen_space_rasterize(pos, tri, 32, 32)

    assert covered(rast).sum() > 100
    assert torch.equal(covered(rast)[~ambiguous], (expected >= 0)[~ambiguous])


def test_barycentrics_reproject_to_pixel_centres(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    rast, _ = dr.rasterize(ctx, pos, tri, (32, 32))
    point, _ = dr.interpolate(pos, rast, tri)

    mask = covered(rast)
    ndc = point[..., :3] / point[..., 3:]
    centres = pixel_centres(32, 32).expand(1, -1, -1, -1)
    torch.testing.assert_close(ndc[..., :2][mask], centres[mask])
    torch.testing.assert_close(ndc[..., 2][mask], rast[..., 2][mask])


def test_background_is_zero(ctx, triangle):
    pos, tri = triangle
    rast, rast_db = dr.rasterize(ctx, pos, tri, (16, 16))
    background = ~covered(rast)
    assert background.any()
    assert (rast[background] == 0).all()
    assert (rast_db[background] == 0).all()


@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_nearest_triangle_wins(ctx, order):
    near = [[-0.9, -0.9, -0.5, 1.0], [0.9, -0.9, -0.5, 1.0], [0.0, 0.9, -0.5, 1.0]]
    far = [[-0.9, 0.9, 0.5, 1.0], [0.9, 0.9, 0.5, 1.0], [0.0, -0.9, 0.5, 1.0]]
    pos = torch.tensor([near + far])
    tris = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)[order]
    rast, _ = dr.rasterize(ctx, pos, tris, (16, 16))

    centre = rast[0, 8, 8]
    assert centre[3] == order.index(0) + 1
    assert centre[2] == pytest.approx(-0.5)


def test_random_mesh_matches_screen_space_rasterizer(ctx):
    gen = torch.Generator().manual_seed(0)
    n = 40
    xy = torch.rand(1, 3 * n, 2, generator=gen) * 2.4 - 1.2
    z = torch.rand(1, 3 * n, 1, generator=gen) * 1.8 - 0.9
    w = torch.rand(1, 3 * n, 1, generator=gen) + 1.0
    pos = torch.cat([xy * w, z * w, w], -1)
    tri = torch.arange(3 * n, dtype=torch.int32).view(n, 3)

    rast, _ = dr.rasterize(ctx, pos, tri, (24, 32))
    expected_id, expected_depth, ambiguous = screen_space_rasterize(pos, tri, 24, 32)

    ok = ~ambiguous
    assert ok.float().mean() > 0.9
    assert torch.equal((rast[..., 3].long() - 1)[ok], expected_id[ok])
    hit = ok & (expected_id >= 0)
    torch.testing.assert_close(rast[..., 2][hit].double(), expected_depth[hit], atol=1e-5, rtol=0)


def test_triangle_behind_camera_is_not_drawn(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    # Negating all four coordinates gives the same projection, but with w < 0.
    rast, _ = dr.rasterize(ctx, -pos, tri, (16, 16))
    assert not covered(rast).any()


def test_triangle_crossing_camera_plane(ctx):
    pos = torch.tensor([[[-0.5, -0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0], [0.0, 0.2, 0.3, -0.5]]])
    rast, _ = dr.rasterize(ctx, pos, TRI, (32, 32))
    point, _ = dr.interpolate(pos, rast, TRI)

    mask = covered(rast)
    assert mask.sum() > 10
    assert (point[..., 3][mask] > 0).all()
    centres = pixel_centres(32, 32).expand(1, -1, -1, -1)
    ndc = point[..., :2] / point[..., 3:]
    torch.testing.assert_close(ndc[mask], centres[mask], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("z", [-1.5, 1.5])
def test_depth_outside_clip_range_is_not_drawn(ctx, z):
    pos = torch.tensor([[[-0.9, -0.9, z, 1.0], [0.9, -0.9, z, 1.0], [0.0, 0.9, z, 1.0]]])
    rast, _ = dr.rasterize(ctx, pos, TRI, (16, 16))
    assert not covered(rast).any()


def test_degenerate_triangle_is_not_drawn(ctx):
    pos = torch.tensor([[[-0.9, -0.9, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.9, 0.9, 0.0, 1.0]]])
    rast, rast_db = dr.rasterize(ctx, pos, TRI, (16, 16))
    assert not covered(rast).any()
    assert torch.isfinite(rast_db).all()


def test_shared_edge_has_no_cracks(ctx):
    # The diagonal passes exactly through the centres of the pixels on it.
    pos = torch.tensor(
        [
            [
                [-1.0, -1.0, 0.0, 1.0],
                [1.0, -1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [-1.0, 1.0, 0.0, 1.0],
            ]
        ]
    )
    tris = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    rast, _ = dr.rasterize(ctx, pos, tris, (8, 8))

    ids = rast[0, ..., 3]
    assert covered(rast).all()
    assert torch.equal(ids.diagonal(), torch.ones(8))
    assert set(ids.unique().tolist()) == {1.0, 2.0}


def test_batch_matches_individual_views(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    views = torch.cat([pos, pos * torch.tensor([0.7, 1.1, 1.0, 1.0])])
    rast, rast_db = dr.rasterize(ctx, views, tri, (16, 16))
    for i in range(2):
        single, single_db = dr.rasterize(ctx, views[i : i + 1], tri, (16, 16))
        torch.testing.assert_close(rast[i : i + 1], single)
        torch.testing.assert_close(rast_db[i : i + 1], single_db)


def test_non_square_resolution(ctx, triangle):
    pos, tri = triangle
    rast, _ = dr.rasterize(ctx, pos, tri, (8, 16))
    point, _ = dr.interpolate(pos, rast, tri)
    mask = covered(rast)
    assert rast.shape == (1, 8, 16, 4)
    centres = pixel_centres(8, 16).expand(1, -1, -1, -1)
    torch.testing.assert_close(point[..., :2][mask], centres[mask])


def test_empty_mesh(ctx, triangle):
    pos, _ = triangle
    rast, rast_db = dr.rasterize(ctx, pos, torch.zeros(0, 3, dtype=torch.int32), (4, 4))
    assert (rast == 0).all() and (rast_db == 0).all()


def test_chunking_does_not_change_result(ctx, monkeypatch):
    gen = torch.Generator().manual_seed(1)
    pos = torch.cat([torch.rand(1, 30, 3, generator=gen) * 1.6 - 0.8, torch.ones(1, 30, 1)], -1)
    tri = torch.randint(0, 30, (25, 3), generator=gen, dtype=torch.int32)
    full, _ = dr.rasterize(ctx, pos, tri, (16, 16))
    monkeypatch.setattr(reference, "_CHUNK_ELEMENTS", 16 * 16 * 3)
    chunked, _ = dr.rasterize(ctx, pos, tri, (16, 16))
    torch.testing.assert_close(full, chunked)


@pytest.mark.gpu
def test_gpu_matches_cpu(perspective_triangle):
    pos, tri = perspective_triangle
    cpu = dr.rasterize(dr.RasterizeContext("cpu"), pos, tri, (32, 32))
    gpu_ctx = dr.RasterizeContext("cuda")
    gpu = dr.rasterize(gpu_ctx, pos.to(gpu_ctx.device), tri.to(gpu_ctx.device), (32, 32))
    for a, b in zip(cpu, gpu, strict=True):
        torch.testing.assert_close(a, b.cpu(), atol=1e-5, rtol=1e-5)
