"""The Triton backend is checked against the reference backend on the same device."""

import pytest
import torch

pytest.importorskip("triton")

from diffrast.backends import _triton_rasterize, reference
from diffrast.backends import triton as triton_backend

pytestmark = pytest.mark.gpu


def random_mesh(num_tri, gen, z_range=0.9, w_range=(1.0, 2.0), xy_range=1.2, batch=1):
    n = 3 * num_tri
    xy = (torch.rand(batch, n, 2, generator=gen) * 2 - 1) * xy_range
    z = (torch.rand(batch, n, 1, generator=gen) * 2 - 1) * z_range
    w = torch.rand(batch, n, 1, generator=gen) * (w_range[1] - w_range[0]) + w_range[0]
    pos = torch.cat([xy * w, z * w, w], -1).cuda()
    tri = torch.arange(n, dtype=torch.int32).view(num_tri, 3).cuda()
    return pos, tri


def assert_matches_reference(pos, tri, resolution, max_mismatch=0.0):
    ref, ref_db = reference.rasterize(pos, tri, resolution, True)
    out, out_db = triton_backend.rasterize(pos, tri, resolution, True)

    same = ref[..., 3] == out[..., 3]
    mismatch = 1 - same.float().mean().item()
    assert mismatch <= max_mismatch, f"{mismatch:.2%} of pixels picked a different triangle"
    torch.testing.assert_close(out[same], ref[same])
    torch.testing.assert_close(out_db[same], ref_db[same])
    return ref


def test_single_triangle(perspective_triangle):
    pos, tri = (t.cuda() for t in perspective_triangle)
    assert_matches_reference(pos, tri, (40, 56))


@pytest.mark.parametrize("resolution", [(16, 16), (37, 61), (128, 96)])
def test_random_mesh(resolution):
    gen = torch.Generator().manual_seed(0)
    pos, tri = random_mesh(200, gen)
    ref = assert_matches_reference(pos, tri, resolution, max_mismatch=1e-3)
    assert (ref[..., 3] > 0).float().mean() > 0.5


def test_batch_of_views():
    gen = torch.Generator().manual_seed(1)
    pos, tri = random_mesh(50, gen, batch=4)
    assert_matches_reference(pos, tri, (48, 64), max_mismatch=1e-3)


def test_triangles_crossing_camera_plane():
    gen = torch.Generator().manual_seed(2)
    pos, tri = random_mesh(60, gen, w_range=(-1.0, 2.0))
    ref = assert_matches_reference(pos, tri, (64, 64), max_mismatch=1e-3)
    assert (ref[..., 3] > 0).any()


def test_depth_clipping_and_culling():
    gen = torch.Generator().manual_seed(3)
    pos, tri = random_mesh(100, gen, z_range=2.0, xy_range=3.0)
    assert_matches_reference(pos, tri, (64, 64), max_mismatch=1e-3)


def test_many_triangles_in_one_tile(monkeypatch):
    # Overlapping triangles in a single tile, split across several programs that merge their
    # results with atomics.
    monkeypatch.setattr(_triton_rasterize, "CHUNK", 8)
    gen = torch.Generator().manual_seed(4)
    pos, tri = random_mesh(100, gen, xy_range=0.9)
    assert_matches_reference(pos, tri, (16, 16), max_mismatch=1e-3)


def test_equal_depth_resolves_to_lower_id():
    quad = [[-1.0, -1.0, 0.25, 1.0], [1.0, -1.0, 0.25, 1.0], [0.0, 1.0, 0.25, 1.0]]
    pos = torch.tensor([quad * 3]).cuda()
    tri = torch.tensor([[6, 7, 8], [0, 1, 2], [3, 4, 5]], dtype=torch.int32).cuda()
    rast, _ = triton_backend.rasterize(pos, tri, (16, 16), True)
    covered = rast[..., 3] > 0
    assert covered.any()
    assert (rast[..., 3][covered] == 1).all()


def test_negative_and_zero_depth_ordering():
    def flat(z):
        return [[-1.0, -1.0, z, 1.0], [3.0, -1.0, z, 1.0], [-1.0, 3.0, z, 1.0]]

    pos = torch.tensor([flat(0.5) + flat(-0.0) + flat(-0.5) + flat(0.0)]).cuda()
    tri = torch.arange(12, dtype=torch.int32).view(4, 3).cuda()
    rast, _ = triton_backend.rasterize(pos, tri, (8, 8), True)
    assert (rast[..., 3] == 3).all()


def test_empty_and_fully_culled_meshes():
    pos = torch.tensor([[[-0.5, -0.5, 0.0, -1.0], [0.5, -0.5, 0.0, -1.0], [0.0, 0.5, 0.0, -1.0]]])
    pos = pos.cuda()
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32).cuda()
    for t in (tri, tri[:0]):
        rast, rast_db = triton_backend.rasterize(pos, t, (16, 16), True)
        assert (rast == 0).all() and (rast_db == 0).all()


def test_gradients_match_reference():
    gen = torch.Generator().manual_seed(5)
    pos, tri = random_mesh(30, gen)
    weights = torch.randn(1, 32, 32, 4, generator=gen).cuda()

    grads = []
    for backend in (reference, triton_backend):
        p = pos.clone().requires_grad_(True)
        rast, rast_db = backend.rasterize(p, tri, (32, 32), True)
        ((rast[..., :3] * weights[..., :3]).sum() + (rast_db * weights).sum()).backward()
        grads.append(p.grad)
    torch.testing.assert_close(grads[1], grads[0])
