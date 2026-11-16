"""Range mode of rasterize and DepthPeeler, and how both treat their context and their state."""

import pytest
import torch

import diffrast as dr
from diffrast.backends import reference
from diffrast.ops import join_ranges, split_ranges

RES = (24, 32)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def glctx(request):
    """A context per device; on the GPU the ops run on the Triton backend."""
    if request.param == "cuda":
        pytest.importorskip("triton")
        if dr.get_backend_name("cuda", "rasterize") != "triton":
            pytest.skip("the Triton backend is not selected")
    return dr.RasterizeContext(request.param)


def scene(device="cpu", seed=0, num_tri=24):
    """A shared vertex buffer with overlapping triangles, and ranges that overlap as well."""
    gen = torch.Generator().manual_seed(seed)
    xy = torch.rand(3 * num_tri, 2, generator=gen) * 2.4 - 1.2
    z = torch.rand(3 * num_tri, 1, generator=gen) * 1.8 - 0.9
    w = torch.rand(3 * num_tri, 1, generator=gen) + 1.0
    pos = torch.cat([xy * w, z * w, w], -1).to(device)
    tri = torch.arange(3 * num_tri, dtype=torch.int32).view(num_tri, 3).to(device)
    ranges = torch.tensor([[0, 9], [10, 13], [4, 5], [23, 1], [7, 0]], dtype=torch.int32)
    return pos, tri, ranges


def instanced(glctx, pos, tri, ranges, **kwargs):
    """Every range rendered by its own instanced call, with ids shifted to index ``tri``."""
    rasts, rast_dbs = [], []
    for start, count in ranges.tolist():
        rast, rast_db = dr.rasterize(glctx, pos[None], tri[start : start + count], RES, **kwargs)
        ids = torch.where(rast[..., 3:] > 0, rast[..., 3:] + start, 0)
        rasts.append(torch.cat([rast[..., :3], ids], -1))
        rast_dbs.append(rast_db)
    return torch.cat(rasts), torch.cat(rast_dbs)


def peel(glctx, pos, tri, layers, **kwargs):
    with dr.DepthPeeler(glctx, pos, tri, RES, **kwargs) as peeler:
        return [peeler.rasterize_next_layer() for _ in range(layers)]


# Range mode


def test_range_mode_matches_instanced_calls(glctx):
    pos, tri, ranges = scene(glctx.device)
    rast, rast_db = dr.rasterize(glctx, pos, tri, RES, ranges=ranges)
    expected, expected_db = instanced(glctx, pos, tri, ranges)

    assert rast.shape == rast_db.shape == (5, *RES, 4)
    assert torch.equal(rast, expected) and torch.equal(rast_db, expected_db)
    covered = rast[..., 3] > 0
    assert covered[:4].flatten(1).any(1).all() and not covered[4].any()
    for item, (start, count) in enumerate(ranges.tolist()):
        ids = rast[item, ..., 3][covered[item]] - 1
        assert ((ids >= start) & (ids < start + count)).all()


def test_ids_index_the_whole_triangle_buffer(glctx):
    pos, tri, ranges = scene(glctx.device)
    attr = torch.arange(pos.shape[0], dtype=torch.float32, device=pos.device)[:, None]
    rast, rast_db = dr.rasterize(glctx, pos, tri, RES, ranges=ranges)
    out, out_da = dr.interpolate(attr, rast, tri, rast_db=rast_db, diff_attrs="all")

    for item, (start, count) in enumerate(ranges.tolist()):
        single = dr.rasterize(glctx, pos[None], tri[start : start + count], RES)
        expected, expected_da = dr.interpolate(
            attr, single[0], tri[start : start + count], rast_db=single[1], diff_attrs="all"
        )
        torch.testing.assert_close(out[item : item + 1], expected)
        torch.testing.assert_close(out_da[item : item + 1], expected_da)


@pytest.mark.parametrize("grad_db", [True, False])
def test_gradients_reach_the_shared_vertices(glctx, grad_db):
    pos, tri, ranges = scene(glctx.device, seed=1)
    gen = torch.Generator().manual_seed(2)
    weights = torch.randn(5, *RES, 4, generator=gen).to(glctx.device)

    grads = []
    for range_mode in (True, False):
        p = pos.clone().requires_grad_(True)
        if range_mode:
            rast, rast_db = dr.rasterize(glctx, p, tri, RES, ranges=ranges, grad_db=grad_db)
        else:
            rast, rast_db = instanced(glctx, p, tri, ranges, grad_db=grad_db)
        assert rast_db.requires_grad == grad_db
        loss = (rast[..., :3] * weights[..., :3]).sum()
        (loss + (rast_db * weights).sum() if grad_db else loss).backward()
        grads.append(p.grad)
    assert grads[0].abs().sum() > 0
    # Triangle 9 is in no range, so its vertices get no gradient.
    assert (grads[0][27:30] == 0).all()
    scale = float(grads[1].abs().max())
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-4 * scale)


def test_range_mode_gradcheck():
    pos, tri, ranges = scene(seed=3, num_tri=6)
    ranges = torch.tensor([[0, 4], [2, 4], [5, 0]], dtype=torch.int32)
    pos = pos.double().requires_grad_(True)

    def render(p):
        layers = [reference.rasterize(p[None], t, (8, 8), True) for t in split_ranges(tri, ranges)]
        return join_ranges(p, (8, 8), ranges, layers)

    assert (render(pos)[0][:2, ..., 3] > 0).flatten(1).sum(1).min() > 10
    assert torch.autograd.gradcheck(render, (pos,))


def test_depth_peeler_in_range_mode(glctx):
    pos, tri, ranges = scene(glctx.device, seed=4)
    layers = peel(glctx, pos, tri, 4, ranges=ranges)
    assert (layers[2][0][..., 3] > 0).any(), "the scene should have several layers"
    for item, (start, count) in enumerate(ranges.tolist()):
        single = peel(glctx, pos[None], tri[start : start + count], 4)
        for (rast, rast_db), (expected, expected_db) in zip(layers, single, strict=True):
            ids = torch.where(expected[..., 3:] > 0, expected[..., 3:] + start, 0)
            assert torch.equal(rast[item : item + 1, ..., :3], expected[..., :3])
            assert torch.equal(rast[item : item + 1, ..., 3:], ids)
            assert torch.equal(rast_db[item : item + 1], expected_db)


def test_depth_peeler_gradients_in_range_mode(glctx):
    pos, tri, ranges = scene(glctx.device, seed=5)
    grads = []
    for mode in ("ranges", "instanced"):
        p = pos.clone().requires_grad_(True)
        if mode == "ranges":
            second = peel(glctx, p, tri, 2, ranges=ranges)[1][0]
        else:
            second = torch.cat(
                [peel(glctx, p[None], tri[s : s + c], 2)[1][0] for s, c in ranges.tolist()]
            )
        second[..., :3].sum().backward()
        grads.append(p.grad)
    assert grads[0].abs().sum() > 0
    scale = float(grads[1].abs().max())
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-4 * scale)


def test_ranges_may_live_on_either_device(glctx):
    pos, tri, ranges = scene(glctx.device)
    expected, _ = dr.rasterize(glctx, pos, tri, RES, ranges=ranges)
    rast, _ = dr.rasterize(glctx, pos, tri, RES, ranges=ranges.to(glctx.device))
    assert torch.equal(rast, expected)


def test_empty_ranges(ctx):
    pos, tri, _ = scene()
    none = torch.zeros(0, 2, dtype=torch.int32)
    rast, rast_db = dr.rasterize(ctx, pos, tri, RES, ranges=none)
    assert rast.shape == rast_db.shape == (0, *RES, 4)
    assert peel(ctx, pos, tri, 2, ranges=none)[1][0].shape == (0, *RES, 4)

    empty = torch.tensor([[3, 0], [24, 0]], dtype=torch.int32)
    rast, rast_db = dr.rasterize(ctx, pos, tri, RES, ranges=empty)
    assert rast.shape == (2, *RES, 4) and (rast == 0).all() and (rast_db == 0).all()


def test_ranges_are_ignored_in_instanced_mode(ctx):
    pos, tri, _ = scene()
    expected = dr.rasterize(ctx, pos[None], tri, RES)
    for ranges in (
        torch.tensor([[0, 3]], dtype=torch.int32),
        torch.tensor([[100, 100]], dtype=torch.int32),
        torch.zeros(7, dtype=torch.int64),
    ):
        rast = dr.rasterize(ctx, pos[None], tri, RES, ranges=ranges)
        assert torch.equal(rast[0], expected[0]) and torch.equal(rast[1], expected[1])
        layers = peel(ctx, pos[None], tri, 1, ranges=ranges)
        assert torch.equal(layers[0][0], expected[0])


@pytest.mark.parametrize("make", [dr.rasterize, dr.DepthPeeler])
def test_range_mode_validates_ranges(ctx, make):
    pos, tri, ranges = scene()
    with pytest.raises(ValueError, match="ranges is required"):
        make(ctx, pos, tri, RES)
    with pytest.raises(TypeError, match=r"ranges must be a torch\.Tensor"):
        make(ctx, pos, tri, RES, ranges=[[0, 3]])
    with pytest.raises(ValueError, match="ranges must have shape"):
        make(ctx, pos, tri, RES, ranges=ranges[0])
    with pytest.raises(ValueError, match="ranges must have shape"):
        make(ctx, pos, tri, RES, ranges=torch.zeros(2, 3, dtype=torch.int32))
    with pytest.raises(TypeError, match=r"ranges must be torch\.int32"):
        make(ctx, pos, tri, RES, ranges=ranges.long())
    for bad in ([20, 5], [-1, 3], [3, -1], [25, 0]):
        with pytest.raises(ValueError, match=r"ranges\[1\]"):
            make(ctx, pos, tri, RES, ranges=torch.tensor([[0, 24], bad], dtype=torch.int32))
    with pytest.raises(ValueError, match="pos must have shape"):
        make(ctx, pos[:, :3], tri, RES, ranges=ranges)


def test_depth_peeler_keeps_its_ranges(ctx):
    pos, tri, ranges = scene()
    kept = ranges.clone()
    with dr.DepthPeeler(ctx, pos, tri, RES, ranges=kept) as peeler:
        first, _ = peeler.rasterize_next_layer()
        kept.zero_()
        second, _ = peeler.rasterize_next_layer()
    expected = peel(ctx, pos, tri, 2, ranges=ranges)
    assert torch.equal(first, expected[0][0]) and torch.equal(second, expected[1][0])


# Context


class ContextWithoutDerivatives:
    """Stands in for a context that produces no ``rast_db``, like nvdiffrast's OpenGL context."""

    output_db = False

    def __init__(self, device):
        self.device = torch.device(device)


def test_context_is_named_glctx(ctx, triangle):
    pos, tri = triangle
    expected, _ = dr.rasterize(ctx, pos, tri, (8, 8))
    rast, _ = dr.rasterize(glctx=ctx, pos=pos, tri=tri, resolution=(8, 8))
    assert torch.equal(rast, expected)
    with dr.DepthPeeler(glctx=ctx, pos=pos, tri=tri, resolution=(8, 8)) as peeler:
        assert torch.equal(peeler.rasterize_next_layer()[0], expected)


@pytest.mark.parametrize("grad_db", [True, False])
def test_context_without_derivative_output(glctx, grad_db):
    plain = ContextWithoutDerivatives(glctx.device)
    pos, tri, ranges = scene(glctx.device)
    for p, kwargs in ((pos[None].clone(), {}), (pos.clone(), {"ranges": ranges})):
        expected, _ = dr.rasterize(glctx, p, tri, RES, **kwargs)
        p.requires_grad_(True)
        rast, rast_db = dr.rasterize(plain, p, tri, RES, grad_db=grad_db, **kwargs)
        assert rast_db.shape == (*rast.shape[:3], 0) and not rast_db.requires_grad
        assert torch.equal(rast, expected)
        rast[..., :3].sum().backward()
        assert p.grad.abs().sum() > 0

        for rast, rast_db in peel(plain, p, tri, 2, grad_db=grad_db, **kwargs):
            assert rast.shape == expected.shape and rast_db.shape == (*rast.shape[:3], 0)


# DepthPeeler state


def test_depth_peeler_cannot_be_used_after_its_with_block(glctx):
    pos, tri, _ = scene(glctx.device)
    with dr.DepthPeeler(glctx, pos[None], tri, RES) as peeler:
        peeler.rasterize_next_layer()
    with pytest.raises(RuntimeError, match="closed"):
        peeler.rasterize_next_layer()


def test_depth_peeler_without_with_block_keeps_peeling(ctx):
    pos, tri, _ = scene()
    peeler = dr.DepthPeeler(ctx, pos[None], tri, RES)
    layers = [peeler.rasterize_next_layer() for _ in range(3)]
    for (rast, _), (expected, _) in zip(layers, peel(ctx, pos[None], tri, 3), strict=True):
        assert torch.equal(rast, expected)


def test_depth_peeler_keeps_the_geometry_it_was_given(glctx):
    # The buffers are edited in place between layers; pos is a view of a larger one.
    pos, tri, _ = scene(glctx.device)
    buffer, tri = pos[None].repeat(2, 1, 1), tri.clone()
    expected = peel(glctx, pos[None], tri.clone(), 3)
    with dr.DepthPeeler(glctx, buffer[:1], tri, RES) as peeler:
        layers = [peeler.rasterize_next_layer()]
        buffer[1] += 1.0
        layers.append(peeler.rasterize_next_layer())
        buffer[0, :, 2] -= 0.1
        tri[:] = tri.flip(0)
        layers.append(peeler.rasterize_next_layer())
    for (rast, rast_db), (wanted, wanted_db) in zip(layers, expected, strict=True):
        assert torch.equal(rast, wanted) and torch.equal(rast_db, wanted_db)
    assert (layers[2][0][..., 3] > 0).any()


def test_depth_peeler_gradients_reach_the_pos_it_was_given(glctx):
    pos, tri, _ = scene(glctx.device)
    pos = pos[None].clone().requires_grad_(True)
    with dr.DepthPeeler(glctx, pos, tri, RES) as peeler:
        peeler.rasterize_next_layer()
        with torch.no_grad():
            pos[..., :2] *= 0.5
        rast, rast_db = peeler.rasterize_next_layer()
    assert (rast[..., 3] > 0).any()
    (rast[..., :3].sum() + rast_db.sum()).backward()
    assert pos.grad.abs().sum() > 0


def test_depth_peeler_in_inference_mode(ctx):
    pos, tri, _ = scene()
    with torch.inference_mode():
        layers = peel(ctx, pos[None].clone(), tri, 2)
    expected = peel(ctx, pos[None], tri, 2)
    assert torch.equal(layers[1][0], expected[1][0])
