"""Vertex index validation, error messages for common mistakes and interpolate's outputs."""

import pytest
import torch

import diffrast as dr
from diffrast import backends, ops
from diffrast._cache import TensorCache
from diffrast.backends import reference


@pytest.fixture(
    params=[("cpu", "reference"), pytest.param(("cuda", "triton"), marks=pytest.mark.gpu)],
    ids=["reference", "triton"],
)
def device(request):
    """Run a test on the reference backend on the CPU and on the Triton backend on the GPU."""
    name, backend = request.param
    if backend == "triton":
        pytest.importorskip("triton")
    previous = backends._selected
    dr.set_backend(backend)
    yield torch.device(name)
    backends._selected = previous


@pytest.fixture
def scans(monkeypatch):
    """Count how often an index buffer is scanned for its range, starting from an empty cache."""
    calls = []

    def compute(tri):
        calls.append(tri.shape[0])
        return ops._index_range(tri)

    monkeypatch.setattr(ops, "_index_ranges", TensorCache(compute))
    return calls


def scene(device="cpu", resolution=(12, 16)):
    """Two views of two overlapping triangles, with their rasterization."""
    pos = torch.tensor(
        [[-0.8, -0.7, 0.3, 1.0], [0.9, -0.5, 0.2, 1.2], [-0.2, 0.8, 0.4, 1.1],
         [-0.5, 0.6, -0.2, 1.0], [0.1, -0.9, -0.1, 1.3], [0.8, 0.7, -0.3, 0.9]],
        device=device,
    )  # fmt: skip
    pos = torch.stack([pos, pos * torch.tensor([-1.0, 0.9, 1.0, 1.0], device=device)])
    tri = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=device)
    rast, rast_db = dr.rasterize(dr.RasterizeContext(device), pos, tri, resolution)
    return pos, tri, rast, rast_db


class TestVertexIndices:
    @pytest.mark.parametrize("index", [6, 7, 1000, -1, -6])
    def test_out_of_range_index_is_rejected(self, device, index):
        pos, tri, rast, _ = scene(device)
        bad = tri.clone()
        bad[1, 2] = index
        color = torch.zeros(2, 12, 16, 3, device=device)
        message = rf"tri refers to vertex {index}, but valid indices are 0 to 5 for 6 vertices"
        with pytest.raises(ValueError, match=message):
            dr.interpolate(pos, rast, bad)
        with pytest.raises(ValueError, match=message):
            dr.antialias(color, rast, pos, bad)

    def test_checked_against_the_attributes(self, device):
        _, tri, rast, _ = scene(device)
        with pytest.raises(
            ValueError, match="tri refers to vertex 5, but valid indices are 0 to 3"
        ):
            dr.interpolate(torch.zeros(4, 2, device=device), rast, tri)
        with pytest.raises(
            ValueError, match="tri refers to vertex 5, but valid indices are 0 to 4"
        ):
            dr.interpolate(torch.zeros(2, 5, 2, device=device), rast, tri)
        dr.interpolate(torch.zeros(9, 2, device=device), rast, tri)

    def test_checked_against_shared_positions(self, device):
        pos, tri, rast, _ = scene(device)
        color = torch.zeros(2, 12, 16, 3, device=device)
        with pytest.raises(ValueError, match="tri refers to vertex 5"):
            dr.antialias(color, rast, pos[0, :5], tri)

    def test_helper_names_the_argument(self):
        tri = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.int32)
        ops._check_indices(tri, 4)
        ops._check_indices(tri[:0], 0)
        with pytest.raises(ValueError, match="uv_idx refers to vertex 3, but valid indices are 0"):
            ops._check_indices(tri, 3, name="uv_idx")
        with pytest.raises(
            ValueError, match=r"tri refers to vertex 3, but there are no vertices\.$"
        ):
            ops._check_indices(tri, 0)

    def test_range_is_found_once_per_tensor(self, scans):
        pos, tri, rast, _ = scene()
        color = torch.zeros(2, 12, 16, 3)
        for _ in range(3):
            dr.interpolate(pos, rast, tri)
            dr.antialias(color, rast, pos, tri)
            # The stored range serves any vertex count.
            dr.interpolate(torch.zeros(40, 2), rast, tri)
            with pytest.raises(ValueError, match="tri refers to vertex 5"):
                dr.interpolate(torch.zeros(3, 2), rast, tri)
        assert len(scans) == 1

    def test_in_place_edit_is_checked_again(self, scans):
        pos, tri, rast, _ = scene()
        dr.interpolate(pos, rast, tri)
        tri[0, 0] = 6
        with pytest.raises(ValueError, match="tri refers to vertex 6"):
            dr.interpolate(pos, rast, tri)
        tri[:1][0, 0] = 0
        dr.interpolate(pos, rast, tri)
        assert len(scans) == 3

    def test_equal_tensor_is_checked_separately(self, scans):
        pos, tri, rast, _ = scene()
        dr.interpolate(pos, rast, tri)
        dr.interpolate(pos, rast, tri.clone())
        assert len(scans) == 2


class TestTriangleIdsInRast:
    """Ids past the end of ``tri`` are treated as background instead of read out of range."""

    def test_interpolate(self, device):
        _, tri, rast, rast_db = scene(device)
        attr = torch.rand(6, 3, generator=torch.Generator().manual_seed(0)).to(device)
        attr.requires_grad_(True)
        covered = rast[..., 3] > 0
        beyond = rast.clone()
        beyond[..., 3] += 2 * covered
        out, out_da = dr.interpolate(attr, beyond, tri, rast_db=rast_db, diff_attrs="all")
        (out.sum() + out_da.sum()).backward()

        assert covered.any()
        assert (out == 0).all() and (out_da == 0).all()
        assert (attr.grad == 0).all()

    def test_interpolate_keeps_ids_in_range(self, device):
        pos, tri, rast, _ = scene(device)
        second = rast[..., 3] == 2
        beyond = rast.clone()
        beyond[..., 3] += 5 * second
        expected, _ = dr.interpolate(pos, rast, tri)
        out, _ = dr.interpolate(pos, beyond, tri)
        assert second.any() and (~second & (rast[..., 3] > 0)).any()
        torch.testing.assert_close(out, expected * ~second[..., None])

    def test_antialias(self, device):
        pos, tri, rast, _ = scene(device)
        color = torch.rand(2, 12, 16, 3, generator=torch.Generator().manual_seed(1)).to(device)
        second = rast[..., 3] == 2
        beyond = rast.clone()
        beyond[..., 3] += 5 * second
        background = rast.clone()
        background[..., 3] *= ~second
        p = pos.clone().requires_grad_(True)
        out = dr.antialias(color, beyond, p, tri)
        out.sum().backward()
        torch.testing.assert_close(out, dr.antialias(color, background, pos, tri))
        assert p.grad.isfinite().all()


class TestInterpolateOutputs:
    def test_second_output_is_empty_without_derivatives(self, device):
        _, tri, rast, rast_db = scene(device)
        attr = torch.rand(6, 3, device=device, requires_grad=True)
        for kwargs in ({}, {"rast_db": rast_db}, {"rast_db": rast_db, "diff_attrs": []}):
            out, out_da = dr.interpolate(attr, rast, tri, **kwargs)
            assert out_da.shape == (2, 12, 16, 0)
            assert out_da.dtype == torch.float32 and out_da.device == rast.device
            assert not out_da.requires_grad
            assert torch.cat([out, out_da], -1).shape == out.shape
            out.sum().backward()

    def test_second_output_is_empty_without_triangles(self, device):
        _, tri, rast, _ = scene(device)
        out, out_da = dr.interpolate(torch.zeros(6, 3, device=device), rast, tri[:0])
        assert out.shape == (2, 12, 16, 3) and out_da.shape == (2, 12, 16, 0)

    def test_empty_rast_db_counts_as_none(self, device):
        pos, tri, rast, _ = scene(device)
        empty = rast.new_zeros(2, 12, 16, 0)
        expected, _ = dr.interpolate(pos, rast, tri)
        out, out_da = dr.interpolate(pos, rast, tri, rast_db=empty)
        assert torch.equal(out, expected)
        assert out_da.shape == (2, 12, 16, 0)
        with pytest.raises(ValueError, match="rast_db is required when diff_attrs is set, but"):
            dr.interpolate(pos, rast, tri, rast_db=empty, diff_attrs="all")

    def test_rast_db_is_checked_without_diff_attrs(self, device):
        pos, tri, rast, rast_db = scene(device)
        with pytest.raises(
            ValueError, match=r"rast_db must have shape \[batch, height, width, 4\]"
        ):
            dr.interpolate(pos, rast, tri, rast_db=torch.zeros(3, device=device))
        with pytest.raises(ValueError, match="rast_db must have the same shape as rast"):
            dr.interpolate(pos, rast, tri, rast_db=rast_db[:, :6])
        with pytest.raises(TypeError, match=r"rast_db must be torch\.float32"):
            dr.interpolate(pos, rast, tri, rast_db=rast_db.double())

    def test_unused_rast_db_is_not_differentiated(self, device):
        _, tri, rast, rast_db = scene(device)
        attr = torch.rand(6, 3, device=device, requires_grad=True)
        rast_db = rast_db.clone().requires_grad_(True)
        out, _ = dr.interpolate(attr, rast, tri, rast_db=rast_db)
        out.sum().backward()
        assert rast_db.grad is None and attr.grad is not None


class TestInferenceTensors:
    """Inputs made under ``torch.inference_mode`` work in a pass that tracks gradients."""

    def inputs(self, device):
        with torch.inference_mode():
            pos, tri, rast, rast_db = scene(device)
        return pos, tri, rast, rast_db

    def test_interpolate(self, device):
        _, tri, rast, rast_db = self.inputs(device)
        normal = [t.clone() for t in (tri, rast, rast_db)]
        grads = []
        for tri_, rast_, rast_db_ in ((tri, rast, rast_db), normal):
            attr = torch.ones(6, 2, device=device, requires_grad=True)
            out, out_da = dr.interpolate(attr, rast_, tri_, rast_db=rast_db_, diff_attrs=[1])
            (out.square().sum() + out_da.sum()).backward()
            grads.append(attr.grad)
        torch.testing.assert_close(grads[0], grads[1])
        assert grads[0].abs().sum() > 0

    def test_antialias(self, device):
        pos, tri, rast, _ = self.inputs(device)
        color = torch.rand(2, 12, 16, 3, generator=torch.Generator().manual_seed(2)).to(device)
        grads = []
        for tri_, rast_ in ((tri, rast), (tri.clone(), rast.clone())):
            p = pos.clone().requires_grad_(True)
            dr.antialias(color, rast_, p, tri_).square().sum().backward()
            grads.append(p.grad)
        torch.testing.assert_close(grads[0], grads[1])
        assert grads[0].abs().sum() > 0


class TestMessages:
    @pytest.mark.parametrize(
        "resolution", [8, None, 8.0, (8.0, 8.0), (8.5, 8), "88", (8, 8, 8), [[8, 8]], (8, None)]
    )
    def test_resolution_must_be_two_ints(self, resolution):
        with pytest.raises(ValueError, match=r"resolution must be \(height, width\) as two ints"):
            ops._check_resolution(resolution)

    @pytest.mark.parametrize(
        "resolution",
        [
            {8, 16},
            {8: "height", 16: "width"},
            iter([8, 16]),
            range(8, 10),
            (True, 8),
            [8, False],
            torch.tensor([True, True]),
            torch.tensor([8.0, 16.0]),
            torch.tensor([[8, 16]]),
            torch.tensor(8),
        ],
        ids=lambda r: type(r).__name__,
    )
    def test_resolution_must_be_an_ordered_pair_of_ints(self, resolution):
        with pytest.raises(ValueError, match=r"resolution must be \(height, width\) as two ints"):
            ops._check_resolution(resolution)

    def test_resolution_containers(self):
        np = pytest.importorskip("numpy")
        forms = ([8, 16], (8, 16), torch.Size([8, 16]), torch.tensor([8, 16]), np.array([8, 16]))
        for resolution in (*forms, (np.int64(8), torch.tensor(16))):
            size = ops._check_resolution(resolution)
            assert size == (8, 16) and all(type(s) is int for s in size)
        with pytest.raises(ValueError, match="as two ints"):
            ops._check_resolution(np.array([8.0, 16.0]))

    @pytest.mark.parametrize("resolution", [(0, 8), (8, -1), [0, 0]])
    def test_resolution_must_be_positive(self, resolution):
        with pytest.raises(ValueError, match="resolution must be positive"):
            ops._check_resolution(resolution)

    def test_resolution_is_checked_by_rasterize(self, ctx, triangle):
        pos, tri = triangle
        with pytest.raises(ValueError, match=r"resolution must be .* got 8\."):
            dr.rasterize(ctx, pos, tri, 8)

    def test_missing_batch_dimension(self, triangle):
        pos, tri = triangle
        with pytest.raises(
            ValueError, match=r"rast must have shape \[batch, height, width, 4\].*rast\[None\]"
        ):
            dr.interpolate(pos, torch.zeros(8, 8, 4), tri)
        expected = r"pos must have shape \[batch, vertices, 4\], got \(3, 4\); add a batch"
        with pytest.raises(ValueError, match=expected):
            ops._check(pos[0], "pos", ndim=3, last=4)
        with pytest.raises(
            ValueError, match=r"tex must have shape \[batch, tex_height.*tex\[None\]"
        ):
            ops._check(torch.zeros(4, 4, 3), "tex", ndim=4)

    def test_layout_is_named(self, triangle):
        pos, tri = triangle
        rast = torch.zeros(1, 8, 8, 4)
        with pytest.raises(ValueError, match=r"tri must have shape \[triangles, 3\], got \(1, 4\)"):
            dr.interpolate(pos, rast, torch.zeros(1, 4, dtype=torch.int32))
        with pytest.raises(
            ValueError, match=r"attr must have shape \[batch, vertices, channels\] or"
        ):
            dr.interpolate(pos[0, 0], rast, tri)
        with pytest.raises(
            ValueError, match=r"rast_db must have shape \[batch, height, width, 4\]"
        ):
            dr.interpolate(pos, rast, tri, rast_db=rast[..., :2], diff_attrs="all")
        with pytest.raises(ValueError, match=r"same shape as rast, got \(1, 4, 8, 4\) and"):
            dr.interpolate(pos, rast, tri, rast_db=rast[:, :4], diff_attrs="all")
        with pytest.raises(ValueError, match=r"color must match rast.*\(1, 4, 4, 3\) and"):
            dr.antialias(torch.zeros(1, 4, 4, 3), rast, pos, tri)

    def test_helper_falls_back_to_placeholders(self):
        with pytest.raises(ValueError, match=r"weights must have shape \[\.\.\., \.\.\., 5\], got"):
            ops._check(torch.zeros(2), "weights", ndim=3, last=5)
        with pytest.raises(ValueError, match=r"uv must have shape \[\.\.\., \.\.\., \.\.\., 3\]"):
            ops._check(torch.zeros(1, 2, 2, 2), "uv", ndim=4, last=3)
        with pytest.raises(
            ValueError, match=r"tex must have shape \[\.\.\., \.\.\., \.\.\., \.\.\., \.\.\.\]"
        ):
            ops._check(torch.zeros(1, 2, 2, 2), "tex", ndim=5)
        with pytest.raises(ValueError, match=r"pos must have shape \[vertices, 4\], got \(3,\)\."):
            ops._check(torch.zeros(3), "pos", ndim=2, last=4, layout="[vertices, 4]")

    def test_wrong_dtype_says_how_to_convert(self, triangle):
        pos, tri = triangle
        rast = torch.zeros(1, 8, 8, 4)
        with pytest.raises(
            TypeError, match=r"tri must be torch\.int32, got torch\.int64; .*tri\.int\(\)"
        ):
            dr.interpolate(pos, rast, tri.long())
        with pytest.raises(
            TypeError, match=r"attr must be torch\.float32, got torch\.float64; .*attr\.float\(\)"
        ):
            dr.interpolate(pos.double(), rast, tri)
        with pytest.raises(TypeError, match=r"pos must be torch\.float32, got torch\.int64"):
            dr.antialias(rast, rast, pos.long(), tri)
        with pytest.raises(TypeError, match=r"tri must be torch\.int32, got torch\.float32"):
            dr.antialias(rast, rast, pos, tri.float())
        # A description that is not an argument name gets no code to paste.
        with pytest.raises(TypeError, match=r"level 1 must be torch\.float32, got torch\.int32\.$"):
            ops._check(tri, "level 1", ndim=2, dtype=torch.float32)

    def test_non_tensors(self, triangle):
        pos, tri = triangle
        rast = torch.zeros(1, 8, 8, 4)
        with pytest.raises(TypeError, match=r"attr must be a torch\.Tensor, got list"):
            dr.interpolate(pos.tolist(), rast, tri)
        with pytest.raises(TypeError, match=r"rast_db must be a torch\.Tensor, got bool"):
            dr.interpolate(pos, rast, tri, rast_db=True)
        with pytest.raises(TypeError, match=r"pos must be a torch\.Tensor, got NoneType"):
            dr.antialias(rast, rast, None, tri)

    @pytest.mark.parametrize("diff_attrs", [1, 1.5, [0.5], [None]])
    def test_diff_attrs_of_a_wrong_type(self, triangle, diff_attrs):
        pos, tri = triangle
        rast = torch.zeros(1, 8, 8, 4)
        with pytest.raises(TypeError, match="diff_attrs must be a sequence of channel indices"):
            dr.interpolate(pos, rast, tri, rast_db=rast, diff_attrs=diff_attrs)

    def test_diff_attrs_misspelt(self, triangle):
        pos, tri = triangle
        rast = torch.zeros(1, 8, 8, 4)
        with pytest.raises(ValueError, match="'all' or None, got 'ALL'"):
            dr.interpolate(pos, rast, tri, rast_db=rast, diff_attrs="ALL")

    def test_diff_attrs_accepts_any_integer_sequence(self, ctx, perspective_triangle):
        pos, tri = perspective_triangle
        rast, rast_db = dr.rasterize(ctx, pos, tri, (8, 8))
        _, expected = dr.interpolate(pos, rast, tri, rast_db=rast_db, diff_attrs=[2, 0])
        for diff_attrs in ((2, 0), torch.tensor([2, 0]), range(2, -1, -2)):
            _, out_da = dr.interpolate(pos, rast, tri, rast_db=rast_db, diff_attrs=diff_attrs)
            assert torch.equal(out_da, expected)

    @pytest.mark.parametrize("boost", ["2", None, [1.0]])
    def test_pos_gradient_boost_must_be_a_number(self, triangle, boost):
        pos, tri = triangle
        rast = torch.zeros(1, 8, 8, 4)
        with pytest.raises(TypeError, match="pos_gradient_boost must be a number"):
            dr.antialias(rast, rast, pos, tri, pos_gradient_boost=boost)


def test_reference_antialias_returns_a_contiguous_copy(ctx, perspective_triangle):
    pos, tri = perspective_triangle
    rast, _ = dr.rasterize(ctx, pos, tri, (8, 8))
    color = torch.rand(1, 8, 3, 8).permute(0, 1, 3, 2)
    out = reference.antialias(color, rast, pos, tri, 1.0)
    assert out.is_contiguous()
    empty = reference.antialias(color, rast, pos, tri[:0], 1.0)
    assert empty is not color and torch.equal(empty, color)
