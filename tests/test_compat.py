"""Compatibility with code written for nvdiffrast.

Contexts, the ``diffrast.torch`` module path, the log level and the ``nvdiffrast`` import alias,
followed by the ways in which real pipelines call the ops.
"""

import importlib
import os
import pathlib
import pkgutil
import subprocess
import sys
import types
import warnings

import pytest
import torch

import diffrast as dr
import diffrast.torch as dr_torch
from diffrast import backends, compat, log
from diffrast.context import RasterizeContext, RasterizeCudaContext, RasterizeGLContext

CONTEXTS = [RasterizeContext, RasterizeCudaContext, RasterizeGLContext]
NEW_NAMES = [
    "RasterizeGLContext",
    "antialias_construct_topology_hash",
    "get_log_level",
    "set_log_level",
    "install_as_nvdiffrast",
    "uninstall_as_nvdiffrast",
]

RENDERER = '''
from typing import Union

import nvdiffrast.torch as dr


class Renderer:
    """A renderer in the style of the wrappers found in image-to-3D pipelines."""

    ctx: {annotation}

    def __init__(self, device, context: {annotation} = None):
        self.ctx = context if context is not None else dr.{default}(device=device)

    def render(self, pos, tri, colour, resolution):
        rast, _ = dr.rasterize(self.ctx, pos, tri, resolution)
        image, _ = dr.interpolate(colour, rast, tri)
        return dr.antialias(image, rast, pos, tri)
'''


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


class TestContexts:
    def test_class_relations(self):
        assert RasterizeCudaContext is RasterizeContext
        assert issubclass(RasterizeGLContext, RasterizeContext)

    @pytest.mark.parametrize("context", CONTEXTS)
    def test_default_device(self, context):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert context().device.type == expected
        assert context(device=None).device.type == expected

    @pytest.mark.parametrize("context", CONTEXTS)
    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
    def test_cpu(self, context, device):
        assert context(device=device).device == torch.device("cpu")

    @pytest.mark.gpu
    @pytest.mark.parametrize("context", CONTEXTS)
    @pytest.mark.parametrize("device", ["cuda", "cuda:0", torch.device("cuda:0"), 0])
    def test_gpu(self, context, device):
        ctx = context(device=device)
        index = torch.cuda.current_device() if device == "cuda" else 0
        assert ctx.device == torch.device("cuda", index)

    def test_plain_contexts_output_db(self):
        assert RasterizeContext("cpu").output_db is True

    def test_gl_context_arguments(self):
        ctx = RasterizeGLContext(device="cpu")
        assert ctx.output_db is True and ctx.mode == "automatic"
        ctx = RasterizeGLContext(False, "manual", "cpu")
        assert ctx.output_db is False and ctx.mode == "manual"
        assert RasterizeGLContext(output_db=False, device="cpu").output_db is False

    @pytest.mark.gpu
    def test_gl_context_without_output_db_on_gpu(self):
        ctx = RasterizeGLContext(output_db=False, device="cuda")
        assert ctx.output_db is False and ctx.device.type == "cuda"

    @pytest.mark.parametrize("output_db", ["cuda", "no", torch.device("cpu")])
    def test_gl_context_refuses_a_device_as_output_db(self, output_db):
        with pytest.raises(
            TypeError, match=r"output_db must be a bool, got .*; pass the device as"
        ):
            RasterizeGLContext(output_db)

    @pytest.mark.parametrize("context", CONTEXTS)
    def test_device_of_a_wrong_type(self, context):
        for device in (True, False, 0.0, [0], b"cuda"):
            message = "device must be a torch.device, a string or a GPU index, got"
            with pytest.raises(TypeError, match=message):
                context(device=device)
        with pytest.raises(ValueError, match="device must not be a negative GPU index, got -1"):
            context(device=-1)

    @pytest.mark.gpu
    def test_numpy_gpu_index(self):
        np = pytest.importorskip("numpy")
        assert RasterizeContext(device=np.int64(0)).device == torch.device("cuda", 0)

    def test_gl_context_without_output_db_rasterizes(self, device, triangle):
        pos, tri = (t.to(device) for t in triangle)
        rast, rast_db = dr.rasterize(RasterizeGLContext(False, device=device), pos, tri, (8, 8))
        expected, _ = dr.rasterize(RasterizeGLContext(device=device), pos, tri, (8, 8))
        assert torch.equal(rast, expected)
        assert rast_db.shape == (1, 8, 8, 0)
        out, out_da = dr.interpolate(pos, rast, tri, rast_db=rast_db)
        assert out.shape == (1, 8, 8, 4) and out_da.shape == (1, 8, 8, 0)

    def test_gl_context_methods_do_nothing(self):
        ctx = RasterizeGLContext(device="cpu")
        assert ctx.set_context() is None
        assert ctx.release_context() is None

    def test_gl_context_rasterizes(self, triangle):
        pos, tri = triangle
        rast, _ = dr.rasterize(RasterizeGLContext(device="cpu"), pos, tri, (8, 8))
        expected, _ = dr.rasterize(RasterizeContext("cpu"), pos, tri, (8, 8))
        assert torch.equal(rast, expected)

    def test_repr(self):
        assert repr(RasterizeContext("cpu")) == "RasterizeContext(device='cpu')"
        assert repr(RasterizeGLContext(output_db=False, device="cpu")) == (
            "RasterizeGLContext(output_db=False, device='cpu')"
        )

    def test_gpu_requested_without_one(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        for device in ("cuda", "cuda:0", 0, torch.device("cuda")):
            with pytest.raises(RuntimeError, match="no CUDA or ROCm device is available"):
                RasterizeGLContext(device=device)
        assert RasterizeContext().device == torch.device("cpu")

    def test_gpu_index_that_does_not_exist(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        assert RasterizeContext("cuda:1").device == torch.device("cuda", 1)
        with pytest.raises(ValueError, match="device 'cuda:7' does not exist; 2 CUDA or ROCm"):
            RasterizeContext("cuda:7")
        with pytest.raises(ValueError, match="device 'cuda:2' does not exist"):
            RasterizeCudaContext(device=2)


class TestLogLevel:
    @pytest.fixture(autouse=True)
    def restore(self):
        previous = log.get_log_level()
        yield
        log.set_log_level(previous)

    def test_default_is_warning(self):
        assert log.get_log_level() == 1

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_round_trip(self, level):
        log.set_log_level(level)
        assert log.get_log_level() == level

    @pytest.mark.parametrize("level", [-1, 4, 100])
    def test_rejects_unknown_levels(self, level):
        with pytest.raises(ValueError, match="level must be an int from 0 to 3"):
            log.set_log_level(level)
        assert log.get_log_level() == 1

    @pytest.mark.parametrize("level", [1.0, "1", None])
    def test_rejects_other_types(self, level):
        with pytest.raises(TypeError, match="level must be an int from 0 to 3"):
            log.set_log_level(level)

    @pytest.mark.parametrize("level", [0, 1])
    def test_warnings_are_issued(self, level):
        log.set_log_level(level)
        with pytest.warns(UserWarning, match="something odd") as record:
            log.warn("something odd")
        assert record[0].filename == __file__

    @pytest.mark.parametrize("level", [2, 3])
    def test_warnings_are_silenced(self, level):
        log.set_log_level(level)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            log.warn("something odd")


class TestTorchModule:
    def test_exports_everything_diffrast_exports(self):
        assert dr_torch.__all__ == dr.__all__
        for name in dr.__all__:
            assert getattr(dr_torch, name) is getattr(dr, name)

    def test_from_import(self):
        from diffrast.torch import RasterizeCudaContext, rasterize

        assert rasterize is dr.rasterize
        assert RasterizeCudaContext is dr.RasterizeCudaContext

    def test_new_exports_are_picked_up(self, monkeypatch):
        marker = object()
        monkeypatch.setattr(dr, "an_op_added_later", marker, raising=False)
        monkeypatch.setattr(dr, "__all__", [*dr.__all__, "an_op_added_later"])
        try:
            assert importlib.reload(dr_torch).an_op_added_later is marker
        finally:
            monkeypatch.undo()
            vars(dr_torch).pop("an_op_added_later", None)
            importlib.reload(dr_torch)
        assert not hasattr(dr_torch, "an_op_added_later")

    def test_does_not_shadow_pytorch(self):
        pytorch = sys.modules["torch"]
        assert pytorch.__name__ == "torch" and hasattr(pytorch, "Tensor")
        assert dr_torch.__name__ == "diffrast.torch" and dr_torch is not pytorch
        checked = 0
        for info in pkgutil.walk_packages(dr.__path__, "diffrast."):
            try:
                module = importlib.import_module(info.name)
            except ImportError:
                continue  # The Triton kernels, where Triton is not installed.
            if info.name != "diffrast.torch" and hasattr(module, "torch"):
                assert module.torch is pytorch, info.name
                checked += 1
        assert checked >= 4

    @pytest.mark.parametrize("statement", ["import diffrast", "import torch"])
    def test_package_directory_on_the_path_is_reported(self, statement):
        # From inside the package directory, "import torch" finds diffrast/torch.py.
        package = pathlib.Path(dr.__file__).parent
        env = {**os.environ, "PYTHONPATH": str(package.parent)}
        result = subprocess.run(
            [sys.executable, "-c", statement], cwd=package, env=env, capture_output=True, text=True
        )
        assert result.returncode != 0
        assert "found diffrast/torch.py instead of PyTorch" in result.stderr

    @pytest.mark.parametrize("name", NEW_NAMES)
    def test_compatibility_names_are_exported(self, name):
        assert name in dr.__all__
        assert getattr(dr_torch, name) is getattr(dr, name)


class TestInstallAsNvdiffrast:
    NAMES = ("nvdiffrast", "nvdiffrast.torch")

    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch):
        """Start without nvdiffrast, real or aliased, and put everything back afterwards."""
        saved = {name: sys.modules.pop(name) for name in self.NAMES if name in sys.modules}
        monkeypatch.setattr(compat, "_is_installed", lambda name: False)
        yield
        compat.uninstall_as_nvdiffrast()
        compat._displaced.clear()
        for name in self.NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)

    def test_nothing_is_registered_until_asked(self):
        assert not any(name in sys.modules for name in self.NAMES)

    def test_import_resolves_to_diffrast(self):
        compat.install_as_nvdiffrast()
        import nvdiffrast
        import nvdiffrast.torch as nvdr
        from nvdiffrast.torch import interpolate

        assert nvdiffrast is dr
        assert nvdr is dr_torch
        assert nvdiffrast.torch is dr_torch
        assert nvdr.rasterize is dr.rasterize and interpolate is dr.interpolate
        assert importlib.import_module("nvdiffrast.torch") is dr_torch

    def test_can_be_called_twice(self):
        compat.install_as_nvdiffrast()
        compat.install_as_nvdiffrast()
        assert sys.modules["nvdiffrast.torch"] is dr_torch
        assert not compat._displaced

    def test_uninstall(self):
        compat.install_as_nvdiffrast()
        compat.uninstall_as_nvdiffrast()
        assert not any(name in sys.modules for name in self.NAMES)
        compat.uninstall_as_nvdiffrast()

    def test_refuses_when_nvdiffrast_is_installed(self, monkeypatch):
        monkeypatch.setattr(compat, "_is_installed", lambda name: name == "nvdiffrast")
        with pytest.raises(RuntimeError, match="An nvdiffrast package is installed"):
            compat.install_as_nvdiffrast()
        assert not any(name in sys.modules for name in self.NAMES)
        compat.install_as_nvdiffrast(force=True)
        assert sys.modules["nvdiffrast.torch"] is dr_torch

    def test_refuses_when_nvdiffrast_is_imported(self):
        real = types.ModuleType("nvdiffrast")
        real_torch = types.ModuleType("nvdiffrast.torch")
        sys.modules.update({"nvdiffrast": real, "nvdiffrast.torch": real_torch})
        with pytest.raises(RuntimeError, match="nvdiffrast is already imported"):
            compat.install_as_nvdiffrast()
        assert sys.modules["nvdiffrast"] is real

        with pytest.warns(UserWarning, match="Replacing the imported nvdiffrast") as record:
            compat.install_as_nvdiffrast(force=True)
        assert record[0].filename == __file__
        assert sys.modules["nvdiffrast"] is dr and sys.modules["nvdiffrast.torch"] is dr_torch

        compat.uninstall_as_nvdiffrast()
        assert sys.modules["nvdiffrast"] is real
        assert sys.modules["nvdiffrast.torch"] is real_torch

    def test_uninstall_leaves_other_modules_alone(self):
        compat.install_as_nvdiffrast()
        other = types.ModuleType("nvdiffrast")
        sys.modules["nvdiffrast"] = other
        compat.uninstall_as_nvdiffrast()
        assert sys.modules["nvdiffrast"] is other
        assert "nvdiffrast.torch" not in sys.modules

    def import_renderer(self, tmp_path, monkeypatch, annotation, default):
        source = RENDERER.format(annotation=annotation, default=default)
        (tmp_path / "third_party_renderer.py").write_text(source, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "third_party_renderer", raising=False)
        module = importlib.import_module("third_party_renderer")
        monkeypatch.delitem(sys.modules, "third_party_renderer")
        return module

    def render(self, module, triangle):
        pos, tri = triangle
        colour = torch.eye(3)
        image = module.Renderer("cpu").render(pos, tri, colour, [16, 16])
        rast, _ = dr.rasterize(dr.RasterizeContext("cpu"), pos, tri, (16, 16))
        expected = dr.antialias(dr.interpolate(colour, rast, tri)[0], rast, pos, tri)
        torch.testing.assert_close(image, expected)
        assert image.sum() > 0

    def test_unmodified_code_runs(self, tmp_path, monkeypatch, triangle):
        compat.install_as_nvdiffrast()
        module = self.import_renderer(
            tmp_path, monkeypatch, "dr.RasterizeCudaContext", "RasterizeCudaContext"
        )
        assert module.dr is dr_torch
        self.render(module, triangle)

    def test_unmodified_code_with_opengl_context_runs(self, tmp_path, monkeypatch, triangle):
        # The annotations are evaluated when the module is imported.
        dr.install_as_nvdiffrast()
        annotation = "Union[dr.RasterizeGLContext, dr.RasterizeCudaContext]"
        module = self.import_renderer(tmp_path, monkeypatch, annotation, "RasterizeGLContext")
        assert isinstance(module.Renderer("cpu").ctx, dr.RasterizeGLContext)
        self.render(module, triangle)


def sheet(device, size=6):
    """A curved sheet of quads with shared vertices, seen in perspective from two sides.

    Returns ``pos`` ``[2, vertices, 4]``, ``tri`` ``[triangles, 3]`` and ``uv`` ``[vertices, 2]``.
    No two triangles overlap on screen.
    """
    line = torch.linspace(-0.83, 0.86, size + 1)
    x, y = (t.flatten() for t in torch.meshgrid(line, line, indexing="xy"))
    w = 1.3 + 0.25 * x - 0.1 * y
    pos = torch.stack([x * w, y * w, 0.4 * x * y * w, w], -1)
    pos = torch.stack([pos, pos * torch.tensor([-1.0, 1.0, 1.0, 1.0])])
    tri = []
    for row in range(size):
        for column in range(size):
            a = row * (size + 1) + column
            b, c, d = a + 1, a + size + 2, a + size + 1
            tri += [[a, b, c], [a, c, d]]
    uv = torch.stack([x, y], -1) * 0.5 + 0.5
    return pos.to(device), torch.tensor(tri, dtype=torch.int32, device=device), uv.to(device)


def strided(tensor):
    """The same values in memory that is not contiguous."""
    view = tensor.repeat_interleave(2, dim=-1)[..., ::2]
    assert not view.is_contiguous()
    return view


def random(*shape, device, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed)).to(device)


class TestCallPatterns:
    RESOLUTION = (24, 32)

    @pytest.fixture
    def scene(self, device):
        pos, tri, uv = sheet(device)
        ctx = dr.RasterizeContext(device)
        rast, rast_db = dr.rasterize(ctx, pos, tri, self.RESOLUTION)
        assert (rast[..., 3] > 0).float().mean() > 0.5
        return types.SimpleNamespace(
            ctx=ctx, pos=pos, tri=tri, uv=uv, rast=rast, rast_db=rast_db, device=device
        )

    def test_list_and_tuple_resolutions(self, scene):
        for resolution in ([24, 32], (24, 32), torch.tensor([24, 32])):
            rast, rast_db = dr.rasterize(scene.ctx, scene.pos, scene.tri, resolution)
            assert torch.equal(rast, scene.rast) and torch.equal(rast_db, scene.rast_db)

    def test_numpy_resolutions(self, scene):
        np = pytest.importorskip("numpy")
        shape = np.zeros((24, 32)).shape
        forms = (np.array([24, 32]), (np.int64(24), np.int32(32)), [np.intp(24), 32], shape)
        for resolution in forms:
            rast, _ = dr.rasterize(scene.ctx, scene.pos, scene.tri, resolution)
            assert torch.equal(rast, scene.rast)

    def test_positional_rasterize(self, scene):
        rast, rast_db = dr.rasterize(scene.ctx, scene.pos, scene.tri, [24, 32], None, True)
        assert torch.equal(rast, scene.rast) and torch.equal(rast_db, scene.rast_db)

    def test_positional_texture(self, scene):
        tex = random(1, 16, 16, 3, device=scene.device)
        uv, uv_da = dr.interpolate(scene.uv, scene.rast, scene.tri, scene.rast_db, "all")
        out = dr.texture(tex, uv, uv_da, None, None, "linear", "wrap", None)
        expected = dr.texture(tex, uv, filter_mode="linear", boundary_mode="wrap")
        assert torch.equal(out, expected)

    def test_shared_attributes_with_batched_rast(self, scene):
        attr = random(scene.pos.shape[1], 5, device=scene.device)
        shared, _ = dr.interpolate(attr, scene.rast, scene.tri)
        batched, _ = dr.interpolate(attr[None].expand(2, -1, -1), scene.rast, scene.tri)
        single, _ = dr.interpolate(attr[None], scene.rast, scene.tri)
        assert shared.shape == (2, *self.RESOLUTION, 5)
        torch.testing.assert_close(shared, batched)
        torch.testing.assert_close(shared, single)

    def test_per_face_attributes(self, scene):
        faces = scene.tri.shape[0]
        face_tri = torch.arange(faces, dtype=torch.int32, device=scene.device)
        face_tri = face_tri[:, None].repeat(1, 3)
        attr = random(faces, 3, device=scene.device)
        out, _ = dr.interpolate(attr, scene.rast, face_tri)

        ids = scene.rast[..., 3].long()
        expected = attr[(ids - 1).clamp(min=0)] * (ids > 0)[..., None]
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)

    def test_separate_uv_index_buffer(self, scene):
        # One uv per triangle corner, as in a mesh whose uv map has seams.
        corner_uv = scene.uv[scene.tri.long()].reshape(-1, 2)
        uv_tri = torch.arange(corner_uv.shape[0], dtype=torch.int32, device=scene.device)
        uv_tri = uv_tri.view(-1, 3)
        out, out_da = dr.interpolate(corner_uv, scene.rast, uv_tri, scene.rast_db, "all")
        expected = dr.interpolate(scene.uv, scene.rast, scene.tri, scene.rast_db, "all")
        torch.testing.assert_close(out, expected[0])
        torch.testing.assert_close(out_da, expected[1])

    def test_rast_db_without_diff_attrs(self, scene):
        out, out_da = dr.interpolate(scene.uv, scene.rast, scene.tri, rast_db=scene.rast_db)
        expected, _ = dr.interpolate(scene.uv, scene.rast, scene.tri)
        assert torch.equal(out, expected)
        assert out_da.shape == (2, *self.RESOLUTION, 0)
        out, out_da = dr.interpolate(scene.uv, scene.rast, scene.tri, scene.rast_db, None)
        assert torch.equal(out, expected) and out_da.numel() == 0

    def test_rast_edited_in_place(self, scene):
        expected, _ = dr.interpolate(scene.uv, scene.rast, scene.tri)
        rast = scene.rast.clone()
        rast[..., :2].clamp_(0, 1)
        rast[..., :2] = torch.clamp(rast[..., :2], 0, 1)
        out, _ = dr.interpolate(scene.uv, rast, scene.tri)
        torch.testing.assert_close(out, expected)

    def test_rast_edited_in_place_with_gradients(self, scene):
        # The edits are made on the tensor that rasterize returned, which is part of the graph.
        weights = random(2, *self.RESOLUTION, 2, device=scene.device, seed=3)

        def render(edit):
            pos = scene.pos.clone().requires_grad_(True)
            rast, rast_db = dr.rasterize(scene.ctx, pos, scene.tri, self.RESOLUTION)
            rast = edit(rast)
            uv, uv_da = dr.interpolate(scene.uv, rast, scene.tri, rast_db, "all")
            out = dr.antialias(uv, rast, pos, scene.tri)
            ((out * weights).sum() + uv_da.square().sum()).backward()
            return out.detach(), pos.grad

        def in_place(rast):
            rast[..., 3:4] += 0
            rast[..., :2].clamp_(0, 1)
            return rast

        def copied(rast):
            return torch.cat([rast[..., :2].clamp(0, 1), rast[..., 2:]], -1)

        out, grad = render(in_place)
        expected_out, expected_grad = render(copied)
        assert grad.isfinite().all() and (grad != 0).any()
        torch.testing.assert_close(out, expected_out)
        scale = float(expected_grad.abs().max())
        torch.testing.assert_close(grad, expected_grad, atol=1e-5 * scale, rtol=1e-4)

    def test_chunked_triangles(self, scene):
        # Rasterize a few triangles at a time, shift the ids of each chunk to their place in
        # the whole index buffer, and keep the first chunk to cover a pixel.
        rast = torch.zeros_like(scene.rast)
        for start in range(0, scene.tri.shape[0], 20):
            chunk, _ = dr.rasterize(
                scene.ctx, scene.pos, scene.tri[start : start + 20], self.RESOLUTION
            )
            covered = chunk[..., 3:4] > 0
            chunk[..., 3:4] += start
            rast = torch.where(covered & (rast[..., 3:4] == 0), chunk, rast)

        assert torch.equal(rast[..., 3] > 0, scene.rast[..., 3] > 0)
        assert rast[..., 3].max() > 60
        out, _ = dr.interpolate(scene.uv, rast, scene.tri)
        expected, _ = dr.interpolate(scene.uv, scene.rast, scene.tri)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("filter_mode", ["auto", "linear", "nearest"])
    def test_empty_derivatives_passed_on_to_texture(self, scene, filter_mode):
        # Without diff_attrs, interpolate returns an empty tensor in place of the derivatives.
        tex = random(1, 8, 8, 3, device=scene.device)
        uv, uv_da = dr.interpolate(scene.uv, scene.rast, scene.tri)
        assert uv_da.shape[-1] == 0
        out = dr.texture(tex, uv, uv_da, filter_mode=filter_mode)
        assert torch.equal(out, dr.texture(tex, uv, filter_mode=filter_mode))
        assert torch.equal(dr.texture(tex, uv, uv_da=uv_da), dr.texture(tex, uv))

    def test_nearest_filter_with_uv_da(self, scene):
        tex = random(1, 8, 8, 3, device=scene.device)
        uv, uv_da = dr.interpolate(scene.uv, scene.rast, scene.tri, scene.rast_db, "all")
        out = dr.texture(tex, uv, uv_da, filter_mode="nearest")
        assert torch.equal(out, dr.texture(tex, uv, filter_mode="nearest"))

    def test_point_lookups(self, device):
        tex = random(1, 4, 4, 3, device=device)
        centres = (torch.arange(4, device=device) + 0.5) / 4
        v, u = torch.meshgrid(centres, centres, indexing="ij")
        uv = torch.stack([u, v], -1).reshape(1, 16, 1, 2)
        out = dr.texture(tex, uv, filter_mode="linear")
        assert out.shape == (1, 16, 1, 3)
        torch.testing.assert_close(out.reshape(4, 4, 3), tex[0])

    def test_constant_texture_with_mipmaps(self, scene):
        tex = torch.tensor([0.2, 0.5, 0.9], device=scene.device).view(1, 1, 1, 3)
        tex.requires_grad_(True)
        uv, uv_da = dr.interpolate(scene.uv, scene.rast, scene.tri, scene.rast_db, "all")
        out = dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear")
        out.sum().backward()
        torch.testing.assert_close(out, tex.detach().expand_as(out))
        pixels = float(out[..., 0].numel())
        torch.testing.assert_close(tex.grad.flatten(), torch.full((3,), pixels).to(tex))

    def test_antialias_of_a_mask(self, scene):
        pos = scene.pos.clone().requires_grad_(True)
        mask = (scene.rast[..., 3:] > 0).float()
        out = dr.antialias(mask, scene.rast, pos, scene.tri)
        out.sum().backward()

        assert out.shape == mask.shape
        assert ((out > 0) & (out < 1)).any()
        assert pos.grad.isfinite().all()
        assert (pos.grad != 0).any()

    def test_without_grad_db(self, scene):
        pos = scene.pos.clone().requires_grad_(True)
        rast, _ = dr.rasterize(scene.ctx, pos, scene.tri, self.RESOLUTION, grad_db=False)
        out, _ = dr.interpolate(scene.uv, rast, scene.tri)
        out.sum().backward()
        assert torch.equal(rast, scene.rast)
        assert pos.grad.isfinite().all() and (pos.grad != 0).any()

    def test_non_contiguous_inputs(self, scene):
        tex = random(1, 16, 16, 3, device=scene.device, seed=1)
        weights = random(2, *self.RESOLUTION, 3, device=scene.device, seed=2)

        def render(convert):
            pos = scene.pos.clone().requires_grad_(True)
            texture = tex.clone().requires_grad_(True)
            tri = convert(scene.tri)
            rast, rast_db = dr.rasterize(scene.ctx, convert(pos), tri, self.RESOLUTION)
            uv, uv_da = dr.interpolate(convert(scene.uv), rast, tri, rast_db, "all")
            color = dr.texture(convert(texture), convert(uv), convert(uv_da))
            out = dr.antialias(convert(color), rast, convert(pos), tri)
            (out * weights).sum().backward()
            return out.detach(), pos.grad, texture.grad

        expected = render(lambda t: t)
        for ours, theirs in zip(render(strided), expected, strict=True):
            scale = theirs.abs().max()
            assert scale > 0
            torch.testing.assert_close(ours, theirs, atol=1e-5 * float(scale), rtol=1e-4)
