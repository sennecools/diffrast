import pytest
import torch

import diffrast as dr
from diffrast import backends


@pytest.fixture(autouse=True)
def restore_backend():
    previous = backends._selected
    yield
    backends._selected = previous


def test_auto_uses_reference_on_cpu():
    dr.set_backend("auto")
    assert dr.get_backend_name("cpu", "rasterize") == "reference"


def test_explicit_reference():
    dr.set_backend("reference")
    assert dr.get_backend_name("cpu", "rasterize") == "reference"


def test_triton_rejects_cpu():
    dr.set_backend("triton")
    with pytest.raises(RuntimeError, match="triton backend"):
        dr.get_backend_name("cpu", "rasterize")


def test_auto_skips_ops_triton_does_not_implement(monkeypatch):
    monkeypatch.setattr(backends, "_triton_usable", lambda device: True)
    monkeypatch.setattr(backends._load("triton"), "IMPLEMENTED", frozenset({"rasterize"}))
    dr.set_backend("auto")
    assert dr.get_backend_name("cuda", "rasterize") == "triton"
    assert dr.get_backend_name("cuda", "texture") == "reference"


def test_unknown_op():
    with pytest.raises(ValueError, match="Unknown op"):
        dr.get_backend_name("cpu", "shade")


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        dr.set_backend("opengl")


def test_context_defaults_to_available_device():
    ctx = dr.RasterizeContext()
    assert ctx.device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_nvdiffrast_alias():
    assert dr.RasterizeCudaContext is dr.RasterizeContext
