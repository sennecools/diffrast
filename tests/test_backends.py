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
    assert dr.get_backend_name("cpu") == "reference"


def test_explicit_reference():
    dr.set_backend("reference")
    assert dr.get_backend_name("cpu") == "reference"


def test_triton_rejects_cpu():
    dr.set_backend("triton")
    with pytest.raises(RuntimeError, match="triton backend"):
        dr.get_backend_name("cpu")


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        dr.set_backend("opengl")


def test_context_defaults_to_available_device():
    ctx = dr.RasterizeContext()
    assert ctx.device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_nvdiffrast_alias():
    assert dr.RasterizeCudaContext is dr.RasterizeContext
