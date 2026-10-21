import pytest
import torch

import diffrast as dr


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA or ROCm device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def ctx():
    return dr.RasterizeContext("cpu")


@pytest.fixture
def triangle():
    pos = torch.tensor(
        [[[-0.5, -0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return pos, tri


@pytest.fixture
def perspective_triangle():
    """A triangle with distinct w per vertex, so barycentrics are not affine in screen space."""
    pos = torch.tensor(
        [[[-0.8, -0.6, 0.2, 1.0], [1.2, -0.9, 0.5, 2.0], [0.1, 1.5, -0.3, 1.5]]],
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    return pos, tri
