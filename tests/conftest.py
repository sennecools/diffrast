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
