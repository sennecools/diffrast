from __future__ import annotations

import torch


class RasterizeContext:
    """Device that :func:`diffrast.rasterize` runs on.

    Args:
        device: Device to rasterize on. Defaults to the current CUDA/ROCm device, or the CPU
            when no GPU is available.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device

    def __repr__(self) -> str:
        return f"RasterizeContext(device={str(self.device)!r})"


# Alias so code written against the nvdiffrast API runs unchanged.
RasterizeCudaContext = RasterizeContext
