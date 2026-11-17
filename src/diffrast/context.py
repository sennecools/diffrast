from __future__ import annotations

import operator

import torch


class RasterizeContext:
    """Device that :func:`diffrast.rasterize` runs on.

    Args:
        device: Device to rasterize on, as a ``torch.device``, a string such as ``"cuda:0"`` or
            a GPU index. Defaults to the current CUDA/ROCm device, or the CPU when no GPU is
            available.

    Attributes:
        device: The resolved ``torch.device``, with an explicit index for a GPU.
        output_db: Whether :func:`diffrast.rasterize` returns ``rast_db``. Always true here;
            see :class:`RasterizeGLContext`.
    """

    output_db = True

    def __init__(self, device: torch.device | str | int | None = None) -> None:
        self.device = _resolve_device(device)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(device={str(self.device)!r})"


# Alias so code written against the nvdiffrast API runs unchanged.
RasterizeCudaContext = RasterizeContext


class RasterizeGLContext(RasterizeContext):
    """Stand-in for nvdiffrast's OpenGL context, so that code which creates one runs unchanged.

    diffrast has no OpenGL rasterizer: this context rasterizes exactly like
    :class:`RasterizeContext` on the same device.

    Args:
        output_db: Whether :func:`diffrast.rasterize` returns ``rast_db``. With ``False`` it
            returns an empty tensor of shape ``[batch, height, width, 0]`` in its place.
        mode: Accepted for compatibility and ignored.
        device: As for :class:`RasterizeContext`.
    """

    def __init__(
        self,
        output_db: bool = True,
        mode: str = "automatic",
        device: torch.device | str | int | None = None,
    ) -> None:
        super().__init__(device)
        # A device in the first position, where the other contexts take it, would count as true.
        if isinstance(output_db, (str, torch.device)):
            raise TypeError(
                f"output_db must be a bool, got {output_db!r}; pass the device as device=..."
            )
        self.output_db = bool(output_db)
        self.mode = mode

    def __repr__(self) -> str:
        return f"RasterizeGLContext(output_db={self.output_db}, device={str(self.device)!r})"

    def set_context(self) -> None:
        """Do nothing; there is no OpenGL context to make current."""

    def release_context(self) -> None:
        """Do nothing; there is no OpenGL context to release."""


def _resolve_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif not isinstance(device, (torch.device, str)):
        try:
            if isinstance(device, bool):
                raise TypeError
            index = operator.index(device)
        except TypeError:
            raise TypeError(
                f"device must be a torch.device, a string or a GPU index, got {device!r}."
            ) from None
        if index < 0:
            raise ValueError(f"device must not be a negative GPU index, got {device!r}.")
        device = torch.device("cuda", index)
    device = torch.device(device)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device {str(device)!r} was requested, but no CUDA or ROCm device is available."
        )
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    if device.index >= torch.cuda.device_count():
        raise ValueError(
            f"device {str(device)!r} does not exist; {torch.cuda.device_count()} CUDA or ROCm "
            "devices are available."
        )
    return device
