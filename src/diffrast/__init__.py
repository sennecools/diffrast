"""Portable differentiable mesh rasterization for PyTorch."""

from diffrast.backends import get_backend_name, set_backend
from diffrast.context import RasterizeContext, RasterizeCudaContext
from diffrast.ops import antialias, interpolate, rasterize, texture
from diffrast.peeling import DepthPeeler

__version__ = "0.1.0.dev0"

__all__ = [
    "DepthPeeler",
    "RasterizeContext",
    "RasterizeCudaContext",
    "__version__",
    "antialias",
    "get_backend_name",
    "interpolate",
    "rasterize",
    "set_backend",
    "texture",
]
