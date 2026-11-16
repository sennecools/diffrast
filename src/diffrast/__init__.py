"""Portable differentiable mesh rasterization for PyTorch."""

from diffrast.backends import get_backend_name, set_backend
from diffrast.compat import install_as_nvdiffrast, uninstall_as_nvdiffrast
from diffrast.context import RasterizeContext, RasterizeCudaContext, RasterizeGLContext
from diffrast.log import get_log_level, set_log_level
from diffrast.ops import (
    MipStack,
    antialias,
    antialias_construct_topology_hash,
    interpolate,
    rasterize,
    texture,
    texture_construct_mip,
)
from diffrast.peeling import DepthPeeler
from diffrast.topology import TopologyHash

__version__ = "0.1.0.dev0"

__all__ = [
    "DepthPeeler",
    "MipStack",
    "RasterizeContext",
    "RasterizeCudaContext",
    "RasterizeGLContext",
    "TopologyHash",
    "__version__",
    "antialias",
    "antialias_construct_topology_hash",
    "get_backend_name",
    "get_log_level",
    "install_as_nvdiffrast",
    "interpolate",
    "rasterize",
    "set_backend",
    "set_log_level",
    "texture",
    "texture_construct_mip",
    "uninstall_as_nvdiffrast",
]
