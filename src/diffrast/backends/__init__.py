"""Backend selection. Each backend module implements the four ops on validated inputs."""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import torch

BACKENDS = ("reference", "triton")
OPS = ("rasterize", "interpolate", "texture", "antialias")

_selected = os.environ.get("DIFFRAST_BACKEND", "auto")


def set_backend(name: str) -> None:
    """Select the backend used by all ops.

    Args:
        name: ``"triton"``, ``"reference"`` or ``"auto"``. ``"auto"`` (the default) uses Triton
            for GPU tensors when it is installed and implements the op, and the pure-PyTorch
            reference otherwise. The initial value can also be set with the ``DIFFRAST_BACKEND``
            environment variable.
    """
    global _selected
    _check_name(name)
    _selected = name


def get_backend_name(device: torch.device | str, op: str) -> str:
    """Return the name of the backend that ``op`` will run on for tensors on ``device``."""
    _check_name(_selected)
    if op not in OPS:
        raise ValueError(f"Unknown op {op!r}; expected one of {OPS}.")
    device = torch.device(device)
    if _selected == "auto":
        if _triton_usable(device) and op in _load("triton").IMPLEMENTED:
            return "triton"
        return "reference"
    if _selected == "triton" and not _triton_usable(device):
        raise RuntimeError(
            "The triton backend needs a CUDA or ROCm device and the triton package; "
            f"got device {str(device)!r}."
        )
    return _selected


def get_backend(device: torch.device | str, op: str) -> ModuleType:
    """Return the backend module that ``op`` will run on for tensors on ``device``."""
    return _load(get_backend_name(device, op))


def _load(name: str) -> ModuleType:
    return importlib.import_module(f"diffrast.backends.{name}")


def _check_name(name: str) -> None:
    if name not in ("auto", *BACKENDS):
        options = ", ".join(repr(n) for n in ("auto", *BACKENDS))
        raise ValueError(f"Unknown backend {name!r}; expected one of {options}.")


def _triton_usable(device: torch.device) -> bool:
    return device.type == "cuda" and importlib.util.find_spec("triton") is not None
