"""Backend selection. Each backend module implements the four ops on validated inputs."""

from __future__ import annotations

import importlib
import importlib.util
import os
from types import ModuleType

import torch

BACKENDS = ("reference", "triton")

_selected = os.environ.get("DIFFRAST_BACKEND", "auto")


def set_backend(name: str) -> None:
    """Select the backend used by all ops.

    Args:
        name: ``"triton"``, ``"reference"`` or ``"auto"``. ``"auto"`` (the default) uses Triton
            for GPU tensors when it is installed and the pure-PyTorch reference otherwise. The
            initial value can also be set with the ``DIFFRAST_BACKEND`` environment variable.
    """
    global _selected
    _check_name(name)
    _selected = name


def get_backend_name(device: torch.device | str) -> str:
    """Return the name of the backend that ops on ``device`` will run on."""
    _check_name(_selected)
    device = torch.device(device)
    if _selected == "auto":
        return "triton" if _triton_usable(device) else "reference"
    if _selected == "triton" and not _triton_usable(device):
        raise RuntimeError(
            "The triton backend needs a CUDA or ROCm device and the triton package; "
            f"got device {str(device)!r}."
        )
    return _selected


def get_backend(device: torch.device | str) -> ModuleType:
    """Return the backend module that ops on ``device`` will run on."""
    return importlib.import_module(f"diffrast.backends.{get_backend_name(device)}")


def _check_name(name: str) -> None:
    if name not in ("auto", *BACKENDS):
        options = ", ".join(repr(n) for n in ("auto", *BACKENDS))
        raise ValueError(f"Unknown backend {name!r}; expected one of {options}.")


def _triton_usable(device: torch.device) -> bool:
    return device.type == "cuda" and importlib.util.find_spec("triton") is not None
