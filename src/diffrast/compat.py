"""Run code written for nvdiffrast on diffrast without editing it."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType

from diffrast import log

_displaced: dict[str, ModuleType] = {}


def install_as_nvdiffrast(force: bool = False) -> None:
    """Make ``import nvdiffrast.torch as dr`` resolve to diffrast in this process.

    Registers ``diffrast`` and ``diffrast.torch`` in ``sys.modules`` under the names
    ``nvdiffrast`` and ``nvdiffrast.torch``. Call it before importing the code that uses
    nvdiffrast; modules that imported the real package earlier keep using it. Calling it again
    does nothing, and :func:`uninstall_as_nvdiffrast` undoes it.

    Args:
        force: Take the place of a real nvdiffrast package that is installed or already
            imported. Without it, that raises ``RuntimeError``.
    """
    ours = _modules()
    current = {name: sys.modules.get(name) for name in ours}
    if all(current[name] is module for name, module in ours.items()):
        return
    imported = [name for name, module in ours.items() if current[name] not in (None, module)]
    if not force:
        if imported:
            raise RuntimeError(
                f"{imported[0]} is already imported; pass force=True to replace it with diffrast."
            )
        if current["nvdiffrast"] is None and _is_installed("nvdiffrast"):
            raise RuntimeError(
                "An nvdiffrast package is installed; pass force=True to use diffrast in its place."
            )
    if imported:
        log.warn(
            "Replacing the imported nvdiffrast with diffrast; modules that imported it "
            "earlier keep using it.",
            stacklevel=2,
        )
    for name, module in ours.items():
        if name in imported:
            _displaced[name] = current[name]
        sys.modules[name] = module


def uninstall_as_nvdiffrast() -> None:
    """Undo :func:`install_as_nvdiffrast`, restoring any modules it replaced."""
    for name, module in _modules().items():
        previous = _displaced.pop(name, None)
        if sys.modules.get(name) is module:
            del sys.modules[name]
            if previous is not None:
                sys.modules[name] = previous


def _modules() -> dict[str, ModuleType]:
    return {
        "nvdiffrast": importlib.import_module("diffrast"),
        "nvdiffrast.torch": importlib.import_module("diffrast.torch"),
    }


def _is_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
