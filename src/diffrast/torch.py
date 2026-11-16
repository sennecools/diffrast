"""The public API under the module path that nvdiffrast uses.

``import diffrast.torch as dr`` gives the same names as ``import diffrast as dr``, so code written
for nvdiffrast ports by replacing ``nvdiffrast`` with ``diffrast`` in its import line. Every name
that ``diffrast`` exports is available here.

The module shares its name with PyTorch only inside this package; ``import torch`` always means
PyTorch, because imports are absolute.
"""

if __name__ == "torch":
    raise ImportError(
        "'import torch' found diffrast/torch.py instead of PyTorch, because the diffrast package "
        "directory is the working directory or is on sys.path. Run from another directory, or "
        "put the directory that contains the diffrast package on sys.path instead."
    )

from diffrast import *  # noqa: F403
from diffrast import __all__ as __all__
