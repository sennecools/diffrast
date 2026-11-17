"""Helpers shared by the autograd functions of the Triton backend."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch


def first_order(op: str) -> Callable[[Callable], Callable]:
    """Decorator for a ``backward`` whose kernels cannot be differentiated again.

    PyTorch runs ``backward`` with gradient tracking enabled exactly when a differentiable
    gradient was asked for (``create_graph=True``), so that is refused here with an error naming
    ``op``. ``torch.autograd.function.once_differentiable`` is not enough: it only fails once
    the second backward pass reaches its error node, which ``torch.autograd.grad`` skips when
    the requested inputs are also reachable another way, and it stays silent when the incoming
    gradient does not require grad. Both return gradients that lack the second-order terms.
    """

    def decorate(backward: Callable) -> Callable:
        @functools.wraps(backward)
        def guarded(ctx, *grads):
            if torch.is_grad_enabled():
                raise RuntimeError(
                    f"{op} on the Triton backend has no second-order gradients, but its "
                    "backward pass was run with create_graph=True. Use "
                    "diffrast.set_backend('reference') for gradients of gradients."
                )
            return backward(ctx, *grads)

        return guarded

    return decorate
