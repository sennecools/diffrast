"""Cache for values derived from the contents of a tensor."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

from torch import Tensor

T = TypeVar("T")


def stamp(tensor: Tensor) -> tuple | None:
    """A value that changes when ``tensor`` is modified, or ``None`` if that cannot be told.

    It combines the version counter, which PyTorch increments on every in-place operation, also
    one made through a view, with the address and shape of the data, which change when the
    tensor is given new storage. Writes that bypass the version counter are not seen: in-place
    operations on ``tensor.data``, and writes to memory shared with NumPy or another library.
    """
    if tensor.is_inference():
        # Inference tensors have no version counter.
        return None
    return tensor._version, tensor.data_ptr(), tensor.shape


class TensorCache(Generic[T]):
    """Remembers ``compute(tensor)`` per tensor object until the tensor is modified or freed.

    An entry is keyed on the identity of the tensor object and holds a weak reference to it
    together with its :func:`stamp`. A lookup only hits when both still match, so a modified
    tensor and a new tensor with the same contents are computed again, and an entry is dropped
    as soon as its tensor is freed.

    Args:
        compute: Function of the tensor whose result is cached. The result must not keep the
            tensor alive.
        max_entries: Number of tensors remembered before the oldest entry is dropped.
    """

    def __init__(self, compute: Callable[[Tensor], T], max_entries: int = 64) -> None:
        self._compute = compute
        self._max_entries = max_entries
        self._entries: OrderedDict[int, tuple[weakref.ref[Tensor], tuple, T]] = OrderedDict()
        # Re-entrant, because a weak reference callback can run while this thread holds it.
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, tensor: Tensor) -> T:
        """Return ``compute(tensor)``, reusing the stored result while ``tensor`` is unchanged."""
        current = stamp(tensor)
        if current is None:
            return self._compute(tensor)
        key = id(tensor)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is tensor and entry[1] == current:
            return entry[2]

        value = self._compute(tensor)
        ref = weakref.ref(tensor, lambda ref: self._discard(key, ref))
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (ref, current, value)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
        return value

    def clear(self) -> None:
        """Forget all stored results."""
        with self._lock:
            self._entries.clear()

    def _discard(self, key: int, ref: weakref.ref[Tensor]) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] is ref:
                del self._entries[key]
