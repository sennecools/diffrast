"""Edge adjacency of a triangle mesh, the part of silhouette detection that depends only on ``tri``.

:func:`diffrast.antialias` needs to know which triangle shares each edge. Finding that takes a
sort over all edges, so it is done once per index buffer: explicitly through
:func:`diffrast.ops.antialias_construct_topology_hash`, or automatically through a cache keyed on
the ``tri`` tensor object.
"""

from __future__ import annotations

import weakref

from torch import Tensor

from diffrast._cache import TensorCache, stamp
from diffrast.backends.reference import edge_neighbours


class TopologyHash:
    """Edge adjacency of an index buffer, as built by ``antialias_construct_topology_hash``.

    The contents are an implementation detail. Pass the object to :func:`diffrast.antialias`
    together with the ``tri`` it was built from, or with a tensor of the same contents.
    """

    def __init__(self, tri: Tensor) -> None:
        self._neighbours = edge_neighbours(tri)
        self._shape = tuple(tri.shape)
        self._device = tri.device
        self._source: weakref.ref[Tensor] | None = weakref.ref(tri)
        self._stamp = stamp(tri)

    def __repr__(self) -> str:
        return f"TopologyHash(triangles={self._shape[0]}, device={str(self._device)!r})"

    def __getstate__(self) -> dict:
        # A copy or an unpickled object keeps the adjacency but not the link to the tensor it
        # was built from, which cannot be pickled and only serves to notice in-place edits.
        return {**self.__dict__, "_source": None, "_stamp": None}

    def neighbours(self, tri: Tensor) -> Tensor:
        """Return the adjacency, int64 ``[triangles, 3]``, if it can belong to ``tri``."""
        if tuple(tri.shape) != self._shape or tri.device != self._device:
            raise ValueError(
                f"topology_hash was built for a tri of shape {self._shape} on {self._device}, "
                f"got shape {tuple(tri.shape)} on {tri.device}."
            )
        if self._source is not None and self._source() is tri and stamp(tri) != self._stamp:
            raise ValueError(
                "tri was modified in place after topology_hash was built from it; build a new "
                "one with antialias_construct_topology_hash."
            )
        return self._neighbours


# Each entry is twice the size of its index buffer, so only a few are kept.
_neighbour_cache: TensorCache[Tensor] = TensorCache(edge_neighbours, max_entries=16)


def neighbours_for(tri: Tensor, topology_hash: TopologyHash | None) -> Tensor:
    """Edge adjacency of ``tri``, int64 of shape ``[triangles, 3]``.

    Taken from ``topology_hash`` when one is given and from the cache otherwise.
    """
    if topology_hash is None:
        return _neighbour_cache.get(tri)
    if not isinstance(topology_hash, TopologyHash):
        raise TypeError(
            "topology_hash must come from antialias_construct_topology_hash, got "
            f"{type(topology_hash).__name__}."
        )
    return topology_hash.neighbours(tri)
