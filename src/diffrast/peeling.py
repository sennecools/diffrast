from __future__ import annotations

from types import TracebackType

from torch import Tensor

from diffrast.backends import get_backend
from diffrast.context import RasterizeContext
from diffrast.ops import check_rasterize_inputs, join_ranges, split_ranges


class DepthPeeler:
    """Rasterize successive depth layers of the same geometry.

    The first call to :meth:`rasterize_next_layer` gives the same result as :func:`rasterize`.
    Each later call gives, per pixel, the nearest triangle behind the one returned before, until
    a pixel runs out of triangles and stays empty. Triangles are ordered by depth and then by
    index, so triangles whose depths at a pixel are exactly equal are returned in turn, in order
    of their index. Every layer has the same format as :func:`rasterize`'s output, so it works
    with :func:`interpolate`, :func:`texture` and :func:`antialias` and passes gradients back to
    ``pos``.

    Arguments are as for :func:`rasterize`, including range mode. The peeler keeps its own copy
    of ``pos``, ``tri`` and ``ranges``, so every layer shows the geometry as it was when the
    peeler was created, whatever is done to those tensors afterwards; gradients still reach the
    ``pos`` that was passed in. It can be used as a context manager, after which no more layers
    can be rasterized::

        with dr.DepthPeeler(glctx, pos, tri, (512, 512)) as peeler:
            for _ in range(num_layers):
                rast, rast_db = peeler.rasterize_next_layer()
    """

    def __init__(
        self,
        glctx: RasterizeContext,
        pos: Tensor,
        tri: Tensor,
        resolution: tuple[int, int],
        ranges: Tensor | None = None,
        grad_db: bool = True,
    ) -> None:
        self._resolution = check_rasterize_inputs(glctx, pos, tri, resolution, ranges)
        self._output_db = getattr(glctx, "output_db", True)
        self._grad_db = grad_db and self._output_db
        # Each layer is found from the per-pixel state of the one before, which belongs to the
        # geometry it was rasterized from. Copies keep in-place edits made between layers out.
        self._pos, self._tri = pos.clone(), tri.clone()
        # In range mode every batch item is peeled on its own, from its own slice of tri.
        self._ranges = ranges.clone() if pos.ndim == 2 else None
        # Each backend keeps its own per-pixel state between layers, so the backend is fixed here.
        self._backend = get_backend(pos.device, "rasterize")
        self._state = None
        self._closed = False

    def __enter__(self) -> DepthPeeler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._state = None
        self._closed = True

    def rasterize_next_layer(self) -> tuple[Tensor, Tensor]:
        """Return ``(rast, rast_db)`` for the next depth layer."""
        if self._closed:
            raise RuntimeError(
                "This DepthPeeler was closed when its with block exited; create a new one to "
                "rasterize more layers."
            )
        if self._ranges is None:
            rast, rast_db, self._state = self._backend.rasterize_layer(
                self._pos, self._tri, self._resolution, self._grad_db, self._state
            )
        else:
            triangles = split_ranges(self._tri, self._ranges)
            states = self._state or [None] * len(triangles)
            layers = [
                self._backend.rasterize_layer(
                    self._pos[None], item, self._resolution, self._grad_db, state
                )
                for item, state in zip(triangles, states, strict=True)
            ]
            self._state = [state for _, _, state in layers]
            rast, rast_db = join_ranges(
                self._pos, self._resolution, self._ranges, [layer[:2] for layer in layers]
            )
        return rast, (rast_db if self._output_db else rast.new_zeros(*rast.shape[:3], 0))
