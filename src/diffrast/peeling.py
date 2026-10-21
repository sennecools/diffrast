from __future__ import annotations

from torch import Tensor

from diffrast.backends import get_backend
from diffrast.context import RasterizeContext
from diffrast.ops import check_rasterize_inputs


class DepthPeeler:
    """Rasterize successive depth layers of the same geometry.

    The first call to :meth:`rasterize_next_layer` gives the same result as :func:`rasterize`.
    Each later call gives, per pixel, the nearest triangle behind the one returned before, until
    a pixel runs out of triangles and stays empty. Triangles at exactly the same depth are
    returned in turn, in order of their index. Every layer has the same format as
    :func:`rasterize`'s output, so it works with :func:`interpolate`, :func:`texture` and
    :func:`antialias` and passes gradients back to ``pos``.

    Arguments are as for :func:`rasterize`. It can be used as a context manager::

        with dr.DepthPeeler(ctx, pos, tri, (512, 512)) as peeler:
            for _ in range(num_layers):
                rast, rast_db = peeler.rasterize_next_layer()
    """

    def __init__(
        self,
        ctx: RasterizeContext,
        pos: Tensor,
        tri: Tensor,
        resolution: tuple[int, int],
        ranges: Tensor | None = None,
        grad_db: bool = True,
    ) -> None:
        self._resolution = check_rasterize_inputs(ctx, pos, tri, resolution, ranges)
        self._pos, self._tri, self._grad_db = pos, tri, grad_db
        # Each backend keeps its own per-pixel state between layers, so the backend is fixed here.
        self._backend = get_backend(pos.device, "rasterize")
        self._state = None

    def __enter__(self) -> DepthPeeler:
        return self

    def __exit__(self, *exc_info) -> None:
        self._state = None

    def rasterize_next_layer(self) -> tuple[Tensor, Tensor]:
        """Return ``(rast, rast_db)`` for the next depth layer."""
        rast, rast_db, self._state = self._backend.rasterize_layer(
            self._pos, self._tri, self._resolution, self._grad_db, self._state
        )
        return rast, rast_db
