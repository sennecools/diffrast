"""Public differentiable rasterization ops.

Each op validates its inputs and dispatches to the active backend (see
:func:`diffrast.set_backend`). All ops accept a leading batch dimension so several views can be
rendered in one call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

import torch
from torch import Tensor

from diffrast.backends import get_backend
from diffrast.context import RasterizeContext

FilterMode = Literal["auto", "nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear"]
BoundaryMode = Literal["wrap", "clamp", "zero"]

FILTER_MODES = get_args(FilterMode)
BOUNDARY_MODES = get_args(BoundaryMode)


def rasterize(
    ctx: RasterizeContext,
    pos: Tensor,
    tri: Tensor,
    resolution: tuple[int, int],
    ranges: Tensor | None = None,
    grad_db: bool = True,
) -> tuple[Tensor, Tensor]:
    """Rasterize triangles into per-pixel barycentrics, depth and triangle ids.

    Args:
        ctx: Rasterization context.
        pos: Clip-space vertex positions, float32 of shape ``[batch, vertices, 4]``.
        tri: Vertex indices, int32 of shape ``[triangles, 3]``.
        resolution: Output size as ``(height, width)``.
        ranges: Range mode is not supported yet; must be ``None``.
        grad_db: Propagate gradients from ``rast_db`` back to ``pos``.

    Returns:
        ``(rast, rast_db)``, both of shape ``[batch, height, width, 4]``. Each pixel of ``rast``
        holds ``(u, v, z / w, triangle_id + 1)``, and is all zeros where no triangle covers it.
        ``rast_db`` holds the screen-space derivatives ``(du/dx, du/dy, dv/dx, dv/dy)``.
    """
    if ranges is not None:
        raise NotImplementedError("Range mode (ranges=...) is not supported yet.")
    _check(pos, "pos", ndim=3, last=4, dtype=torch.float32)
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    _check_devices(pos=pos, tri=tri)
    if pos.device != ctx.device:
        raise ValueError(f"pos is on {pos.device} but the context is on {ctx.device}.")
    height, width = _check_resolution(resolution)
    return get_backend(pos.device, "rasterize").rasterize(pos, tri, (height, width), grad_db)


def interpolate(
    attr: Tensor,
    rast: Tensor,
    tri: Tensor,
    rast_db: Tensor | None = None,
    diff_attrs: Sequence[int] | Literal["all"] | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Interpolate vertex attributes across the rasterized triangles.

    Args:
        attr: Vertex attributes, float32 of shape ``[batch, vertices, channels]``, or
            ``[vertices, channels]`` to share them across the batch.
        rast: First output of :func:`rasterize`.
        tri: Vertex indices used to produce ``rast``.
        rast_db: Second output of :func:`rasterize`. Required when ``diff_attrs`` is set.
        diff_attrs: Attribute channels to compute screen-space derivatives for, or ``"all"``.

    Returns:
        ``(out, out_da)``. ``out`` has shape ``[batch, height, width, channels]``. ``out_da`` has
        shape ``[batch, height, width, 2 * len(diff_attrs)]`` with ``(d/dx, d/dy)`` per selected
        channel, or is ``None`` when ``diff_attrs`` is not set.
    """
    if attr.ndim not in (2, 3):
        raise ValueError(f"attr must have 2 or 3 dimensions, got shape {tuple(attr.shape)}.")
    _check(attr, "attr", ndim=attr.ndim, dtype=torch.float32)
    _check(rast, "rast", ndim=4, last=4, dtype=torch.float32)
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    if attr.ndim == 3 and attr.shape[0] not in (1, rast.shape[0]):
        raise ValueError(
            f"attr batch size {attr.shape[0]} does not match rast batch size {rast.shape[0]}."
        )

    if diff_attrs is None:
        channels: list[int] = []
    elif diff_attrs == "all":
        channels = list(range(attr.shape[-1]))
    else:
        channels = [int(c) for c in diff_attrs]
        if any(c < 0 or c >= attr.shape[-1] for c in channels):
            raise ValueError(f"diff_attrs {channels} out of range for {attr.shape[-1]} channels.")
    if channels:
        if rast_db is None:
            raise ValueError("rast_db is required when diff_attrs is set.")
        _check(rast_db, "rast_db", ndim=4, last=4, dtype=torch.float32)
        if rast_db.shape != rast.shape:
            raise ValueError("rast_db must have the same shape as rast.")

    _check_devices(attr=attr, rast=rast, tri=tri, rast_db=rast_db)
    return get_backend(rast.device, "interpolate").interpolate(attr, rast, tri, rast_db, channels)


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None = None,
    mip_level_bias: Tensor | None = None,
    mip: Sequence[Tensor] | None = None,
    filter_mode: FilterMode = "auto",
    boundary_mode: BoundaryMode = "wrap",
    max_mip_level: int | None = None,
) -> Tensor:
    """Sample a 2D texture at per-pixel texture coordinates.

    Args:
        tex: Texture, float32 of shape ``[batch, tex_height, tex_width, channels]``. A batch of
            one is shared across all views.
        uv: Texture coordinates, float32 of shape ``[batch, height, width, 2]``. ``u`` runs along
            the texture width and ``v`` along its height; ``(0, 0)`` is the outer corner of texel
            ``[0, 0]`` and ``(1, 1)`` the outer corner of the last texel.
        uv_da: Screen-space derivatives ``(du/dx, du/dy, dv/dx, dv/dy)`` of ``uv``, shape
            ``[batch, height, width, 4]``, as returned by :func:`interpolate`. Required for the
            mipmapped filter modes.
        mip_level_bias: Not supported yet; must be ``None``.
        mip: Precomputed mip levels are not supported yet; must be ``None``.
        filter_mode: ``"auto"`` picks ``"linear-mipmap-linear"`` when ``uv_da`` is given and
            ``"linear"`` otherwise.
        boundary_mode: Coordinates outside ``[0, 1]`` repeat the texture (``"wrap"``), repeat its
            edge texels (``"clamp"``), or read zeros (``"zero"``).
        max_mip_level: Index of the coarsest mip level to use, or ``None`` for a full chain.
            ``0`` disables mipmapping.

    Returns:
        Sampled values of shape ``[batch, height, width, channels]``.
    """
    if mip_level_bias is not None or mip is not None:
        raise NotImplementedError("mip_level_bias and mip are not supported yet.")
    _check(tex, "tex", ndim=4, dtype=torch.float32)
    _check(uv, "uv", ndim=4, last=2, dtype=torch.float32)
    if tex.shape[0] not in (1, uv.shape[0]):
        raise ValueError(
            f"tex batch size {tex.shape[0]} does not match uv batch size {uv.shape[0]}."
        )
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"Unknown filter_mode {filter_mode!r}; expected one of {FILTER_MODES}.")
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"Unknown boundary_mode {boundary_mode!r}; expected one of {BOUNDARY_MODES}."
        )
    if filter_mode == "auto":
        filter_mode = "linear-mipmap-linear" if uv_da is not None else "linear"
    if "mipmap" in filter_mode and uv_da is None:
        raise ValueError(f"filter_mode {filter_mode!r} requires uv_da.")
    if uv_da is not None:
        _check(uv_da, "uv_da", ndim=4, last=4, dtype=torch.float32)
        if uv_da.shape[:3] != uv.shape[:3]:
            raise ValueError("uv_da must match uv in batch, height and width.")
    if max_mip_level is not None and max_mip_level < 0:
        raise ValueError(f"max_mip_level must be non-negative, got {max_mip_level}.")

    _check_devices(tex=tex, uv=uv, uv_da=uv_da)
    return get_backend(uv.device, "texture").texture(
        tex, uv, uv_da, filter_mode, boundary_mode, max_mip_level
    )


def antialias(
    color: Tensor,
    rast: Tensor,
    pos: Tensor,
    tri: Tensor,
    topology_hash: Tensor | None = None,
    pos_gradient_boost: float = 1.0,
) -> Tensor:
    """Antialias silhouette edges so that visibility changes produce gradients in ``pos``.

    Args:
        color: Shaded image, float32 of shape ``[batch, height, width, channels]``.
        rast: First output of :func:`rasterize` for the same geometry.
        pos: Clip-space vertex positions used to produce ``rast``.
        tri: Vertex indices used to produce ``rast``.
        topology_hash: Accepted for API compatibility and ignored.
        pos_gradient_boost: Multiplier applied to the gradients flowing into ``pos``.

    Returns:
        Antialiased image with the same shape as ``color``.
    """
    _check(color, "color", ndim=4, dtype=torch.float32)
    _check(rast, "rast", ndim=4, last=4, dtype=torch.float32)
    _check(pos, "pos", ndim=3, last=4, dtype=torch.float32)
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    if color.shape[:3] != rast.shape[:3]:
        raise ValueError("color must match rast in batch, height and width.")
    if pos.shape[0] != rast.shape[0]:
        raise ValueError(
            f"pos batch size {pos.shape[0]} does not match rast batch size {rast.shape[0]}."
        )
    _check_devices(color=color, rast=rast, pos=pos, tri=tri)
    return get_backend(color.device, "antialias").antialias(
        color, rast, pos, tri, float(pos_gradient_boost)
    )


def _check(
    t: Tensor,
    name: str,
    ndim: int,
    last: int | None = None,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(t, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}.")
    if t.ndim != ndim or (last is not None and t.shape[-1] != last):
        expected = f"[{', '.join(['...'] * (ndim - 1) + [str(last) if last else '...'])}]"
        raise ValueError(f"{name} must have shape {expected}, got {tuple(t.shape)}.")
    if dtype is not None and t.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}.")


def _check_devices(**tensors: Tensor | None) -> None:
    devices = {name: t.device for name, t in tensors.items() if t is not None}
    if len(set(devices.values())) > 1:
        listing = ", ".join(f"{name} on {device}" for name, device in devices.items())
        raise ValueError(f"All inputs must be on the same device; got {listing}.")


def _check_resolution(resolution: Sequence[int]) -> tuple[int, int]:
    if len(resolution) != 2:
        raise ValueError(f"resolution must be (height, width), got {resolution!r}.")
    height, width = (int(r) for r in resolution)
    if height <= 0 or width <= 0:
        raise ValueError(f"resolution must be positive, got {resolution!r}.")
    return height, width
