"""Triton implementation of the ops, for CUDA and ROCm devices."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

# Ops with a Triton kernel. "auto" backend selection routes the others to the reference backend.
IMPLEMENTED: frozenset[str] = frozenset({"rasterize", "interpolate", "texture", "antialias"})


def rasterize(
    pos: Tensor, tri: Tensor, resolution: tuple[int, int], grad_db: bool
) -> tuple[Tensor, Tensor]:
    rast, rast_db, _ = rasterize_layer(pos, tri, resolution, grad_db, None)
    return rast, rast_db


def rasterize_layer(
    pos: Tensor,
    tri: Tensor,
    resolution: tuple[int, int],
    grad_db: bool,
    previous: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rasterize the nearest layer behind ``previous``, the state returned for the layer before.

    The state is the packed ``(depth, triangle)`` visibility key per pixel.
    """
    from diffrast.backends._triton_rasterize import EMPTY, visible_keys

    if tri.shape[0] == 0 or pos.shape[0] == 0:
        empty = pos.new_zeros(pos.shape[0], *resolution, 4)
        keys = torch.full((pos.shape[0], *resolution), EMPTY, device=pos.device)
        return empty, empty.clone(), keys
    keys = visible_keys(pos, tri, resolution, previous)
    rast, rast_db = resolve(pos, tri, keys, grad_db)
    return rast, rast_db, keys


def resolve(pos: Tensor, tri: Tensor, keys: Tensor, grad_db: bool) -> tuple[Tensor, Tensor]:
    """Build ``rast`` and ``rast_db`` from the visibility keys of :func:`rasterize`."""
    from diffrast.backends import _triton_resolve

    return _triton_resolve.resolve(pos, tri, keys, grad_db)


def interpolate(
    attr: Tensor, rast: Tensor, tri: Tensor, rast_db: Tensor | None, diff_attrs: Sequence[int]
) -> tuple[Tensor, Tensor]:
    from diffrast.backends._triton_interpolate import Interpolate

    diff = torch.tensor(list(diff_attrs), dtype=torch.int32, device=rast.device)
    return Interpolate.apply(attr, rast, rast_db, tri, diff)


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None,
    filter_mode: str,
    boundary_mode: str,
    max_mip_level: int | None,
    mip_level_bias: Tensor | None = None,
    mip: Sequence[Tensor] | None = None,
    packed: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    args = (tex, uv, uv_da, filter_mode, boundary_mode, max_mip_level, mip_level_bias, mip, packed)
    if boundary_mode == "cube":
        # The kernels are 2D only. Cube maps use the PyTorch implementation, which runs on the
        # GPU as well.
        from diffrast.backends import reference

        return reference.texture(*args)
    from diffrast.backends import _triton_texture

    return _triton_texture.texture(*args)


def antialias(
    color: Tensor,
    rast: Tensor,
    pos: Tensor,
    tri: Tensor,
    pos_gradient_boost: float,
    neighbours: Tensor | None = None,
) -> Tensor:
    from diffrast.backends import _triton_antialias

    return _triton_antialias.antialias(color, rast, pos, tri, pos_gradient_boost, neighbours)
