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
    from diffrast.backends._triton_rasterize import visible_triangles
    from diffrast.backends.reference import rasterize_from_ids

    if tri.shape[0] == 0:
        empty = pos.new_zeros(pos.shape[0], *resolution, 4)
        return empty, empty.clone()
    tri_id = visible_triangles(pos, tri, resolution)
    return rasterize_from_ids(pos, tri, tri_id, grad_db)


def interpolate(
    attr: Tensor, rast: Tensor, tri: Tensor, rast_db: Tensor | None, diff_attrs: Sequence[int]
) -> tuple[Tensor, Tensor | None]:
    from diffrast.backends._triton_interpolate import Interpolate

    diff = torch.tensor(list(diff_attrs), dtype=torch.int32, device=rast.device)
    out, out_da = Interpolate.apply(attr, rast, rast_db, tri, diff)
    return out, (out_da if diff_attrs else None)


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None,
    filter_mode: str,
    boundary_mode: str,
    max_mip_level: int | None,
) -> Tensor:
    from diffrast.backends import _triton_texture

    return _triton_texture.texture(tex, uv, uv_da, filter_mode, boundary_mode, max_mip_level)


def antialias(
    color: Tensor, rast: Tensor, pos: Tensor, tri: Tensor, pos_gradient_boost: float
) -> Tensor:
    from diffrast.backends import _triton_antialias

    return _triton_antialias.antialias(color, rast, pos, tri, pos_gradient_boost)
