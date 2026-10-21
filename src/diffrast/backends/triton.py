"""Triton implementation of the ops, for CUDA and ROCm devices."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

# Ops with a Triton kernel. "auto" backend selection routes the others to the reference backend.
IMPLEMENTED: frozenset[str] = frozenset()


def rasterize(
    pos: Tensor, tri: Tensor, resolution: tuple[int, int], grad_db: bool
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("rasterize is not implemented in the triton backend yet.")


def interpolate(
    attr: Tensor, rast: Tensor, tri: Tensor, rast_db: Tensor | None, diff_attrs: Sequence[int]
) -> tuple[Tensor, Tensor | None]:
    raise NotImplementedError("interpolate is not implemented in the triton backend yet.")


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None,
    filter_mode: str,
    boundary_mode: str,
    max_mip_level: int | None,
) -> Tensor:
    raise NotImplementedError("texture is not implemented in the triton backend yet.")


def antialias(
    color: Tensor, rast: Tensor, pos: Tensor, tri: Tensor, pos_gradient_boost: float
) -> Tensor:
    raise NotImplementedError("antialias is not implemented in the triton backend yet.")
