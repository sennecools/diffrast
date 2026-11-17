"""Cube-map geometry for ``texture``: face selection and lookups across face edges.

Faces are stored in the order ``+x, -x, +y, -y, +z, -z`` and oriented as in the OpenGL
specification. For a direction ``r`` whose largest component is along the face's major axis, the
face coordinates are ``s = (sc / |ma| + 1) / 2`` and ``t = (tc / |ma| + 1) / 2`` with

====  ====  ====  ====
face   sc    tc    ma
====  ====  ====  ====
 +x   -rz   -ry    rx
 -x   +rz   -ry    rx
 +y   +rx   +rz    ry
 -y   +rx   -rz    ry
 +z   +rx   -ry    rz
 -z   -rx   -ry    rz
====  ====  ====  ====

``s`` runs along the columns of a face and ``t`` along its rows. Where components tie for the
largest magnitude, x takes precedence over y and y over z. Everything here is plain PyTorch, so
it runs on any device and gradients come from autograd.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor

# Per face: the major axis, and the directions in which s and t increase.
_MAJOR = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_S_AXIS = ((0, 0, -1), (0, 0, 1), (1, 0, 0), (1, 0, 0), (1, 0, 0), (-1, 0, 0))
_T_AXIS = ((0, -1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0), (0, -1, 0))
# Derivatives of a direction saturate at this multiple of its length, like footprints in texels.
# An infinite derivative then selects the coarsest level instead of cancelling to NaN.
_DERIVATIVE_LIMIT = 1e18


@lru_cache(maxsize=16)
def _axes(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    return tuple(torch.tensor(t, device=device, dtype=dtype) for t in (_MAJOR, _S_AXIS, _T_AXIS))


def project(
    dirs: Tensor, dirs_da: Tensor | None = None
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    """Project direction vectors onto the cube.

    Args:
        dirs: Directions of shape ``[..., 3]``; they need not be normalised.
        dirs_da: Screen-space derivatives ``(d/dx, d/dy)`` of each component of ``dirs``, shape
            ``[..., 6]``, or ``None``.

    Returns:
        ``(face, st, st_da, valid)``. ``face`` is the int64 face index of shape ``[...]``, ``st``
        the face coordinates in ``[0, 1]`` of shape ``[..., 2]``, and ``st_da`` their screen-space
        derivatives ``(ds/dx, ds/dy, dt/dx, dt/dy)`` of shape ``[..., 4]`` (``None`` without
        ``dirs_da``). ``valid`` is false for directions that select no point on the cube:
        vectors with a component that is not finite, the zero vector, and vectors so short that
        their largest component is subnormal, which have lost the precision a direction needs.
        Those get a placeholder that passes no gradient to ``dirs`` or ``dirs_da``.

        Derivatives saturate far beyond any footprint, so huge and infinite ones stay finite.
        Where a derivative of a valid direction is NaN, ``st_da`` is NaN and no gradient passes
        to ``dirs`` or ``dirs_da`` through it.
    """
    shortest = torch.finfo(dirs.dtype).tiny
    valid = torch.isfinite(dirs).all(-1) & (dirs.detach().abs().amax(-1) >= shortest)
    dirs = torch.where(valid[..., None], dirs, dirs.new_tensor((0.0, 0.0, 1.0)))

    ax, ay, az = dirs.detach().abs().unbind(-1)
    x_major = (ax >= ay) & (ax >= az)
    y_major = ~x_major & (ay >= az)
    axis = 2 - 2 * x_major.long() - y_major.long()
    negative = dirs.detach().gather(-1, axis[..., None]).squeeze(-1) < 0
    face = 2 * axis + negative.long()

    # The face coordinates do not depend on the length of the direction. With the major
    # component scaled to one, neither they nor their gradients overflow or underflow on the way,
    # however long or short the direction is.
    length = dirs.detach().abs().amax(-1, keepdim=True)
    dirs = dirs / length
    major, s_axis, t_axis = (table[face] for table in _axes(dirs.device, dirs.dtype))
    ma = (dirs * major).sum(-1, keepdim=True)
    ratio = torch.stack([(dirs * s_axis).sum(-1), (dirs * t_axis).sum(-1)], -1) / ma
    st = 0.5 * (ratio + 1)
    if dirs_da is None:
        return face, st, None, valid

    # A NaN derivative is kept out of the arithmetic, where it would turn the gradient of the
    # direction into NaN as well, and put back at the end.
    undefined = valid & dirs_da.isnan().any(-1)
    da = torch.where((valid & ~undefined)[..., None], dirs_da, 0) / length
    da = da.clamp(-_DERIVATIVE_LIMIT, _DERIVATIVE_LIMIT).unflatten(-1, (3, 2))
    d_ma = (da * major[..., None]).sum(-2)
    d_sc = (da * s_axis[..., None]).sum(-2)
    d_tc = (da * t_axis[..., None]).sum(-2)
    # Quotient rule for sc / ma.
    ds = 0.5 * (d_sc - ratio[..., :1] * d_ma) / ma
    dt = 0.5 * (d_tc - ratio[..., 1:] * d_ma) / ma
    st_da = torch.where(undefined[..., None], torch.nan, torch.cat([ds, dt], -1))
    return face, st, st_da, valid


def fold(
    face: Tensor, row: Tensor, col: Tensor, size: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve texel coordinates that may lie one step outside their face.

    Args:
        face: Face indices, int64.
        row: Texel rows in ``[-1, size]``, int64.
        col: Texel columns in ``[-1, size]``, int64.
        size: Edge length in texels of the face each coordinate is on, int64.

    All four broadcast against each other.

    Returns:
        ``(face, row, col, exists)``. A texel outside one edge becomes the texel of the
        neighbouring face that touches it in 3D; texels inside are returned unchanged. A texel
        outside two edges lies beyond a corner of the cube, where no face continues: ``exists``
        is false and the returned coordinates are an arbitrary texel.
    """
    major, s_axis, t_axis = (table[face] for table in _axes(face.device, torch.long))
    over_col = (col < 0) | (col >= size)
    over_row = (row < 0) | (row >= size)

    # In units of half a texel the cube spans [-size, size] and texel centres are integers, so
    # the lookup is exact. A centre sticking out past an edge at size + 1 is folded around that
    # edge: onto the neighbouring face's plane, at size - 1 along the old major axis.
    extent = size[..., None]
    sc = (2 * col + 1 - size)[..., None]
    tc = (2 * row + 1 - size)[..., None]
    point = (extent * major + sc * s_axis + tc * t_axis).clamp(-extent, extent)
    point = point - (over_col | over_row)[..., None] * major

    axis = point.abs().argmax(-1)
    negative = point.gather(-1, axis[..., None]).squeeze(-1) < 0
    face = 2 * axis + negative.long()
    _, s_axis, t_axis = (table[face] for table in _axes(face.device, torch.long))
    col = ((point * s_axis).sum(-1) + size - 1) // 2
    row = ((point * t_axis).sum(-1) + size - 1) // 2
    return face, row.clamp(min=0), col.clamp(min=0), ~(over_col & over_row)
