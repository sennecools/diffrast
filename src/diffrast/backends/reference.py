"""Pure-PyTorch implementation of the ops.

Slow, but runs on any device and dtype PyTorch supports. It is the ground truth the Triton backend
is tested against. Gradients come from autograd.

Rasterization is done in homogeneous coordinates (Olano and Greer, 1997): for a pixel at
normalized device coordinates ``(px, py)``, the perspective-correct barycentrics ``b`` of a
triangle satisfy ``sum(b * (x - px * w)) = 0`` and ``sum(b * (y - py * w)) = 0``, so ``b`` is the
normalized cross product of those two vectors. A pixel is covered when all ``b >= 0``, the
interpolated ``w`` is positive, and ``-1 <= z / w <= 1``. This needs no near-plane clipping.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

# Upper bound on batch * triangles * pixels evaluated at once during visibility.
_CHUNK_ELEMENTS = 1 << 22


def rasterize(
    pos: Tensor, tri: Tensor, resolution: tuple[int, int], grad_db: bool
) -> tuple[Tensor, Tensor]:
    height, width = resolution
    batch = pos.shape[0]
    rast = pos.new_zeros(batch, height, width, 4)
    rast_db = pos.new_zeros(batch, height, width, 4)
    if tri.shape[0] == 0:
        return rast, rast_db

    tri = tri.long()
    px, py = _pixel_centres(height, width, pos)
    with torch.no_grad():
        tri_id = _visible_triangles(pos, tri, px, py)
    covered = tri_id >= 0

    batch_idx = torch.arange(batch, device=pos.device)[:, None, None, None]
    verts = pos[batch_idx, tri[tri_id.clamp(min=0)]]
    x, y, z, w = verts.unbind(-1)
    px, py = px.view(1, 1, width), py.view(1, height, 1)

    e = _edge_functions(x, y, w, px, py)
    area = torch.where(covered, e.sum(-1), 1.0)
    b = e / area[..., None]
    w_p = torch.where(covered, (b * w).sum(-1), 1.0)
    depth = (b * z).sum(-1) / w_p

    de_dx, de_dy = _edge_derivatives(x, y, w, px, py)
    db_dx = (de_dx - b * de_dx.sum(-1, keepdim=True)) / area[..., None]
    db_dy = (de_dy - b * de_dy.sum(-1, keepdim=True)) / area[..., None]
    if not grad_db:
        db_dx, db_dy = db_dx.detach(), db_dy.detach()

    mask = covered[..., None]
    rast = torch.stack([b[..., 0], b[..., 1], depth, (tri_id + 1).to(pos.dtype)], -1)
    rast = torch.where(mask, rast, 0.0)
    # The pixel grid spans 2 NDC units, so one pixel is 2 / size in NDC.
    rast_db = torch.stack(
        [
            db_dx[..., 0] * (2 / width),
            db_dy[..., 0] * (2 / height),
            db_dx[..., 1] * (2 / width),
            db_dy[..., 1] * (2 / height),
        ],
        -1,
    )
    rast_db = torch.where(mask, rast_db, 0.0)
    return rast, rast_db


def interpolate(
    attr: Tensor, rast: Tensor, tri: Tensor, rast_db: Tensor | None, diff_attrs: Sequence[int]
) -> tuple[Tensor, Tensor | None]:
    batch, height, width, _ = rast.shape
    channels = attr.shape[-1]
    out = attr.new_zeros(batch, height, width, channels)
    out_da = attr.new_zeros(batch, height, width, 2 * len(diff_attrs)) if diff_attrs else None
    if tri.shape[0] == 0:
        return out, out_da

    tri_id = rast[..., 3].round().long() - 1
    covered = (tri_id >= 0)[..., None]
    if attr.ndim == 2:
        attr = attr[None]
    attr = attr.expand(batch, -1, -1)
    batch_idx = torch.arange(batch, device=rast.device)[:, None, None, None]
    a0, a1, a2 = attr[batch_idx, tri.long()[tri_id.clamp(min=0)]].unbind(-2)

    u, v = rast[..., 0:1], rast[..., 1:2]
    out = torch.where(covered, u * a0 + v * a1 + (1 - u - v) * a2, 0.0)
    if not diff_attrs:
        return out, None

    da_du = (a0 - a2)[..., list(diff_attrs)]
    da_dv = (a1 - a2)[..., list(diff_attrs)]
    du_dx, du_dy, dv_dx, dv_dy = rast_db[..., None].unbind(-2)
    d_dx = du_dx * da_du + dv_dx * da_dv
    d_dy = du_dy * da_du + dv_dy * da_dv
    out_da = torch.stack([d_dx, d_dy], -1).flatten(-2)
    return out, torch.where(covered, out_da, 0.0)


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None,
    filter_mode: str,
    boundary_mode: str,
    max_mip_level: int | None,
) -> Tensor:
    raise NotImplementedError("texture is not implemented in the reference backend yet.")


def antialias(
    color: Tensor, rast: Tensor, pos: Tensor, tri: Tensor, pos_gradient_boost: float
) -> Tensor:
    raise NotImplementedError("antialias is not implemented in the reference backend yet.")


def _pixel_centres(height: int, width: int, like: Tensor) -> tuple[Tensor, Tensor]:
    px = (torch.arange(width, dtype=like.dtype, device=like.device) * 2 + 1) / width - 1
    py = (torch.arange(height, dtype=like.dtype, device=like.device) * 2 + 1) / height - 1
    return px, py


def _edge_functions(x: Tensor, y: Tensor, w: Tensor, px: Tensor, py: Tensor) -> Tensor:
    """Unnormalized homogeneous barycentrics of the pixels ``(px, py)``, stacked on the last dim."""
    a0, a1, a2 = (x - px[..., None] * w).unbind(-1)
    c0, c1, c2 = (y - py[..., None] * w).unbind(-1)
    return torch.stack([a1 * c2 - a2 * c1, a2 * c0 - a0 * c2, a0 * c1 - a1 * c0], -1)


def _edge_derivatives(
    x: Tensor, y: Tensor, w: Tensor, px: Tensor, py: Tensor
) -> tuple[Tensor, Tensor]:
    """Derivatives of :func:`_edge_functions` with respect to ``px`` and ``py``."""
    a0, a1, a2 = (x - px[..., None] * w).unbind(-1)
    c0, c1, c2 = (y - py[..., None] * w).unbind(-1)
    w0, w1, w2 = w.unbind(-1)
    de_dx = torch.stack([w2 * c1 - w1 * c2, w0 * c2 - w2 * c0, w1 * c0 - w0 * c1], -1)
    de_dy = torch.stack([a2 * w1 - a1 * w2, a0 * w2 - a2 * w0, a1 * w0 - a0 * w1], -1)
    return de_dx, de_dy


def _visible_triangles(pos: Tensor, tri: Tensor, px: Tensor, py: Tensor) -> Tensor:
    """Index of the nearest covering triangle per pixel, or -1. Ties go to the lower index."""
    batch, height, width = pos.shape[0], py.shape[0], px.shape[0]
    best_depth = pos.new_full((batch, height, width), torch.inf)
    best_id = torch.full((batch, height, width), -1, dtype=torch.long, device=pos.device)
    chunk = max(1, _CHUNK_ELEMENTS // (batch * height * width))
    px, py = px.view(1, 1, 1, width), py.view(1, 1, height, 1)

    for start in range(0, tri.shape[0], chunk):
        verts = pos[:, tri[start : start + chunk]][:, :, None, None]
        x, y, z, w = verts.unbind(-1)
        e = _edge_functions(x, y, w, px, py)
        b = e / e.sum(-1, keepdim=True)
        w_p = (b * w).sum(-1)
        depth = (b * z).sum(-1) / w_p
        inside = (b >= 0).all(-1) & (w_p > 0) & (depth >= -1) & (depth <= 1)
        depth = torch.where(inside, depth, torch.inf)

        chunk_depth, chunk_id = depth.min(dim=1)
        closer = chunk_depth < best_depth
        best_depth = torch.where(closer, chunk_depth, best_depth)
        best_id = torch.where(closer, chunk_id + start, best_id)
    return best_id
