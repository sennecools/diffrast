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
import torch.nn.functional as F
from torch import Tensor

# Upper bound on batch * triangles * pixels evaluated at once during visibility.
_CHUNK_ELEMENTS = 1 << 22


def rasterize(
    pos: Tensor, tri: Tensor, resolution: tuple[int, int], grad_db: bool
) -> tuple[Tensor, Tensor]:
    height, width = resolution
    if tri.shape[0] == 0:
        empty = pos.new_zeros(pos.shape[0], height, width, 4)
        return empty, empty.clone()
    px, py = pixel_centres(height, width, pos)
    with torch.no_grad():
        tri_id = _visible_triangles(pos, tri.long(), px, py)
    return rasterize_from_ids(pos, tri, tri_id, grad_db)


def rasterize_from_ids(
    pos: Tensor, tri: Tensor, tri_id: Tensor, grad_db: bool
) -> tuple[Tensor, Tensor]:
    """Build ``rast`` and ``rast_db`` from the visible triangle per pixel.

    ``tri_id`` has shape ``[batch, height, width]`` and holds the index of the visible triangle,
    or -1 for background. Barycentrics are computed here with autograd enabled, so gradients reach
    the vertices of the visible triangles. Other backends reuse this after their own visibility
    pass.
    """
    batch, height, width = tri_id.shape
    px, py = pixel_centres(height, width, pos)

    # Work on covered pixels only. Besides saving memory, this keeps background pixels from all
    # gathering one placeholder triangle, whose gradient accumulation would then serialize.
    pixel = (tri_id >= 0).nonzero(as_tuple=True)
    visible = tri_id[pixel]
    verts = pos[pixel[0][:, None], tri.long()[visible]]
    x, y, z, w = verts.unbind(-1)
    px, py = px[pixel[2]], py[pixel[1]]

    e = _edge_functions(x, y, w, px, py)
    area = e.sum(-1, keepdim=True)
    b = e / area
    depth = (b * z).sum(-1) / (b * w).sum(-1)

    de_dx, de_dy = _edge_derivatives(x, y, w, px, py)
    db_dx = (de_dx - b * de_dx.sum(-1, keepdim=True)) / area
    db_dy = (de_dy - b * de_dy.sum(-1, keepdim=True)) / area
    if not grad_db:
        db_dx, db_dy = db_dx.detach(), db_dy.detach()

    values = torch.stack([b[..., 0], b[..., 1], depth, (visible + 1).to(pos.dtype)], -1)
    # The pixel grid spans 2 NDC units, so one pixel is 2 / size in NDC.
    derivs = torch.stack(
        [
            db_dx[..., 0] * (2 / width),
            db_dy[..., 0] * (2 / height),
            db_dx[..., 1] * (2 / width),
            db_dy[..., 1] * (2 / height),
        ],
        -1,
    )
    rast = pos.new_zeros(batch, height, width, 4).index_put(pixel, values)
    rast_db = pos.new_zeros(batch, height, width, 4).index_put(pixel, derivs)
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
    pixel = (tri_id >= 0).nonzero(as_tuple=True)
    if attr.ndim == 2:
        attr = attr[None]
    attr = attr.expand(batch, -1, -1)
    a0, a1, a2 = attr[pixel[0][:, None], tri.long()[tri_id[pixel]]].unbind(-2)

    u, v = rast[pixel][:, 0:1], rast[pixel][:, 1:2]
    out = out.index_put(pixel, u * a0 + v * a1 + (1 - u - v) * a2)
    if not diff_attrs:
        return out, None

    da_du = (a0 - a2)[:, list(diff_attrs)]
    da_dv = (a1 - a2)[:, list(diff_attrs)]
    du_dx, du_dy, dv_dx, dv_dy = rast_db[pixel][:, :, None].unbind(1)
    d_dx = du_dx * da_du + dv_dx * da_dv
    d_dy = du_dy * da_du + dv_dy * da_dv
    out_da = out_da.index_put(pixel, torch.stack([d_dx, d_dy], -1).flatten(-2))
    return out, out_da


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None,
    filter_mode: str,
    boundary_mode: str,
    max_mip_level: int | None,
) -> Tensor:
    if filter_mode == "nearest":
        return _sample_nearest(tex, uv, boundary_mode)
    if filter_mode == "linear":
        return _sample_linear(tex, uv, boundary_mode)

    levels = _mip_chain(tex, max_mip_level)
    lod = _level_of_detail(uv_da, tex.shape[1], tex.shape[2]).clamp(0, len(levels) - 1)
    if filter_mode == "linear-mipmap-nearest":
        nearest = torch.floor(lod + 0.5).detach()
        weights = [(nearest == i).to(uv.dtype) for i in range(len(levels))]
        used = [nearest == i for i in range(len(levels))]
    else:
        lower = lod.floor().detach()
        frac = lod - lower
        weights = [
            torch.where(lower == i, 1 - frac, 0) + torch.where(lower + 1 == i, frac, 0)
            for i in range(len(levels))
        ]
        used = [(lower == i) | (lower + 1 == i) for i in range(len(levels))]

    out = uv.new_zeros(*uv.shape[:3], tex.shape[-1])
    for level, weight, mask in zip(levels, weights, used, strict=True):
        # Levels no pixel touches are skipped. A touched level is sampled even where its weight
        # is zero, so the blend factor still receives a gradient.
        if mask.any():
            out = out + weight[..., None] * _sample_linear(level, uv, boundary_mode)
    return out


def antialias(
    color: Tensor, rast: Tensor, pos: Tensor, tri: Tensor, pos_gradient_boost: float
) -> Tensor:
    """Blend colours across silhouette edges by where each edge crosses between pixel centres.

    For every pair of horizontally or vertically adjacent pixels showing different triangles,
    the silhouette edge between them is found and the point ``t`` (in ``[0, 1]``, from the first
    pixel) where it crosses the segment joining their centres. Treating each pixel as a unit box
    along that axis, the pixel whose half contains the crossing takes ``|t - 0.5|`` of the other
    pixel's colour. ``t`` depends on the vertex positions, so moving an edge changes the image
    continuously and gives gradients to ``pos``.
    """
    if tri.shape[0] == 0:
        return color
    height, width = rast.shape[1:3]
    tri = tri.long()
    tri_id = rast[..., 3].detach().round().long() - 1
    depth = rast[..., 2].detach()
    with torch.no_grad():
        silhouette = _silhouette_edges(pos.detach(), tri)
    px, py = pixel_centres(height, width, pos)

    out = color
    for d_row, d_col in ((0, 1), (1, 0)):
        id_p = tri_id[:, : height - d_row, : width - d_col]
        id_q = tri_id[:, d_row:, d_col:]
        b, row, col = (id_p != id_q).nonzero(as_tuple=True)
        row_q, col_q = row + d_row, col + d_col
        horizontal = d_col == 1

        args = (pos, tri, silhouette, b, horizontal, width, height)
        from_p, t_p = _edge_crossing(
            *args, id_p[b, row, col], px[col], py[row], px[col_q], py[row_q]
        )
        from_q, t_q = _edge_crossing(
            *args, id_q[b, row, col], px[col_q], py[row_q], px[col], py[row]
        )

        # When both pixels' triangles have an edge between them, the nearer one is in front.
        use_p = from_p & (~from_q | (depth[b, row, col] <= depth[b, row_q, col_q]))
        use_q = from_q & ~use_p
        t = torch.where(use_p, t_p, 1 - t_q)
        t = t.detach() + pos_gradient_boost * (t - t.detach())
        active = (use_p | use_q)[:, None]

        c_p, c_q = color[b, row, col], color[b, row_q, col_q]
        delta_p = torch.where(active & (t < 0.5)[:, None], (0.5 - t)[:, None] * (c_q - c_p), 0.0)
        delta_q = torch.where(active & (t >= 0.5)[:, None], (t - 0.5)[:, None] * (c_p - c_q), 0.0)
        out = out.index_put((b, row, col), delta_p, accumulate=True)
        out = out.index_put((b, row_q, col_q), delta_q, accumulate=True)
    return out


def pixel_centres(height: int, width: int, like: Tensor) -> tuple[Tensor, Tensor]:
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


def _mip_chain(tex: Tensor, max_mip_level: int | None) -> list[Tensor]:
    """Box-filtered mip levels, halving each dimension (rounding up) until both are 1."""
    levels = [tex]
    while max(levels[-1].shape[1:3]) > 1 and (
        max_mip_level is None or len(levels) <= max_mip_level
    ):
        height, width = levels[-1].shape[1:3]
        kernel = (2 if height > 1 else 1, 2 if width > 1 else 1)
        # With ceil_mode, a window hanging over an odd edge averages only the texels it covers.
        level = F.avg_pool2d(levels[-1].permute(0, 3, 1, 2), kernel, ceil_mode=True)
        levels.append(level.permute(0, 2, 3, 1))
    return levels


def _level_of_detail(uv_da: Tensor, height: int, width: int) -> Tensor:
    """Mip level from the longer of the two screen-space footprint axes, in texels."""
    du_dx, du_dy, dv_dx, dv_dy = uv_da.unbind(-1)
    len_x = (du_dx * width) ** 2 + (dv_dx * height) ** 2
    len_y = (du_dy * width) ** 2 + (dv_dy * height) ** 2
    return 0.5 * torch.log2(torch.maximum(len_x, len_y).clamp_min(1e-30))


def _sample_nearest(tex: Tensor, uv: Tensor, boundary_mode: str) -> Tensor:
    height, width = tex.shape[1:3]
    col = (uv[..., 0] * width).floor().long()
    row = (uv[..., 1] * height).floor().long()
    return _fetch(tex, row, col, boundary_mode)


def _sample_linear(tex: Tensor, uv: Tensor, boundary_mode: str) -> Tensor:
    height, width = tex.shape[1:3]
    x = uv[..., 0] * width - 0.5
    y = uv[..., 1] * height - 0.5
    x0, y0 = x.floor().detach(), y.floor().detach()
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    col, row = x0.long(), y0.long()
    top = (1 - fx) * _fetch(tex, row, col, boundary_mode) + fx * _fetch(
        tex, row, col + 1, boundary_mode
    )
    bottom = (1 - fx) * _fetch(tex, row + 1, col, boundary_mode) + fx * _fetch(
        tex, row + 1, col + 1, boundary_mode
    )
    return (1 - fy) * top + fy * bottom


def _fetch(tex: Tensor, row: Tensor, col: Tensor, boundary_mode: str) -> Tensor:
    """Texels at integer coordinates ``[batch, height, width]``, with out-of-range handling."""
    tex_batch, height, width, channels = tex.shape
    valid = None
    if boundary_mode == "wrap":
        row, col = row % height, col % width
    else:
        if boundary_mode == "zero":
            valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        row, col = row.clamp(0, height - 1), col.clamp(0, width - 1)

    image = torch.arange(row.shape[0], device=row.device)[:, None, None] if tex_batch > 1 else 0
    texels = tex.reshape(-1, channels)[(image * height + row) * width + col]
    return texels if valid is None else texels * valid[..., None]


def _edge_crossing(
    pos: Tensor,
    tri: Tensor,
    silhouette: Tensor,
    b: Tensor,
    horizontal: bool,
    width: int,
    height: int,
    tri_id: Tensor,
    src_x: Tensor,
    src_y: Tensor,
    dst_x: Tensor,
    dst_y: Tensor,
) -> tuple[Tensor, Tensor]:
    """Where the triangle visible at ``src`` is exited along the segment to ``dst``.

    Returns ``(found, t)``. ``found`` is false when there is no triangle at ``src``, the triangle
    also covers ``dst``, the edge it exits through is not a silhouette, or that edge is closer to
    the other axis (it is handled by the pixel pairs along that axis). ``t`` is the crossing as a
    fraction of the segment from ``src``.
    """
    visible = tri_id >= 0
    tri_id = tri_id.clamp(min=0)
    x, y, _, w = pos[b[:, None], tri[tri_id]].unbind(-1)

    # The edge functions are affine in screen space, so each edge's crossing is found by
    # linear interpolation between the two pixel centres.
    e_src = _edge_functions(x, y, w, src_x, src_y)
    e_dst = _edge_functions(x, y, w, dst_x, dst_y)
    sign = torch.where(e_src.sum(-1, keepdim=True) < 0, -1.0, 1.0)
    outside = e_dst * sign < 0
    t_edge = torch.where(outside, e_src / torch.where(outside, e_src - e_dst, 1.0), torch.inf)
    t, edge = t_edge.min(-1)

    de_dx, de_dy = _edge_derivatives(x, y, w, src_x, src_y)
    normal_x = de_dx.gather(-1, edge[:, None]).squeeze(-1).abs() * (2 / width)
    normal_y = de_dy.gather(-1, edge[:, None]).squeeze(-1).abs() * (2 / height)
    along_axis = normal_x >= normal_y if horizontal else normal_y > normal_x

    found = visible & outside.any(-1) & silhouette[b, tri_id, edge] & along_axis
    return found.detach(), torch.where(found, t, 0.0)


def _silhouette_edges(pos: Tensor, tri: Tensor) -> Tensor:
    """Which triangle edges are silhouettes, ``[batch, triangles, 3]``.

    Edge ``k`` of a triangle is the one opposite vertex ``k``. It is a silhouette when no other
    triangle shares it, when more than two do, or when the triangle sharing it projects onto the
    same side of it (the surface folds over at that edge).
    """
    num_tri = tri.shape[0]
    start, end = tri[:, [1, 2, 0]].flatten(), tri[:, [2, 0, 1]].flatten()
    key = torch.minimum(start, end) * (int(tri.max()) + 1) + torch.maximum(start, end)
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    first = (counts.cumsum(0) - counts)[(counts == 2).nonzero().squeeze(1)]
    neighbour = torch.full_like(key, -1)
    neighbour[order[first]] = order[first + 1]
    neighbour[order[first + 1]] = order[first]

    # The side of the edge's projected line a point lies on is the sign of det[a, b, p] / w_p in
    # homogeneous (x, y, w) coordinates; the factor 1 / (w_a * w_b) is shared and drops out.
    xyw = pos[..., [0, 1, 3]]
    line = torch.linalg.cross(xyw[:, start], xyw[:, end], dim=-1)

    def side(vertex: Tensor) -> Tensor:
        p = xyw[:, vertex]
        return torch.sign((line * p).sum(-1) * p[..., 2])

    opposite = tri.flatten()
    shared = neighbour >= 0
    fold = side(opposite) * side(opposite[neighbour.clamp(min=0)]) > 0
    return (~shared | fold).view(-1, num_tri, 3)
