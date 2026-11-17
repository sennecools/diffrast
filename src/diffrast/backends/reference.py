"""Pure-PyTorch implementation of the ops.

Slow, but runs on any device and dtype PyTorch supports. It is the ground truth the Triton backend
is tested against. Gradients come from autograd.

Rasterization is done in homogeneous coordinates (Olano and Greer, 1997): for a pixel at
normalized device coordinates ``(px, py)``, the perspective-correct barycentrics ``b`` of a
triangle satisfy ``sum(b * (x - px * w)) = 0`` and ``sum(b * (y - py * w)) = 0``, so ``b`` is the
normalized cross product of those two vectors. A pixel is covered when all ``b >= 0``, the
interpolated ``w`` is positive, and ``-1 <= z / w <= 1``. This needs no near-plane clipping. A
pixel centre exactly on an edge belongs to one side of it only (see :func:`_inside_edge`), so
triangles sharing the edge never both cover it and never leave it empty.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

# Upper bound on batch * triangles * pixels evaluated at once during visibility.
_CHUNK_ELEMENTS = 1 << 22
# A triangle is not drawn at a pixel where its area is within this many epsilons of the magnitude
# of the terms it is computed from. The Triton visibility kernels use the same constant.
AREA_TOLERANCE = 2

# Antialiasing gives an edge to the horizontal pixel pairs when it is at least as steep as 45
# degrees in pixel space. Edges within this relative margin of 45 degrees count as steep too, so
# that the rounding of the slope cannot decide where an edge at exactly 45 degrees goes.
AXIS_TIE = 1 + 1e-4
# Antialiasing counts a pixel centre as lying on an edge when the edge function there is zero
# up to its rounding error, taken as this many epsilons of the terms it is computed from.
EDGE_TIE = 4


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
    previous: tuple[Tensor, Tensor] | None,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
    """Rasterize the nearest layer behind ``previous``, the state returned for the layer before.

    The state is the visibility depth and triangle index per pixel. Layers are ordered by depth
    and then by triangle index, so triangles at equal depth each get their own layer.
    """
    height, width = resolution
    if tri.shape[0] == 0 or pos.shape[0] == 0:
        empty = pos.new_zeros(pos.shape[0], height, width, 4)
        none = torch.full((pos.shape[0], height, width), -1, device=pos.device)
        return empty, empty.clone(), (pos.new_full(none.shape, torch.inf), none)
    px, py = pixel_centres(height, width, pos)
    with torch.no_grad():
        depth, tri_id = _visible_triangles(pos, tri.long(), px, py, previous)
    rast, rast_db = rasterize_from_ids(pos, tri, tri_id, grad_db)
    return rast, rast_db, (depth, tri_id)


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
) -> tuple[Tensor, Tensor]:
    batch, height, width, _ = rast.shape
    channels = attr.shape[-1]
    out = attr.new_zeros(batch, height, width, channels)
    out_da = attr.new_zeros(batch, height, width, 2 * len(diff_attrs))
    if tri.shape[0] == 0:
        return out, out_da

    # Ids past the end of tri are drawn as background rather than read out of range.
    tri_id = rast[..., 3].round().long() - 1
    pixel = ((tri_id >= 0) & (tri_id < tri.shape[0])).nonzero(as_tuple=True)
    if attr.ndim == 2:
        attr = attr[None]
    attr = attr.expand(batch, -1, -1)
    a0, a1, a2 = attr[pixel[0][:, None], tri.long()[tri_id[pixel]]].unbind(-2)

    u, v = rast[pixel][:, 0:1], rast[pixel][:, 1:2]
    out = out.index_put(pixel, u * a0 + v * a1 + (1 - u - v) * a2)
    if not diff_attrs:
        return out, out_da

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
    mip_level_bias: Tensor | None = None,
    mip: Sequence[Tensor] | None = None,
    packed: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> Tensor:
    """Sample ``tex``; ``mip`` holds mip levels 1 and up to use instead of a box-filtered chain.

    ``packed`` is the result of :func:`_pack` for ``tex`` followed by all of ``mip``, for a
    caller that has it already.

    A cube map is reduced to a 2D lookup per face: its directions are projected to a face index
    and face coordinates, and only fetching texels across face edges differs from there on.
    """
    mipmapped = "mipmap" in filter_mode
    face = valid = None
    if boundary_mode == "cube":
        from diffrast.backends import _cube

        face, uv, uv_da, valid = _cube.project(uv, uv_da if mipmapped else None)
    image = torch.arange(uv.shape[0], device=uv.device)[:, None, None]

    if mipmapped:
        texels, heights, widths, offsets = _pack_levels(tex, mip, max_mip_level, packed)
        lod = uv.new_zeros(uv.shape[:3])
        if uv_da is not None:
            lod = lod + _level_of_detail(uv_da, tex.shape[-3], tex.shape[-2])
        if mip_level_bias is not None:
            lod = lod + mip_level_bias
        blend = filter_mode == "linear-mipmap-linear"
        sizes = (heights, widths, offsets)
        out, bad = _sample_mipmapped(texels, sizes, uv, lod, blend, boundary_mode, image, face)
    else:
        texels, heights, widths, offsets = _pack([tex])
        sample = _sample_nearest if filter_mode == "nearest" else _sample_linear
        size = (heights[0], widths[0], offsets[0])
        out, bad = sample(texels, size, uv, boundary_mode, image, face)
    # A pixel whose coordinates or level of detail are not finite is NaN and passes no gradient.
    out = torch.where(bad[..., None], torch.nan, out)
    # Directions that select no point on the cube read zeros and pass no gradient.
    return out if valid is None else torch.where(valid[..., None], out, 0)


def texture_construct_mip(tex: Tensor, max_mip_level: int | None) -> list[Tensor]:
    """Mip levels 1 and up of ``tex``, in the form :func:`texture` takes as ``mip``."""
    return _mip_chain(tex, max_mip_level)[1:]


def antialias(
    color: Tensor,
    rast: Tensor,
    pos: Tensor,
    tri: Tensor,
    pos_gradient_boost: float,
    neighbours: Tensor | None = None,
) -> Tensor:
    """Blend colours across silhouette edges by where each edge crosses between pixel centres.

    For every pair of horizontally or vertically adjacent pixels showing different triangles,
    the silhouette edge between them is found and the point ``t`` (in ``[0, 1]``, from the first
    pixel) where it crosses the segment joining their centres. Treating each pixel as a unit box
    along that axis, the pixel whose half contains the crossing takes ``|t - 0.5|`` of the other
    pixel's colour. ``t`` depends on the vertex positions, so moving an edge changes the image
    continuously and gives gradients to ``pos``.

    ``pos`` is ``[batch, vertices, 4]``, or ``[vertices, 4]`` to share it across the batch.
    ``neighbours`` is the result of :func:`edge_neighbours` for ``tri``, if already known.
    """
    if tri.shape[0] == 0:
        return color.clone()
    batch, height, width = rast.shape[:3]
    tri = tri.long()
    if pos.ndim == 2:
        pos = pos[None]
    tri_id = rast[..., 3].detach().round().long() - 1
    # Ids past the end of tri are treated as background rather than read out of range.
    tri_id = torch.where(tri_id < tri.shape[0], tri_id, -1)
    depth = rast[..., 2].detach()
    with torch.no_grad():
        silhouette = _silhouette_edges(pos, tri, neighbours).expand(batch, -1, -1)
    pos = pos.expand(batch, -1, -1)
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
    return out.contiguous()


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


def _coverage(verts: Tensor, px: Tensor, py: Tensor) -> tuple[Tensor, Tensor]:
    """Which pixels each triangle covers, and its depth ``z / w`` at every pixel.

    ``verts`` holds clip-space triangles, ``[batch, triangles, 3, 1, 1, 4]``; ``px`` and ``py``
    broadcast against ``[batch, triangles, height, width]``, the shape of both results. The Triton
    visibility kernels evaluate the same operations in the same order, so that in float32 the two
    backends decide every pixel alike.
    """
    x, y, z, w = verts.unbind(-1)
    (x0, x1, x2), (y0, y1, y2), (z0, z1, z2), (w0, w1, w2) = (t.unbind(2) for t in (x, y, z, w))
    a0, a1, a2 = x0 - px * w0, x1 - px * w1, x2 - px * w2
    c0, c1, c2 = y0 - py * w0, y1 - py * w1, y2 - py * w2
    e0 = a1 * c2 - a2 * c1
    e1 = a2 * c0 - a0 * c2
    e2 = a0 * c1 - a1 * c0
    s = e0 + e1 + e2
    ew = e0 * w0 + e1 * w1 + e2 * w2
    depth = (e0 * z0 + e1 * z1 + e2 * z2) / ew

    positive = s > 0
    inside = _inside_edge(e0, a1, c1, w1, a2, c2, w2, positive)
    inside &= _inside_edge(e1, a2, c2, w2, a0, c0, w0, positive)
    inside &= _inside_edge(e2, a0, c0, w0, a1, c1, w1, positive)

    # Rounding of the vertex coordinates and of a and c can change s by about this much, so a
    # smaller area is indistinguishable from zero. Without this test, a zero-area triangle is
    # drawn wherever the rounding residue of its edge functions happens to agree in sign. A
    # pixel centre has |px| < 1, so |x| + |w| bounds the terms of a = x - px * w.
    reach_x = torch.maximum(
        torch.maximum(x0.abs() + w0.abs(), x1.abs() + w1.abs()), x2.abs() + w2.abs()
    )
    reach_y = torch.maximum(
        torch.maximum(y0.abs() + w0.abs(), y1.abs() + w1.abs()), y2.abs() + w2.abs()
    )
    span_x = a0.abs() + a1.abs() + a2.abs()
    span_y = c0.abs() + c1.abs() + c2.abs()
    tolerance = AREA_TOLERANCE * torch.finfo(verts.dtype).eps
    inside &= s.abs() > (reach_x * span_y + span_x * (reach_y + span_y)) * tolerance

    inside &= torch.where(positive, ew > 0, ew < 0)
    inside &= (depth >= -1) & (depth <= 1)
    inside &= verts.isfinite().all(-1).all(2)
    return inside, depth


def _inside_edge(
    e: Tensor,
    ai: Tensor,
    ci: Tensor,
    wi: Tensor,
    aj: Tensor,
    cj: Tensor,
    wj: Tensor,
    positive: Tensor,
) -> Tensor:
    """Whether a pixel is on the inner side of the edge from vertex i to vertex j.

    ``e = a_i * c_j - a_j * c_i`` is the edge function at the pixel, and the inner side is where
    it has the sign of the triangle's area, given by ``positive``. A pixel centre exactly on the
    edge is inside only when the interior lies in the +x direction of screen space, or in the +y
    direction for a horizontal edge, like the top-left rule of hardware rasterizers.

    The direction is the sign of the screen-space gradient ``(gx, gy)`` of ``e``. It is computed
    from the coordinates relative to the pixel, which are small and often exact where ``e`` is
    zero: at a pixel centre on a vertex, the gradients of the edges through that vertex are single
    products, so the triangles around it agree on which of them owns the pixel. ``e`` and the
    gradient change sign exactly when i and j are swapped, so of two triangles on opposite sides
    of an edge exactly one owns the pixels on it, whatever their winding.
    """
    gx = ci * wj - cj * wi
    gy = aj * wi - ai * wj
    q = torch.where(e == 0, torch.where(gx == 0, gy, gx), e)
    return torch.where(positive, q > 0, q < 0)


def _visible_triangles(
    pos: Tensor, tri: Tensor, px: Tensor, py: Tensor, previous: tuple[Tensor, Tensor] | None = None
) -> tuple[Tensor, Tensor]:
    """Depth and index of the nearest covering triangle per pixel; index -1 where there is none.

    Ties go to the lower index. With ``previous`` (depth and index of an earlier layer), only
    triangles ordered after it by ``(depth, index)`` are considered.
    """
    batch, height, width = pos.shape[0], py.shape[0], px.shape[0]
    best_depth = pos.new_full((batch, height, width), torch.inf)
    best_id = torch.full((batch, height, width), -1, dtype=torch.long, device=pos.device)
    chunk = max(1, _CHUNK_ELEMENTS // (batch * height * width))
    px, py = px.view(1, 1, 1, width), py.view(1, 1, height, 1)
    if previous is not None:
        prev_depth, prev_id = (t[:, None] for t in previous)

    for start in range(0, tri.shape[0], chunk):
        verts = pos[:, tri[start : start + chunk]][:, :, :, None, None]
        inside, depth = _coverage(verts, px, py)
        if previous is not None:
            index = torch.arange(start, start + verts.shape[1], device=pos.device)[:, None, None]
            later = (depth > prev_depth) | ((depth == prev_depth) & (index > prev_id))
            inside = inside & later
        depth = torch.where(inside, depth, torch.inf)

        chunk_depth, chunk_id = depth.min(dim=1)
        closer = chunk_depth < best_depth
        best_depth = torch.where(closer, chunk_depth, best_depth)
        best_id = torch.where(closer, chunk_id + start, best_id)
    return best_depth, best_id


# Footprints saturate at this many texels. Both backends use the same value, so that they agree
# on inputs that are out of range.
_FOOTPRINT_LIMIT = 1e18


def _mip_levels(
    tex: Tensor, mip: Sequence[Tensor] | None, max_mip_level: int | None
) -> list[Tensor]:
    """The levels to sample, base level first: the given ones, or a chain built from ``tex``."""
    if mip is None:
        return _mip_chain(tex, max_mip_level)
    levels = [tex, *mip]
    return levels if max_mip_level is None else levels[: max_mip_level + 1]


def _pack_levels(
    tex: Tensor,
    mip: Sequence[Tensor] | None,
    max_mip_level: int | None,
    packed: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The buffer and size tables, as from :func:`_pack`, of the levels a lookup may use.

    With ``packed`` given, nothing is copied: the buffer keeps every level and ``max_mip_level``
    only shortens the tables.
    """
    if packed is None:
        return _pack(_mip_levels(tex, mip, max_mip_level))
    texels, *tables = packed
    count = None if max_mip_level is None else max_mip_level + 1
    return (texels, *(table[:count] for table in tables))


def _mip_chain(tex: Tensor, max_mip_level: int | None) -> list[Tensor]:
    """Box-filtered mip levels, halving each dimension (rounding up) until both are 1.

    The faces of a cube map are filtered one by one.
    """
    cube = tex.ndim == 5
    levels = [tex.flatten(0, 1) if cube else tex]
    while max(levels[-1].shape[1:3]) > 1 and (
        max_mip_level is None or len(levels) <= max_mip_level
    ):
        levels.append(_downsample(levels[-1]))
    if cube:
        levels = [tex, *(level.unflatten(0, tex.shape[:2]) for level in levels[1:])]
    return levels


def _downsample(level: Tensor) -> Tensor:
    """The next mip level of a ``[batch, height, width, channels]`` level.

    Each dimension is split into half as many equal bins (rounding up), and a coarse texel is the
    area-weighted mean of the texels its bin overlaps. Every texel therefore has the same total
    weight in the result, and the levels stay aligned in ``uv``. For an even size the bins are
    whole pairs of texels.
    """
    height, width = level.shape[1:3]
    image = level.permute(0, 3, 1, 2)
    pairs = (2 if height % 2 == 0 else 1, 2 if width % 2 == 0 else 1)
    if pairs != (1, 1):
        image = F.avg_pool2d(image, pairs)
    if height > 1 and height % 2 == 1:
        image = _halve_odd(image, 2)
    if width > 1 and width % 2 == 1:
        image = _halve_odd(image, 3)
    return image.permute(0, 2, 3, 1)


def _halve_odd(image: Tensor, dim: int) -> Tensor:
    """Resample an odd dimension of ``n`` texels into ``(n + 1) / 2`` equal bins.

    A bin is less than two texels wide, so it overlaps at most three texels.
    """
    n = image.shape[dim]
    m = (n + 1) // 2
    j = torch.arange(m, device=image.device)
    first = j * n // m
    shape = [1] * image.ndim
    shape[dim] = m
    out = 0
    for tap in range(3):
        i = first + tap
        # Scaled by m, bin j spans [j * n, (j + 1) * n] and texel i spans [i * m, (i + 1) * m],
        # so the overlaps are exact integers.
        overlap = torch.minimum((j + 1) * n, (i + 1) * m) - torch.maximum(j * n, i * m)
        weight = (overlap.clamp(min=0).to(image.dtype) / n).view(shape)
        out = out + weight * image.index_select(dim, i.clamp(max=n - 1))
    return out


def _level_of_detail(uv_da: Tensor, height: int, width: int) -> Tensor:
    """Mip level from the longer of the two screen-space footprint axes, in texels.

    The footprint saturates far beyond any texture size, so that huge or infinite derivatives
    select the coarsest level with a zero gradient instead of overflowing.
    """
    limit = _FOOTPRINT_LIMIT
    du_dx, du_dy, dv_dx, dv_dy = uv_da.unbind(-1)
    len_x = (du_dx * width).clamp(-limit, limit) ** 2 + (dv_dx * height).clamp(-limit, limit) ** 2
    len_y = (du_dy * width).clamp(-limit, limit) ** 2 + (dv_dy * height).clamp(-limit, limit) ** 2
    return 0.5 * torch.log2(torch.maximum(len_x, len_y).clamp_min(1e-30))


def _pack(levels: Sequence[Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Join mip levels into one buffer of shape ``[tex_batch, texels, channels]``.

    Also returns the height, width and texel offset of each level as int64 tensors, so that a
    lookup can address a different level per pixel. A level shared across the batch is repeated
    when the base level is not.
    """
    batch, channels = levels[0].shape[0], levels[0].shape[-1]
    sizes = [[*level.shape[-3:-1], level.shape[1:-1].numel()] for level in levels]
    parts = [
        level.expand(batch, *level.shape[1:]).reshape(batch, size[2], channels)
        for level, size in zip(levels, sizes, strict=True)
    ]
    texels = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
    heights, widths, counts = torch.tensor(sizes, device=texels.device).unbind(1)
    return texels, heights, widths, counts.cumsum(0) - counts


def _sample_mipmapped(
    texels: Tensor,
    sizes: tuple[Tensor, Tensor, Tensor],
    uv: Tensor,
    lod: Tensor,
    blend: bool,
    boundary_mode: str,
    image: Tensor,
    face: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Bilinear lookups in the level nearest to ``lod``, or blended between the two around it.

    ``sizes`` holds the height, width and offset of every level in ``texels``. Returns the
    values and which pixels are unusable, which includes those whose level of detail is NaN.
    """
    top = sizes[0].shape[0] - 1
    bad = lod.isnan()
    lod = torch.where(bad, 0, lod).clamp(0, top)
    lower = (lod.detach() + (0.0 if blend else 0.5)).floor().long()
    args = (uv, boundary_mode, image, face)
    out, bad_lower = _sample_linear(texels, [t[lower] for t in sizes], *args)
    bad = bad | bad_lower
    if blend:
        # At the top level both lookups read that level, so the level of detail has no effect
        # and no gradient there.
        upper = (lower + 1).clamp(max=top)
        frac = (lod - lower)[..., None]
        above, bad_upper = _sample_linear(texels, [t[upper] for t in sizes], *args)
        out = out * (1 - frac) + above * frac
        bad = bad | bad_upper
    return out, bad


def _sample_nearest(
    texels: Tensor,
    size: Sequence[Tensor],
    uv: Tensor,
    boundary_mode: str,
    image: Tensor,
    face: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The texel containing ``uv``, and which pixels have coordinates that are not finite.

    ``size`` is the height, width and offset of the level to sample.
    """
    height, width, _ = size
    x, bad_x = _texel_coordinate(uv[..., 0], width, 0.0, boundary_mode)
    y, bad_y = _texel_coordinate(uv[..., 1], height, 0.0, boundary_mode)
    col, row = x.detach().floor().long(), y.detach().floor().long()
    # A direction exactly on the far edge of a cube face stays on that face.
    boundary_mode = "clamp" if boundary_mode == "cube" else boundary_mode
    return _fetch(texels, size, image, row, col, boundary_mode, face), bad_x | bad_y


def _sample_linear(
    texels: Tensor,
    size: Sequence[Tensor],
    uv: Tensor,
    boundary_mode: str,
    image: Tensor,
    face: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Blend of the four texels around ``uv``, and which pixels have non-finite coordinates.

    ``size`` is the height, width and offset of the level to sample, per pixel or shared.
    """
    height, width, _ = size
    x, bad_x = _texel_coordinate(uv[..., 0], width, 0.5, boundary_mode)
    y, bad_y = _texel_coordinate(uv[..., 1], height, 0.5, boundary_mode)
    x0, y0 = x.floor().detach(), y.floor().detach()
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    col, row = x0.long(), y0.long()

    # The four taps are fetched in one lookup, along a new dimension before the channels.
    rows = torch.stack([row, row, row + 1, row + 1], -1)
    cols = torch.stack([col, col + 1, col, col + 1], -1)
    per_tap = [t[..., None] for t in size]
    tap_face = None if face is None else face[..., None]
    taps = _fetch(texels, per_tap, image[..., None], rows, cols, boundary_mode, tap_face)
    t00, t01, t10, t11 = taps.unbind(-2)
    out = (1 - fy) * ((1 - fx) * t00 + fx * t01) + fy * ((1 - fx) * t10 + fx * t11)
    if boundary_mode == "cube":
        # Beyond a corner of the cube one of the four taps has no texel and was read as zero.
        # It counts as the mean of the other three, which keeps the result continuous there.
        left, right = (col < 0)[..., None], (col + 1 >= width)[..., None]
        above, below = (row < 0)[..., None], (row + 1 >= height)[..., None]
        missing = (1 - fy) * ((1 - fx) * (left & above) + fx * (right & above)) + fy * (
            (1 - fx) * (left & below) + fx * (right & below)
        )
        out = out + missing * taps.sum(-2) / 3
    return out, bad_x | bad_y


def _texel_coordinate(
    coord: Tensor, size: Tensor, shift: float, boundary_mode: str
) -> tuple[Tensor, Tensor]:
    """Continuous texel coordinate ``coord * size - shift``, kept within reach of the texture.

    Also returns where it is not finite; there it is replaced and passes no gradient.

    When wrapping, ``coord`` is first reduced to its fractional part, so a coordinate far outside
    the texture keeps the precision it has and cannot overflow an integer index. With clamping or
    zeros outside, the result saturates one texel outside the texture, where every tap is out of
    range already, so far-out and infinite coordinates behave like any other coordinate outside.
    """
    if boundary_mode == "wrap":
        coord = coord - coord.detach().floor()
    x = coord * size - shift
    if boundary_mode in ("clamp", "zero"):
        x = x.clamp(min=-1).clamp(max=size)
    bad = ~torch.isfinite(x)
    return torch.where(bad, 0, x), bad


def _fetch(
    texels: Tensor,
    size: Sequence[Tensor],
    image: Tensor,
    row: Tensor,
    col: Tensor,
    boundary_mode: str,
    face: Tensor | None = None,
) -> Tensor:
    """Texels at integer coordinates, with out-of-range handling.

    ``texels`` has shape ``[tex_batch, texels, channels]`` and ``size`` is the height, width and
    offset of the level each coordinate refers to. ``image`` is the batch index of each
    coordinate, used unless the texture is shared, and ``face`` the cube face it is on. Texels
    that do not exist are read as zero.
    """
    height, width, offset = size
    valid = None
    if boundary_mode == "cube":
        from diffrast.backends import _cube

        face, row, col, valid = _cube.fold(face, row, col, width)
    elif boundary_mode == "wrap":
        row, col = row % height, col % width
    else:
        if boundary_mode == "zero":
            valid = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        row = row.clamp(min=0).clamp(max=height - 1)
        col = col.clamp(min=0).clamp(max=width - 1)

    if face is not None:
        row = row + face * height
    index = offset + row * width + col
    if texels.shape[0] > 1:
        index = index + image * texels.shape[1]
    out = texels.reshape(-1, texels.shape[-1])[index]
    return out if valid is None else out * valid[..., None]


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
    # Only pairs with a triangle at src are evaluated. A stand-in triangle for the others would
    # send its discarded results through the backward pass, where they can be NaN.
    num_pairs = tri_id.shape[0]
    covered = (tri_id >= 0).nonzero().squeeze(1)
    b, tri_id = b[covered], tri_id[covered]
    src_x, src_y, dst_x, dst_y = src_x[covered], src_y[covered], dst_x[covered], dst_y[covered]
    x, y, _, w = pos[b[:, None], tri[tri_id]].unbind(-1)

    # The edge functions are affine in screen space, so each edge's crossing is found by
    # linear interpolation between the two pixel centres.
    e_src = _edge_functions(x, y, w, src_x, src_y)
    e_dst = _edge_functions(x, y, w, dst_x, dst_y)
    sign = torch.where(e_src.sum(-1, keepdim=True) < 0, -1.0, 1.0)
    span = e_src - e_dst
    with torch.no_grad():
        # rast says that the triangle at src is not shown at dst, so a dst centre that lies on
        # an edge up to rounding counts as beyond it. The crossing is then at dst, whose pixel
        # blends half of its neighbour whichever side of the edge the rasterizer gave it to and
        # however it rounded. An edge through both centres is not crossed between them, and
        # dividing by its zero span would give a NaN gradient even though the quotient is
        # discarded.
        slack = _edge_slack(x, y, w, dst_x, dst_y)
        crossed = (e_dst * sign <= slack) & (span * sign > 2 * slack)
    t_edge = torch.where(crossed, e_src / torch.where(crossed, span, 1.0), torch.inf)
    t, edge = t_edge.min(-1)

    with torch.no_grad():
        along_axis = _along_axis(x, y, w, edge, horizontal, width, height)
        found = crossed.any(-1) & silhouette[b, tri_id, edge] & along_axis
    all_found = found.new_zeros(num_pairs).index_put((covered,), found)
    all_t = t.new_zeros(num_pairs).index_put((covered,), torch.where(found, t, 0.0))
    return all_found, all_t


def _edge_slack(x: Tensor, y: Tensor, w: Tensor, px: Tensor, py: Tensor) -> Tensor:
    """Bound on the rounding error of :func:`_edge_functions` at the pixels ``(px, py)``.

    Two evaluations of an edge function that order or fuse the operations differently, such as
    the rasterizer's and the one made here, differ by less than this.
    """
    a, c = x - px[..., None] * w, y - py[..., None] * w
    # Rounding of a = x - px * w is relative to |x| + |w|, since |px| < 1, and likewise for c.
    reach_x, reach_y = x.abs() + w.abs(), y.abs() + w.abs()
    i, j = [1, 2, 0], [2, 0, 1]
    terms = (
        (reach_x[:, i] + a[:, i].abs()) * c[:, j].abs()
        + (reach_x[:, j] + a[:, j].abs()) * c[:, i].abs()
        + reach_y[:, j] * a[:, i].abs()
        + reach_y[:, i] * a[:, j].abs()
    )
    return terms * (EDGE_TIE * torch.finfo(x.dtype).eps)


def _along_axis(
    x: Tensor, y: Tensor, w: Tensor, edge: Tensor, horizontal: bool, width: int, height: int
) -> Tensor:
    """Whether the pixel pairs along one axis handle edge ``edge`` of each triangle.

    Horizontal pairs handle an edge that is at least as steep as 45 degrees in pixel space, and
    vertical pairs the others. The slope is taken from the two vertices of the edge alone, so
    every pixel pair along an edge decides the same way.
    """
    i, j = ((edge + 1) % 3)[:, None], ((edge + 2) % 3)[:, None]
    xi, yi, wi = (v.gather(-1, i).squeeze(-1) for v in (x, y, w))
    xj, yj, wj = (v.gather(-1, j).squeeze(-1) for v in (x, y, w))
    # Derivatives of the edge function along x and y; multiplying by the other image size
    # compares them per pixel.
    steep = (wj * yi - wi * yj).abs() * height * AXIS_TIE
    flat = (xj * wi - xi * wj).abs() * width
    return steep >= flat if horizontal else flat > steep


def edge_neighbours(tri: Tensor) -> Tensor:
    """For each triangle edge, the edge it is shared with, int64 of shape ``[triangles, 3]``.

    Edge ``k`` of a triangle is the one opposite vertex ``k``, and edges are numbered
    ``3 * triangle + k``. The entry is the number of the same edge in the other triangle that
    uses it, or -1 when no other triangle does or more than one does. This depends on ``tri``
    only, so it can be reused for as long as the index buffer stays the same.
    """
    tri = tri.long()
    start, end = tri[:, [1, 2, 0]].flatten(), tri[:, [2, 0, 1]].flatten()
    # Both vertex indices fit in 32 bits, so the pair packs into one key without a multiplier
    # that would have to be read back from the device.
    key = (torch.minimum(start, end) << 32) | torch.maximum(start, end)
    _, inverse, counts = torch.unique(key, return_inverse=True, return_counts=True)
    order = torch.argsort(inverse, stable=True)
    first = (counts.cumsum(0) - counts)[(counts == 2).nonzero().squeeze(1)]
    neighbour = torch.full_like(key, -1)
    neighbour[order[first]] = order[first + 1]
    neighbour[order[first + 1]] = order[first]
    return neighbour.view(-1, 3)


def _silhouette_edges(pos: Tensor, tri: Tensor, neighbours: Tensor | None = None) -> Tensor:
    """Which triangle edges are silhouettes, ``[batch, triangles, 3]``.

    Edge ``k`` of a triangle is the one opposite vertex ``k``. It is a silhouette when no other
    triangle shares it, when more than two do, or when the triangle sharing it projects onto the
    same side of it (the surface folds over at that edge). ``neighbours`` is the result of
    :func:`edge_neighbours` for ``tri``; it is computed here when not given.
    """
    num_tri = tri.shape[0]
    neighbour = (edge_neighbours(tri) if neighbours is None else neighbours).flatten()
    start, end = tri[:, [1, 2, 0]].flatten(), tri[:, [2, 0, 1]].flatten()

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
