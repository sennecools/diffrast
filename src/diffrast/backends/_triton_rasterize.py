"""Visibility pass of the Triton rasterizer.

Triangles are set up and binned into screen tiles with PyTorch ops. The Triton kernel then runs
one program per (tile, chunk of that tile's triangles): each program finds the nearest covering
triangle per pixel in registers and merges it into a shared buffer with a 64-bit atomic min on
``(depth, triangle id)`` packed into one integer. Coverage and depth use the same homogeneous
formulation as the reference backend.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from diffrast.backends.reference import pixel_centres

TILE = 16
# Triangles per program. Tiles with more triangles are split across programs.
CHUNK = 64

_EMPTY = torch.iinfo(torch.int64).max


@triton.jit
def _coverage_kernel(
    verts_ptr,
    tri_list_ptr,
    item_tile_ptr,
    item_start_ptr,
    item_end_ptr,
    px_ptr,
    py_ptr,
    out_ptr,
    num_tri,
    height,
    width,
    tiles_x,
    tiles_per_image,
    TILE: tl.constexpr,
):
    item = tl.program_id(0)
    tile = tl.load(item_tile_ptr + item)
    start = tl.load(item_start_ptr + item)
    end = tl.load(item_end_ptr + item)

    image = tile // tiles_per_image
    tile_in_image = tile % tiles_per_image
    row = (tile_in_image // tiles_x) * TILE + tl.arange(0, TILE)[:, None]
    col = (tile_in_image % tiles_x) * TILE + tl.arange(0, TILE)[None, :]
    px = tl.load(px_ptr + col, mask=col < width, other=0.0)
    py = tl.load(py_ptr + row, mask=row < height, other=0.0)

    best = tl.full((TILE, TILE), 0x7FFFFFFFFFFFFFFF, tl.int64)
    for i in range(start, end):
        g = tl.load(tri_list_ptr + i).to(tl.int64)
        base = verts_ptr + g * 12
        x0, y0, z0, w0 = tl.load(base), tl.load(base + 1), tl.load(base + 2), tl.load(base + 3)
        x1, y1, z1, w1 = tl.load(base + 4), tl.load(base + 5), tl.load(base + 6), tl.load(base + 7)
        x2, y2, z2, w2 = (
            tl.load(base + 8),
            tl.load(base + 9),
            tl.load(base + 10),
            tl.load(base + 11),
        )

        a0, a1, a2 = x0 - px * w0, x1 - px * w1, x2 - px * w2
        c0, c1, c2 = y0 - py * w0, y1 - py * w1, y2 - py * w2
        e0 = a1 * c2 - a2 * c1
        e1 = a2 * c0 - a0 * c2
        e2 = a0 * c1 - a1 * c0
        s = e0 + e1 + e2
        ew = e0 * w0 + e1 * w1 + e2 * w2
        depth = (e0 * z0 + e1 * z1 + e2 * z2) / ew
        inside = (s != 0) & (e0 * s >= 0) & (e1 * s >= 0) & (e2 * s >= 0) & (ew * s > 0)
        inside = inside & (depth >= -1.0) & (depth <= 1.0)

        # Map the float's bits to an integer with the same ordering, then append the id so
        # equal depths resolve to the lower triangle index.
        bits = depth.to(tl.int32, bitcast=True)
        ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        ordered = tl.where(depth == 0.0, 0, ordered)
        key = (ordered.to(tl.int64) << 32) | (g % num_tri)
        best = tl.minimum(best, tl.where(inside, key, 0x7FFFFFFFFFFFFFFF))

    offs = (image * height + row) * width + col
    mask = (row < height) & (col < width) & (best != 0x7FFFFFFFFFFFFFFF)
    tl.atomic_min(out_ptr + offs, best, mask=mask)


def visible_triangles(pos: Tensor, tri: Tensor, resolution: tuple[int, int]) -> Tensor:
    """Index of the nearest covering triangle per pixel, ``[batch, height, width]``, or -1."""
    height, width = resolution
    batch, num_tri = pos.shape[0], tri.shape[0]
    device = pos.device
    pos = pos.detach().float()
    tri = tri.long()

    verts = pos[:, tri]
    tile_lists = _bin_triangles(verts, height, width)
    out = torch.full((batch, height, width), _EMPTY, dtype=torch.int64, device=device)
    if tile_lists is not None:
        tri_list, item_tile, item_start, item_end, tiles_x, tiles_per_image = tile_lists
        px, py = pixel_centres(height, width, pos)
        with torch.cuda.device(device):
            _coverage_kernel[(item_tile.shape[0],)](
                verts.reshape(-1, 12).contiguous(),
                tri_list,
                item_tile,
                item_start,
                item_end,
                px,
                py,
                out,
                num_tri,
                height,
                width,
                tiles_x,
                tiles_per_image,
                TILE=TILE,
            )
    return torch.where(out == _EMPTY, -1, out & 0xFFFFFFFF)


def _bin_triangles(verts: Tensor, height: int, width: int):
    """Assign triangles to the screen tiles their bounding boxes overlap.

    Returns ``(tri_list, item_tile, item_start, item_end, tiles_x, tiles_per_image)``, where
    ``tri_list`` holds flat ``batch * num_tri + tri`` indices grouped by tile, and each work item
    covers ``tri_list[item_start:item_end]`` for tile ``item_tile``. Returns ``None`` when no
    triangle can be visible.
    """
    batch, num_tri = verts.shape[:2]
    device = verts.device
    x, y, z, w = verts.unbind(-1)

    # A triangle is invisible when all vertices lie behind the camera or outside the same clip
    # plane, since every point on it is a non-negative combination of its vertices.
    culled = (w <= 0).all(-1)
    for c in (x, y, z):
        culled |= (c > w).all(-1) | (c < -w).all(-1)

    # Triangles fully in front of the camera are bounded by their projected vertices. Ones that
    # cross the camera plane project to an unbounded region and get the whole screen.
    in_front = (w > 0).all(-1)
    safe_w = torch.where(in_front[..., None], w, 1.0)
    ndc_x, ndc_y = x / safe_w, y / safe_w
    col_min = _to_pixel(ndc_x.amin(-1), width).floor() - 1
    col_max = _to_pixel(ndc_x.amax(-1), width).ceil() + 1
    row_min = _to_pixel(ndc_y.amin(-1), height).floor() - 1
    row_max = _to_pixel(ndc_y.amax(-1), height).ceil() + 1
    col_min = torch.where(in_front, col_min.clamp(0, width - 1), 0).long()
    col_max = torch.where(in_front, col_max.clamp(0, width - 1), width - 1).long()
    row_min = torch.where(in_front, row_min.clamp(0, height - 1), 0).long()
    row_max = torch.where(in_front, row_max.clamp(0, height - 1), height - 1).long()

    tiles_x = triton.cdiv(width, TILE)
    tiles_y = triton.cdiv(height, TILE)
    tiles_per_image = tiles_x * tiles_y
    tx0, tx1 = col_min // TILE, col_max // TILE
    ty0, ty1 = row_min // TILE, row_max // TILE

    b_idx, t_idx = (~culled).nonzero(as_tuple=True)
    if b_idx.numel() == 0:
        return None
    tx0, tx1, ty0, ty1 = tx0[b_idx, t_idx], tx1[b_idx, t_idx], ty0[b_idx, t_idx], ty1[b_idx, t_idx]
    span_x = tx1 - tx0 + 1
    counts = span_x * (ty1 - ty0 + 1)

    owner = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts)
    local = torch.arange(owner.shape[0], device=device) - (counts.cumsum(0) - counts)[owner]
    tile_x = tx0[owner] + local % span_x[owner]
    tile_y = ty0[owner] + local // span_x[owner]
    tile = (b_idx[owner] * tiles_y + tile_y) * tiles_x + tile_x

    order = torch.argsort(tile, stable=True)
    tri_list = (b_idx * num_tri + t_idx)[owner][order].to(torch.int32)

    tile_count = torch.bincount(tile, minlength=batch * tiles_per_image)
    tile_start = tile_count.cumsum(0) - tile_count
    chunks = (tile_count + CHUNK - 1) // CHUNK
    item_tile = torch.repeat_interleave(torch.arange(chunks.shape[0], device=device), chunks)
    item_chunk = (
        torch.arange(item_tile.shape[0], device=device) - (chunks.cumsum(0) - chunks)[item_tile]
    )
    item_start = tile_start[item_tile] + item_chunk * CHUNK
    item_end = torch.minimum(item_start + CHUNK, (tile_start + tile_count)[item_tile])
    return (
        tri_list,
        item_tile.to(torch.int32),
        item_start.to(torch.int32),
        item_end.to(torch.int32),
        tiles_x,
        tiles_per_image,
    )


def _to_pixel(ndc: Tensor, size: int) -> Tensor:
    """Inverse of the pixel-centre mapping ``ndc = (2 * i + 1) / size - 1``."""
    return ((ndc + 1) * size - 1) / 2
