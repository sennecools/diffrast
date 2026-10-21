"""Visibility pass of the Triton rasterizer.

The output is a buffer of 64-bit keys, one per pixel, that pack ``(depth, triangle id)`` so that a
single atomic min keeps the nearest triangle and resolves equal depths to the lower id.

1. ``_setup_kernel`` runs per triangle: it culls, computes the pixel bounding box, and rasterizes
   triangles whose box fits in ``SMALL x SMALL`` pixels directly with atomics. For every other
   visible triangle it writes the range of screen tiles to visit.
2. Those larger triangles are binned into tiles with PyTorch ops, and each tile's list is cut into
   chunks of ``CHUNK`` triangles.
3. ``_tile_kernel`` runs one program per chunk, keeps the nearest hit per pixel in registers, and
   merges it into the buffer with one atomic per pixel.

Coverage and depth use the same homogeneous formulation as the reference backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

from diffrast.backends.reference import pixel_centres

TILE = 16
# Triangles per tile program. Tiles with more triangles are split across programs.
CHUNK = 64
# Triangles whose pixel bounding box fits in SMALL x SMALL skip binning.
SMALL = 4
# Triangles per setup program.
SETUP_BLOCK = 128

EMPTY = torch.iinfo(torch.int64).max


@triton.jit
def _load_vertex(pos_ptr, tri_ptr, b, t, k, num_vert, mask):
    i = tl.load(tri_ptr + t * 3 + k, mask=mask, other=0).to(tl.int64)
    base = pos_ptr + (b * num_vert + i) * 4
    x = tl.load(base, mask=mask, other=0.0)
    y = tl.load(base + 1, mask=mask, other=0.0)
    z = tl.load(base + 2, mask=mask, other=0.0)
    w = tl.load(base + 3, mask=mask, other=0.0)
    return x, y, z, w


@triton.jit
def _depth_key(x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2, px, py, t):
    """Packed ``(depth, t)`` key where the pixel ``(px, py)`` is covered, else the empty key."""
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

    # Map the float's bits to an integer with the same ordering.
    bits = depth.to(tl.int32, bitcast=True)
    ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    ordered = tl.where(depth == 0.0, 0, ordered)
    key = (ordered.to(tl.int64) << 32) | t
    return tl.where(inside, key, 0x7FFFFFFFFFFFFFFF)


@triton.jit
def _pixel_range(lo, hi, size):
    """Indices of the pixel centres within ``[lo, hi]`` in NDC, as an inclusive range."""
    # The margin keeps centres on the boundary despite rounding; extra pixels are tested exactly.
    first = tl.math.ceil(((lo + 1.0) * size - 1.0) * 0.5 - 1e-3)
    last = tl.math.floor(((hi + 1.0) * size - 1.0) * 0.5 + 1e-3)
    first = tl.minimum(tl.maximum(first, 0.0), size * 1.0)
    last = tl.minimum(tl.maximum(last, -1.0), size - 1.0)
    return first.to(tl.int32), last.to(tl.int32)


@triton.jit
def _behind(key, prev_ptr, offs, mask):
    """``key`` where it is ordered after the previous layer's key at ``offs``, else empty."""
    prev = tl.load(prev_ptr + offs, mask=mask, other=0x7FFFFFFFFFFFFFFF)
    return tl.where(key > prev, key, 0x7FFFFFFFFFFFFFFF)


@triton.jit
def _setup_kernel(
    pos_ptr,
    tri_ptr,
    px_ptr,
    py_ptr,
    out_ptr,
    prev_ptr,
    tile_box_ptr,
    total,
    num_tri,
    num_vert,
    height,
    width,
    TILE: tl.constexpr,
    SMALL: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PREV: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = g < total
    b, t = g // num_tri, g % num_tri
    x0, y0, z0, w0 = _load_vertex(pos_ptr, tri_ptr, b, t, 0, num_vert, valid)
    x1, y1, z1, w1 = _load_vertex(pos_ptr, tri_ptr, b, t, 1, num_vert, valid)
    x2, y2, z2, w2 = _load_vertex(pos_ptr, tri_ptr, b, t, 2, num_vert, valid)

    # Every point on a triangle is a non-negative combination of its vertices, so it is invisible
    # when all vertices are behind the camera or outside the same clip plane.
    culled = (w0 <= 0) & (w1 <= 0) & (w2 <= 0)
    culled |= (x0 > w0) & (x1 > w1) & (x2 > w2)
    culled |= (x0 < -w0) & (x1 < -w1) & (x2 < -w2)
    culled |= (y0 > w0) & (y1 > w1) & (y2 > w2)
    culled |= (y0 < -w0) & (y1 < -w1) & (y2 < -w2)
    culled |= (z0 > w0) & (z1 > w1) & (z2 > w2)
    culled |= (z0 < -w0) & (z1 < -w1) & (z2 < -w2)

    # Triangles in front of the camera are bounded by their projected vertices. Ones crossing the
    # camera plane project to an unbounded region and cover the whole screen.
    in_front = (w0 > 0) & (w1 > 0) & (w2 > 0)
    sw0, sw1, sw2 = (
        tl.where(in_front, w0, 1.0),
        tl.where(in_front, w1, 1.0),
        tl.where(in_front, w2, 1.0),
    )
    nx0, nx1, nx2 = x0 / sw0, x1 / sw1, x2 / sw2
    ny0, ny1, ny2 = y0 / sw0, y1 / sw1, y2 / sw2
    col0, col1 = _pixel_range(
        tl.minimum(tl.minimum(nx0, nx1), nx2), tl.maximum(tl.maximum(nx0, nx1), nx2), width
    )
    row0, row1 = _pixel_range(
        tl.minimum(tl.minimum(ny0, ny1), ny2), tl.maximum(tl.maximum(ny0, ny1), ny2), height
    )
    col0, col1 = tl.where(in_front, col0, 0), tl.where(in_front, col1, width - 1)
    row0, row1 = tl.where(in_front, row0, 0), tl.where(in_front, row1, height - 1)

    visible = valid & ~culled & (col0 <= col1) & (row0 <= row1)
    small = visible & in_front & (col1 - col0 < SMALL) & (row1 - row0 < SMALL)

    lane = tl.arange(0, SMALL * SMALL)[None, :]
    col = col0[:, None] + lane % SMALL
    row = row0[:, None] + lane // SMALL
    lane_ok = small[:, None] & (col <= col1[:, None]) & (row <= row1[:, None])
    px = tl.load(px_ptr + col, mask=lane_ok, other=0.0)
    py = tl.load(py_ptr + row, mask=lane_ok, other=0.0)
    key = _depth_key(
        x0[:, None], y0[:, None], z0[:, None], w0[:, None],
        x1[:, None], y1[:, None], z1[:, None], w1[:, None],
        x2[:, None], y2[:, None], z2[:, None], w2[:, None],
        px, py, t[:, None],
    )  # fmt: skip
    offs = (b[:, None] * height + row) * width + col
    if HAS_PREV:
        key = _behind(key, prev_ptr, offs, lane_ok)
    tl.atomic_min(out_ptr + offs, key, mask=lane_ok & (key != 0x7FFFFFFFFFFFFFFF))

    large = visible & ~small
    box = tile_box_ptr + g * 4
    tl.store(box, tl.where(large, col0 // TILE, -1), mask=valid)
    tl.store(box + 1, col1 // TILE, mask=valid)
    tl.store(box + 2, row0 // TILE, mask=valid)
    tl.store(box + 3, row1 // TILE, mask=valid)


@triton.jit
def _tile_kernel(
    pos_ptr,
    tri_ptr,
    tri_list_ptr,
    item_tile_ptr,
    item_start_ptr,
    item_end_ptr,
    px_ptr,
    py_ptr,
    out_ptr,
    prev_ptr,
    num_tri,
    num_vert,
    height,
    width,
    tiles_x,
    tiles_per_image,
    TILE: tl.constexpr,
    HAS_PREV: tl.constexpr,
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

    offs = (image * height + row) * width + col
    inside = (row < height) & (col < width)
    best = tl.full((TILE, TILE), 0x7FFFFFFFFFFFFFFF, tl.int64)
    for i in range(start, end):
        g = tl.load(tri_list_ptr + i).to(tl.int64)
        b, t = g // num_tri, g % num_tri
        x0, y0, z0, w0 = _load_vertex(pos_ptr, tri_ptr, b, t, 0, num_vert, True)
        x1, y1, z1, w1 = _load_vertex(pos_ptr, tri_ptr, b, t, 1, num_vert, True)
        x2, y2, z2, w2 = _load_vertex(pos_ptr, tri_ptr, b, t, 2, num_vert, True)
        key = _depth_key(x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2, px, py, t)
        if HAS_PREV:
            key = _behind(key, prev_ptr, offs, inside)
        best = tl.minimum(best, key)

    mask = inside & (best != 0x7FFFFFFFFFFFFFFF)
    tl.atomic_min(out_ptr + offs, best, mask=mask)


@dataclass
class Pass:
    """Inputs and buffers shared by the stages of one visibility pass."""

    pos: Tensor
    tri: Tensor
    px: Tensor
    py: Tensor
    keys: Tensor
    tile_box: Tensor
    height: int
    width: int
    previous: Tensor | None = None

    @property
    def batch(self) -> int:
        return self.pos.shape[0]

    @property
    def num_tri(self) -> int:
        return self.tri.shape[0]


def visible_keys(
    pos: Tensor, tri: Tensor, resolution: tuple[int, int], previous: Tensor | None = None
) -> Tensor:
    """Packed ``(depth, triangle)`` key of the nearest triangle per pixel, ``[batch, h, w]``.

    Pixels no triangle covers hold ``EMPTY``. The low 32 bits of a key are the triangle index.
    With ``previous``, the keys of an earlier layer, only keys ordered after it are considered,
    which gives the next depth layer.
    """
    state = prepare(pos, tri, resolution, previous)
    setup_stage(state)
    tile_stage(state, bin_stage(state))
    return state.keys


def visible_triangles(pos: Tensor, tri: Tensor, resolution: tuple[int, int]) -> Tensor:
    """Index of the nearest covering triangle per pixel, ``[batch, height, width]``, or -1."""
    keys = visible_keys(pos, tri, resolution)
    return torch.where(keys == EMPTY, -1, keys & 0xFFFFFFFF)


def prepare(
    pos: Tensor, tri: Tensor, resolution: tuple[int, int], previous: Tensor | None = None
) -> Pass:
    height, width = resolution
    pos = pos.detach().float().contiguous()
    px, py = pixel_centres(height, width, pos)
    batch, num_tri = pos.shape[0], tri.shape[0]
    return Pass(
        pos=pos,
        tri=tri.to(torch.int32).contiguous(),
        px=px,
        py=py,
        keys=torch.full((batch, height, width), EMPTY, dtype=torch.int64, device=pos.device),
        tile_box=torch.empty(batch * num_tri, 4, dtype=torch.int32, device=pos.device),
        height=height,
        width=width,
        previous=previous,
    )


def setup_stage(state: Pass) -> None:
    """Cull and bound every triangle; rasterize small ones; mark the rest for tiling."""
    total = state.batch * state.num_tri
    with torch.cuda.device(state.pos.device):
        _setup_kernel[(triton.cdiv(total, SETUP_BLOCK),)](
            state.pos, state.tri, state.px, state.py, state.keys, _previous(state),
            state.tile_box, total, state.num_tri, state.pos.shape[1], state.height, state.width,
            TILE=TILE, SMALL=SMALL, BLOCK=SETUP_BLOCK, HAS_PREV=state.previous is not None,
        )  # fmt: skip


def _previous(state: Pass) -> Tensor:
    # Without a previous layer the argument is unused; the code reading it is compiled out.
    return state.previous if state.previous is not None else state.keys


def bin_stage(state: Pass):
    return _bin_triangles(state.tile_box, state.batch, state.height, state.width)


def tile_stage(state: Pass, lists) -> None:
    """Rasterize the binned triangles tile by tile."""
    if lists is None:
        return
    tri_list, item_tile, item_start, item_end, tiles_x, tiles_per_image = lists
    with torch.cuda.device(state.pos.device):
        _tile_kernel[(item_tile.shape[0],)](
            state.pos, state.tri, tri_list, item_tile, item_start, item_end, state.px, state.py,
            state.keys, _previous(state), state.num_tri, state.pos.shape[1], state.height,
            state.width, tiles_x, tiles_per_image, TILE=TILE, HAS_PREV=state.previous is not None,
        )  # fmt: skip


def _bin_triangles(tile_box: Tensor, batch: int, height: int, width: int):
    """Assign the triangles marked for tiling to the screen tiles their bounding boxes overlap.

    ``tile_box`` holds ``(first, last)`` tile columns and rows per triangle, with ``first`` column
    -1 for triangles that need no tiling. Returns ``(tri_list, item_tile, item_start, item_end,
    tiles_x, tiles_per_image)``, where ``tri_list`` holds flat ``batch * num_tri + tri`` indices
    grouped by tile, and each work item covers ``tri_list[item_start:item_end]`` of tile
    ``item_tile``. Returns ``None`` when there is nothing to tile.
    """
    device = tile_box.device
    num_tri = tile_box.shape[0] // batch
    tiles_x = triton.cdiv(width, TILE)
    tiles_y = triton.cdiv(height, TILE)
    tiles_per_image = tiles_x * tiles_y

    g = (tile_box[:, 0] >= 0).nonzero().squeeze(1)
    if g.numel() == 0:
        return None
    tx0, tx1, ty0, ty1 = tile_box[g].long().unbind(-1)
    span_x = tx1 - tx0 + 1
    counts = span_x * (ty1 - ty0 + 1)

    owner = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts)
    local = torch.arange(owner.shape[0], device=device) - (counts.cumsum(0) - counts)[owner]
    tile_x = tx0[owner] + local % span_x[owner]
    tile_y = ty0[owner] + local // span_x[owner]
    tile = (g[owner] // num_tri * tiles_y + tile_y) * tiles_x + tile_x

    sorted_tile, order = torch.sort(tile, stable=True)
    tri_list = g[owner][order].to(torch.int32)

    # Binary search on the sorted list instead of bincount, whose atomics contend heavily when
    # many triangles share few tiles.
    bounds = torch.searchsorted(
        sorted_tile, torch.arange(batch * tiles_per_image + 1, device=device)
    )
    tile_start = bounds[:-1]
    tile_count = bounds[1:] - tile_start
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
