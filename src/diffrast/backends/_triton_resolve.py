"""Triton kernels that build ``rast`` and ``rast_db`` from the visibility keys.

Per pixel, the visible triangle's homogeneous edge functions ``e = a x c`` (with ``a = x - px w``
and ``c = y - py w`` per vertex) give the perspective-correct barycentrics ``b = e / S``,
``S = sum(e)``, and ``z / w = (e . z) / (e . w)``. Their screen-space derivatives follow from
``de/dpx = c x w`` and ``de/dpy = w x a``. These are the reference backend's formulas.

The backward kernel runs the same steps in reverse. For a cross product ``f = u x v`` with
upstream gradient ``G``, the gradients are ``v x G`` for ``u`` and ``G x u`` for ``v``. Vertex
gradients are summed in registers over pixels sharing a vertex before they are written with
atomics, since neighbouring pixels mostly show the same triangle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from diffrast.backends.reference import pixel_centres

# Each program handles one tile of pixels, so that its pixels mostly share a few triangles.
TILE_H, TILE_W = 8, 16
# The in-register reduction of vertex gradients stops after MAX_ROUNDS rounds, or once rounds
# merge fewer than MIN_GROUP pixels on average; the remaining gradients go out as plain atomics.
MAX_ROUNDS = 16
MIN_GROUP = 4


@triton.jit
def _cross(u0, u1, u2, v0, v1, v2):
    return u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0


@triton.jit
def _tile_pixels(height, width, TILE_H: tl.constexpr, TILE_W: tl.constexpr):
    """Flat indices of this program's tile of pixels, and which of them lie inside the image."""
    tiles_x = tl.cdiv(width, TILE_W)
    tiles_per_image = tiles_x * tl.cdiv(height, TILE_H)
    program = tl.program_id(0).to(tl.int64)
    image, tile = program // tiles_per_image, program % tiles_per_image
    lane = tl.arange(0, TILE_H * TILE_W)
    row = (tile // tiles_x) * TILE_H + lane // TILE_W
    col = (tile % tiles_x) * TILE_W + lane % TILE_W
    return (image * height + row) * width + col, (row < height) & (col < width)


@triton.jit
def _pixel(keys_ptr, tri_ptr, pos_ptr, px_ptr, py_ptr, p, valid, height, width, num_vert):
    """The pixel's visible triangle: coverage, index, vertex indices and clip coordinates."""
    key = tl.load(keys_ptr + p, mask=valid, other=0x7FFFFFFFFFFFFFFF)
    covered = valid & (key != 0x7FFFFFFFFFFFFFFF)
    tid = key & 0xFFFFFFFF
    b = p // (height * width)
    px = tl.load(px_ptr + p % width, mask=valid, other=0.0)
    py = tl.load(py_ptr + (p // width) % height, mask=valid, other=0.0)
    i0 = tl.load(tri_ptr + tid * 3, mask=covered, other=0).to(tl.int64)
    i1 = tl.load(tri_ptr + tid * 3 + 1, mask=covered, other=0).to(tl.int64)
    i2 = tl.load(tri_ptr + tid * 3 + 2, mask=covered, other=0).to(tl.int64)
    v0 = pos_ptr + (b * num_vert + i0) * 4
    v1 = pos_ptr + (b * num_vert + i1) * 4
    v2 = pos_ptr + (b * num_vert + i2) * 4
    x0, y0 = tl.load(v0, mask=covered, other=0.0), tl.load(v0 + 1, mask=covered, other=0.0)
    z0, w0 = tl.load(v0 + 2, mask=covered, other=0.0), tl.load(v0 + 3, mask=covered, other=1.0)
    x1, y1 = tl.load(v1, mask=covered, other=0.0), tl.load(v1 + 1, mask=covered, other=0.0)
    z1, w1 = tl.load(v1 + 2, mask=covered, other=0.0), tl.load(v1 + 3, mask=covered, other=1.0)
    x2, y2 = tl.load(v2, mask=covered, other=0.0), tl.load(v2 + 1, mask=covered, other=0.0)
    z2, w2 = tl.load(v2 + 2, mask=covered, other=0.0), tl.load(v2 + 3, mask=covered, other=1.0)
    return (covered, tid, b, px, py, i0, i1, i2,
            x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2)  # fmt: skip


@triton.jit
def _forward_kernel(keys_ptr, tri_ptr, pos_ptr, px_ptr, py_ptr, rast_ptr, db_ptr,
                    height, width, num_vert,
                    TILE_H: tl.constexpr, TILE_W: tl.constexpr):  # fmt: skip
    p, valid = _tile_pixels(height, width, TILE_H, TILE_W)
    (covered, tid, _, px, py, _, _, _,
     x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2) = _pixel(
        keys_ptr, tri_ptr, pos_ptr, px_ptr, py_ptr, p, valid, height, width, num_vert
    )  # fmt: skip

    a0, a1, a2 = x0 - px * w0, x1 - px * w1, x2 - px * w2
    c0, c1, c2 = y0 - py * w0, y1 - py * w1, y2 - py * w2
    e0, e1, e2 = _cross(a0, a1, a2, c0, c1, c2)
    s = tl.where(covered, e0 + e1 + e2, 1.0)
    b0, b1 = e0 / s, e1 / s
    ew = tl.where(covered, e0 * w0 + e1 * w1 + e2 * w2, 1.0)
    depth = (e0 * z0 + e1 * z1 + e2 * z2) / ew

    dx0, dx1, dx2 = _cross(c0, c1, c2, w0, w1, w2)
    dy0, dy1, dy2 = _cross(w0, w1, w2, a0, a1, a2)
    sx, sy = dx0 + dx1 + dx2, dy0 + dy1 + dy2
    scale_x, scale_y = 2.0 / width, 2.0 / height

    zero = tl.zeros_like(b0)
    out = rast_ptr + p * 4
    tl.store(out, tl.where(covered, b0, zero), mask=valid)
    tl.store(out + 1, tl.where(covered, b1, zero), mask=valid)
    tl.store(out + 2, tl.where(covered, depth, zero), mask=valid)
    tl.store(out + 3, tl.where(covered, (tid + 1).to(b0.dtype), zero), mask=valid)
    out = db_ptr + p * 4
    tl.store(out, tl.where(covered, (dx0 - b0 * sx) / s * scale_x, zero), mask=valid)
    tl.store(out + 1, tl.where(covered, (dy0 - b0 * sy) / s * scale_y, zero), mask=valid)
    tl.store(out + 2, tl.where(covered, (dx1 - b1 * sx) / s * scale_x, zero), mask=valid)
    tl.store(out + 3, tl.where(covered, (dy1 - b1 * sy) / s * scale_y, zero), mask=valid)


@triton.jit
def _scatter_vertex(grad_ptr, address, ok, gx, gy, gz, gw,
                    MAX_ROUNDS: tl.constexpr, MIN_GROUP: tl.constexpr):  # fmt: skip
    """Atomically add ``(gx, gy, gz, gw)`` to the vertex rows ``address`` of ``grad_ptr``.

    Round by round, the pixels sharing the lowest remaining vertex are summed in registers and
    written with one atomic per component. Once the rounds merge fewer than ``MIN_GROUP`` pixels
    each on average, the rest are written per pixel: for dense meshes, where pixels rarely share
    a vertex within a tile, further rounds would cost more than the contention they save.
    """
    empty = 0x7FFFFFFFFFFFFFFF
    key = tl.where(ok, address, empty)
    rounds = tl.full((), 0, tl.int32)
    merged = tl.full((), 0, tl.int32)
    go = tl.min(key, axis=0) != empty
    while go:
        ref = tl.min(key, axis=0)
        group = key == ref
        base = grad_ptr + ref * 4
        tl.atomic_add(base, tl.sum(tl.where(group, gx, 0.0), axis=0))
        tl.atomic_add(base + 1, tl.sum(tl.where(group, gy, 0.0), axis=0))
        tl.atomic_add(base + 2, tl.sum(tl.where(group, gz, 0.0), axis=0))
        tl.atomic_add(base + 3, tl.sum(tl.where(group, gw, 0.0), axis=0))
        key = tl.where(group, empty, key)
        rounds += 1
        merged += tl.sum(group.to(tl.int32), axis=0)
        worthwhile = merged >= MIN_GROUP * rounds
        go = worthwhile & (rounds < MAX_ROUNDS) & (tl.min(key, axis=0) != empty)
    rest = key != empty
    base = grad_ptr + address * 4
    tl.atomic_add(base, gx, mask=rest)
    tl.atomic_add(base + 1, gy, mask=rest)
    tl.atomic_add(base + 2, gz, mask=rest)
    tl.atomic_add(base + 3, gw, mask=rest)


@triton.jit
def _backward_kernel(keys_ptr, tri_ptr, pos_ptr, px_ptr, py_ptr, grad_rast_ptr, grad_db_ptr,
                     grad_pos_ptr, height, width, num_vert, TILE_H: tl.constexpr,
                     TILE_W: tl.constexpr, GRAD_DB: tl.constexpr, MAX_ROUNDS: tl.constexpr,
                     MIN_GROUP: tl.constexpr):  # fmt: skip
    p, valid = _tile_pixels(height, width, TILE_H, TILE_W)
    (covered, _, b, px, py, i0, i1, i2,
     x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2) = _pixel(
        keys_ptr, tri_ptr, pos_ptr, px_ptr, py_ptr, p, valid, height, width, num_vert
    )  # fmt: skip

    # Forward quantities.
    a0, a1, a2 = x0 - px * w0, x1 - px * w1, x2 - px * w2
    c0, c1, c2 = y0 - py * w0, y1 - py * w1, y2 - py * w2
    e0, e1, e2 = _cross(a0, a1, a2, c0, c1, c2)
    s = tl.where(covered, e0 + e1 + e2, 1.0)
    b0, b1, b2 = e0 / s, e1 / s, e2 / s
    ew = tl.where(covered, e0 * w0 + e1 * w1 + e2 * w2, 1.0)
    depth = (e0 * z0 + e1 * z1 + e2 * z2) / ew

    # Upstream gradients; the triangle id channel has none.
    g = grad_rast_ptr + p * 4
    gb0 = tl.load(g, mask=covered, other=0.0)
    gb1 = tl.load(g + 1, mask=covered, other=0.0)
    g_depth = tl.load(g + 2, mask=covered, other=0.0)
    gb2 = tl.zeros_like(gb0)
    ge0, ge1, ge2 = tl.zeros_like(gb0), tl.zeros_like(gb0), tl.zeros_like(gb0)
    ga0, ga1, ga2 = tl.zeros_like(gb0), tl.zeros_like(gb0), tl.zeros_like(gb0)
    gc0, gc1, gc2 = tl.zeros_like(gb0), tl.zeros_like(gb0), tl.zeros_like(gb0)
    gw0, gw1, gw2 = tl.zeros_like(gb0), tl.zeros_like(gb0), tl.zeros_like(gb0)
    gs = tl.zeros_like(gb0)

    if GRAD_DB:
        dx0, dx1, dx2 = _cross(c0, c1, c2, w0, w1, w2)
        dy0, dy1, dy2 = _cross(w0, w1, w2, a0, a1, a2)
        sx, sy = dx0 + dx1 + dx2, dy0 + dy1 + dy2
        g = grad_db_ptr + p * 4
        gbx0 = tl.load(g, mask=covered, other=0.0) * (2.0 / width)
        gby0 = tl.load(g + 1, mask=covered, other=0.0) * (2.0 / height)
        gbx1 = tl.load(g + 2, mask=covered, other=0.0) * (2.0 / width)
        gby1 = tl.load(g + 3, mask=covered, other=0.0) * (2.0 / height)
        # db_dx_k = (de_dx_k - b_k * sum(de_dx)) / s, and the same along y.
        dbx0, dbx1 = (dx0 - b0 * sx) / s, (dx1 - b1 * sx) / s
        dby0, dby1 = (dy0 - b0 * sy) / s, (dy1 - b1 * sy) / s
        shared_x = (gbx0 * b0 + gbx1 * b1) / s
        shared_y = (gby0 * b0 + gby1 * b1) / s
        gdx0, gdx1, gdx2 = gbx0 / s - shared_x, gbx1 / s - shared_x, -shared_x
        gdy0, gdy1, gdy2 = gby0 / s - shared_y, gby1 / s - shared_y, -shared_y
        gb0 -= (gbx0 * sx + gby0 * sy) / s
        gb1 -= (gbx1 * sx + gby1 * sy) / s
        gs -= (gbx0 * dbx0 + gbx1 * dbx1 + gby0 * dby0 + gby1 * dby1) / s
        # de_dx = c x w and de_dy = w x a.
        t0, t1, t2 = _cross(w0, w1, w2, gdx0, gdx1, gdx2)
        gc0, gc1, gc2 = gc0 + t0, gc1 + t1, gc2 + t2
        t0, t1, t2 = _cross(gdx0, gdx1, gdx2, c0, c1, c2)
        gw0, gw1, gw2 = gw0 + t0, gw1 + t1, gw2 + t2
        t0, t1, t2 = _cross(a0, a1, a2, gdy0, gdy1, gdy2)
        gw0, gw1, gw2 = gw0 + t0, gw1 + t1, gw2 + t2
        t0, t1, t2 = _cross(gdy0, gdy1, gdy2, w0, w1, w2)
        ga0, ga1, ga2 = ga0 + t0, ga1 + t1, ga2 + t2

    # b = e / s with s = sum(e).
    gs -= (gb0 * b0 + gb1 * b1 + gb2 * b2) / s
    ge0, ge1, ge2 = ge0 + gb0 / s + gs, ge1 + gb1 / s + gs, ge2 + gb2 / s + gs
    # depth = (e . z) / (e . w).
    g_ew = g_depth / ew
    ge0 += g_ew * (z0 - depth * w0)
    ge1 += g_ew * (z1 - depth * w1)
    ge2 += g_ew * (z2 - depth * w2)
    gz0, gz1, gz2 = g_ew * e0, g_ew * e1, g_ew * e2
    gw0 -= g_ew * depth * e0
    gw1 -= g_ew * depth * e1
    gw2 -= g_ew * depth * e2
    # e = a x c.
    t0, t1, t2 = _cross(c0, c1, c2, ge0, ge1, ge2)
    ga0, ga1, ga2 = ga0 + t0, ga1 + t1, ga2 + t2
    t0, t1, t2 = _cross(ge0, ge1, ge2, a0, a1, a2)
    gc0, gc1, gc2 = gc0 + t0, gc1 + t1, gc2 + t2
    # a = x - px * w and c = y - py * w.
    gw0 -= px * ga0 + py * gc0
    gw1 -= px * ga1 + py * gc1
    gw2 -= px * ga2 + py * gc2

    _scatter_vertex(grad_pos_ptr, b * num_vert + i0, covered, ga0, gc0, gz0, gw0,
                    MAX_ROUNDS, MIN_GROUP)  # fmt: skip
    _scatter_vertex(grad_pos_ptr, b * num_vert + i1, covered, ga1, gc1, gz1, gw1,
                    MAX_ROUNDS, MIN_GROUP)  # fmt: skip
    _scatter_vertex(grad_pos_ptr, b * num_vert + i2, covered, ga2, gc2, gz2, gw2,
                    MAX_ROUNDS, MIN_GROUP)  # fmt: skip


class Resolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos: Tensor, tri: Tensor, keys: Tensor, grad_db: bool):
        batch, height, width = keys.shape
        pos_c = pos.detach().contiguous()
        tri = tri.to(torch.int32).contiguous()
        px, py = pixel_centres(height, width, pos_c)
        rast = pos.new_empty(batch, height, width, 4)
        rast_db = pos.new_empty(batch, height, width, 4)
        with torch.cuda.device(pos.device):
            _forward_kernel[_grid(batch, height, width)](
                keys, tri, pos_c, px, py, rast, rast_db, height, width, pos.shape[1],
                TILE_H=TILE_H, TILE_W=TILE_W,
            )  # fmt: skip
        ctx.save_for_backward(pos_c, tri, keys, px, py)
        ctx.grad_db = grad_db
        if not grad_db:
            ctx.mark_non_differentiable(rast_db)
        return rast, rast_db

    @staticmethod
    def backward(ctx, grad_rast: Tensor | None, grad_rast_db: Tensor | None):
        pos, tri, keys, px, py = ctx.saved_tensors
        batch, height, width = keys.shape
        if grad_rast is None:
            grad_rast = pos.new_zeros(batch, height, width, 4)
        use_db = ctx.grad_db and grad_rast_db is not None
        grad_rast = grad_rast.contiguous()
        grad_rast_db = grad_rast_db.contiguous() if use_db else grad_rast
        grad_pos = torch.zeros_like(pos)
        with torch.cuda.device(pos.device):
            _backward_kernel[_grid(batch, height, width)](
                keys, tri, pos, px, py, grad_rast, grad_rast_db, grad_pos,
                height, width, pos.shape[1], TILE_H=TILE_H, TILE_W=TILE_W, GRAD_DB=use_db,
                MAX_ROUNDS=MAX_ROUNDS, MIN_GROUP=MIN_GROUP,
            )  # fmt: skip
        return grad_pos, None, None, None


def _grid(batch: int, height: int, width: int) -> tuple[int]:
    return (batch * triton.cdiv(height, TILE_H) * triton.cdiv(width, TILE_W),)


def resolve(pos: Tensor, tri: Tensor, keys: Tensor, grad_db: bool) -> tuple[Tensor, Tensor]:
    return Resolve.apply(pos, tri, keys, grad_db)
