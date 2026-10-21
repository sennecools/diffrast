"""Triton kernels for antialias.

Silhouette edges are classified with the reference backend's PyTorch code, which is per triangle
and cheap. The kernels then run one lane per pixel, handling the pairs it forms with its right and
lower neighbours exactly as the reference backend does. A pixel is touched by at most four pairs,
so blended colours and gradients are added with atomics. In the backward pass, the gradient of the
crossing point ``t`` is chained analytically into the two vertices of the crossed edge.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from diffrast.backends.reference import _silhouette_edges, pixel_centres

BLOCK_PIXELS = 128
BLOCK_CHANNELS = 16


@triton.jit
def _pick(k, v0, v1, v2):
    return tl.where(k == 0, v0, tl.where(k == 1, v1, v2))


@triton.jit
def _vertex(pos_ptr, tri_ptr, b, tid, corner, num_vert, mask):
    i = tl.load(tri_ptr + tid * 3 + corner, mask=mask, other=0).to(tl.int64)
    base = pos_ptr + (b * num_vert + i) * 4
    x = tl.load(base, mask=mask, other=0.0)
    y = tl.load(base + 1, mask=mask, other=0.0)
    w = tl.load(base + 3, mask=mask, other=1.0)
    return i, x, y, w


@triton.jit
def _edge(xi, yi, wi, xj, yj, wj, px, py):
    """Homogeneous edge function through vertices i and j at the pixel ``(px, py)``."""
    return (xi - px * wi) * (yj - py * wj) - (xj - px * wj) * (yi - py * wi)


@triton.jit
def _crossing(pos_ptr, tri_ptr, sil_ptr, b, tri_id, src_x, src_y, dst_x, dst_y, width, height,
              num_vert, num_tri, mask, HORIZONTAL: tl.constexpr):  # fmt: skip
    """Where the triangle visible at ``src`` is exited towards ``dst``: ``(found, t, edge)``.

    Mirrors ``reference._edge_crossing``; see there for when no crossing is reported.
    """
    visible = mask & (tri_id >= 0)
    tid = tl.maximum(tri_id, 0)
    _, x0, y0, w0 = _vertex(pos_ptr, tri_ptr, b, tid, 0, num_vert, visible)
    _, x1, y1, w1 = _vertex(pos_ptr, tri_ptr, b, tid, 1, num_vert, visible)
    _, x2, y2, w2 = _vertex(pos_ptr, tri_ptr, b, tid, 2, num_vert, visible)

    s0 = _edge(x1, y1, w1, x2, y2, w2, src_x, src_y)
    s1 = _edge(x2, y2, w2, x0, y0, w0, src_x, src_y)
    s2 = _edge(x0, y0, w0, x1, y1, w1, src_x, src_y)
    d0 = _edge(x1, y1, w1, x2, y2, w2, dst_x, dst_y)
    d1 = _edge(x2, y2, w2, x0, y0, w0, dst_x, dst_y)
    d2 = _edge(x0, y0, w0, x1, y1, w1, dst_x, dst_y)
    sign = tl.where(s0 + s1 + s2 < 0, -1.0, 1.0)
    out0, out1, out2 = d0 * sign < 0, d1 * sign < 0, d2 * sign < 0
    inf = float("inf")
    t0 = tl.where(out0, s0 / tl.where(out0, s0 - d0, 1.0), inf)
    t1 = tl.where(out1, s1 / tl.where(out1, s1 - d1, 1.0), inf)
    t2 = tl.where(out2, s2 / tl.where(out2, s2 - d2, 1.0), inf)
    edge = tl.where(t1 < t0, 1, 0)
    t = tl.minimum(t0, t1)
    edge = tl.where(t2 < t, 2, edge)
    t = tl.minimum(t, t2)

    # Screen-space gradient of the chosen edge function, in pixel units.
    xi, yi, wi = _pick(edge, x1, x2, x0), _pick(edge, y1, y2, y0), _pick(edge, w1, w2, w0)
    xj, yj, wj = _pick(edge, x2, x0, x1), _pick(edge, y2, y0, y1), _pick(edge, w2, w0, w1)
    ai, aj = xi - src_x * wi, xj - src_x * wj
    ci, cj = yi - src_y * wi, yj - src_y * wj
    normal_x = tl.abs(wj * ci - wi * cj) * (2.0 / width)
    normal_y = tl.abs(aj * wi - ai * wj) * (2.0 / height)
    along_axis = normal_x >= normal_y if HORIZONTAL else normal_y > normal_x

    silhouette = tl.load(sil_ptr + (b * num_tri + tid) * 3 + edge, mask=visible, other=0) != 0
    found = visible & (out0 | out1 | out2) & silhouette & along_axis
    return found, tl.where(found, t, 0.0), edge


@triton.jit
def _pair(pos_ptr, tri_ptr, sil_ptr, rast_ptr, px_ptr, py_ptr, p, b, row, col, valid, height,
          width, num_vert, num_tri, HORIZONTAL: tl.constexpr):  # fmt: skip
    """Pixel ``p`` and its right (or lower) neighbour ``q``: which triangle owns the edge between
    them, where it crosses, and how much each pixel blends."""
    if HORIZONTAL:
        pair = valid & (col < width - 1)
        q = p + 1
        row_q, col_q = row, col + 1
    else:
        pair = valid & (row < height - 1)
        q = p + width
        row_q, col_q = row + 1, col
    id_p = (tl.load(rast_ptr + p * 4 + 3, mask=pair, other=0.0) + 0.5).to(tl.int64) - 1
    id_q = (tl.load(rast_ptr + q * 4 + 3, mask=pair, other=0.0) + 0.5).to(tl.int64) - 1
    pair = pair & (id_p != id_q)
    x_p = tl.load(px_ptr + col, mask=pair, other=0.0)
    y_p = tl.load(py_ptr + row, mask=pair, other=0.0)
    x_q = tl.load(px_ptr + col_q, mask=pair, other=0.0)
    y_q = tl.load(py_ptr + row_q, mask=pair, other=0.0)

    from_p, t_p, edge_p = _crossing(pos_ptr, tri_ptr, sil_ptr, b, id_p, x_p, y_p, x_q, y_q,
                                    width, height, num_vert, num_tri, pair, HORIZONTAL)  # fmt: skip
    from_q, t_q, edge_q = _crossing(pos_ptr, tri_ptr, sil_ptr, b, id_q, x_q, y_q, x_p, y_p,
                                    width, height, num_vert, num_tri, pair, HORIZONTAL)  # fmt: skip
    depth_p = tl.load(rast_ptr + p * 4 + 2, mask=pair, other=0.0)
    depth_q = tl.load(rast_ptr + q * 4 + 2, mask=pair, other=0.0)
    use_p = from_p & (~from_q | (depth_p <= depth_q))
    use_q = from_q & ~use_p
    t = tl.where(use_p, t_p, 1 - t_q)
    active = use_p | use_q
    blend_p = tl.where(active & (t < 0.5), 0.5 - t, 0.0)
    blend_q = tl.where(active & (t >= 0.5), t - 0.5, 0.0)
    owner = tl.where(use_p, id_p, id_q)
    edge = tl.where(use_p, edge_p, edge_q)
    return (q, active, use_p, t, blend_p, blend_q, owner, edge, x_p, y_p, x_q, y_q)


@triton.jit
def _forward_kernel(color_ptr, rast_ptr, pos_ptr, tri_ptr, sil_ptr, px_ptr, py_ptr, out_ptr,
                    num_pixels, height, width, num_vert, num_tri, channels,
                    BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):  # fmt: skip
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    b = p // (height * width)
    row = (p // width) % height
    col = p % width
    for direction in tl.static_range(2):
        q, active, _, _, blend_p, blend_q, _, _, _, _, _, _ = _pair(
            pos_ptr, tri_ptr, sil_ptr, rast_ptr, px_ptr, py_ptr, p, b, row, col, valid,
            height, width, num_vert, num_tri, direction == 0,
        )  # fmt: skip
        for c0 in range(0, channels, BLOCK_C):
            c = c0 + tl.arange(0, BLOCK_C)[None, :]
            m = active[:, None] & (c < channels)
            c_p = tl.load(color_ptr + p[:, None] * channels + c, mask=m, other=0.0)
            c_q = tl.load(color_ptr + q[:, None] * channels + c, mask=m, other=0.0)
            tl.atomic_add(out_ptr + p[:, None] * channels + c, blend_p[:, None] * (c_q - c_p),
                          mask=m & (blend_p[:, None] != 0))  # fmt: skip
            tl.atomic_add(out_ptr + q[:, None] * channels + c, blend_q[:, None] * (c_p - c_q),
                          mask=m & (blend_q[:, None] != 0))  # fmt: skip


@triton.jit
def _add_edge_gradient(grad_pos_ptr, pos_ptr, tri_ptr, b, tid, edge, grad_t, src_x, src_y,
                       dst_x, dst_y, num_vert, mask):  # fmt: skip
    """Chain ``d loss / d t`` into the two vertices of the crossed edge.

    With ``t = e_s / (e_s - e_d)`` for the edge function ``e`` at the source and destination
    pixels, ``dt/de_s = -e_d / (e_s - e_d)^2`` and ``dt/de_d = e_s / (e_s - e_d)^2``.
    """
    corner_i = tl.where(edge == 0, 1, tl.where(edge == 1, 2, 0))
    corner_j = tl.where(edge == 0, 2, tl.where(edge == 1, 0, 1))
    vi, xi, yi, wi = _vertex(pos_ptr, tri_ptr, b, tid, corner_i, num_vert, mask)
    vj, xj, yj, wj = _vertex(pos_ptr, tri_ptr, b, tid, corner_j, num_vert, mask)
    e_s = _edge(xi, yi, wi, xj, yj, wj, src_x, src_y)
    e_d = _edge(xi, yi, wi, xj, yj, wj, dst_x, dst_y)
    denom = e_s - e_d
    denom = tl.where(denom == 0, 1.0, denom)
    g_s = grad_t * -e_d / (denom * denom)
    g_d = grad_t * e_s / (denom * denom)

    # e = a_i * c_j - a_j * c_i with a = x - px * w and c = y - py * w.
    ai_s, aj_s = xi - src_x * wi, xj - src_x * wj
    ci_s, cj_s = yi - src_y * wi, yj - src_y * wj
    ai_d, aj_d = xi - dst_x * wi, xj - dst_x * wj
    ci_d, cj_d = yi - dst_y * wi, yj - dst_y * wj
    gx_i = g_s * cj_s + g_d * cj_d
    gx_j = -(g_s * ci_s + g_d * ci_d)
    gy_i = -(g_s * aj_s + g_d * aj_d)
    gy_j = g_s * ai_s + g_d * ai_d
    gw_i = g_s * (src_y * aj_s - src_x * cj_s) + g_d * (dst_y * aj_d - dst_x * cj_d)
    gw_j = g_s * (src_x * ci_s - src_y * ai_s) + g_d * (dst_x * ci_d - dst_y * ai_d)

    base_i = grad_pos_ptr + (b * num_vert + vi) * 4
    base_j = grad_pos_ptr + (b * num_vert + vj) * 4
    tl.atomic_add(base_i, gx_i, mask=mask)
    tl.atomic_add(base_i + 1, gy_i, mask=mask)
    tl.atomic_add(base_i + 3, gw_i, mask=mask)
    tl.atomic_add(base_j, gx_j, mask=mask)
    tl.atomic_add(base_j + 1, gy_j, mask=mask)
    tl.atomic_add(base_j + 3, gw_j, mask=mask)


@triton.jit
def _backward_kernel(color_ptr, rast_ptr, pos_ptr, tri_ptr, sil_ptr, px_ptr, py_ptr,
                     grad_out_ptr, grad_color_ptr, grad_pos_ptr, boost,
                     num_pixels, height, width, num_vert, num_tri, channels,
                     BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):  # fmt: skip
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    b = p // (height * width)
    row = (p // width) % height
    col = p % width
    for direction in tl.static_range(2):
        q, active, use_p, t, blend_p, blend_q, owner, edge, x_p, y_p, x_q, y_q = _pair(
            pos_ptr, tri_ptr, sil_ptr, rast_ptr, px_ptr, py_ptr, p, b, row, col, valid,
            height, width, num_vert, num_tri, direction == 0,
        )  # fmt: skip
        below = active & (t < 0.5)
        grad_t = tl.zeros((BLOCK_P,), dtype=t.dtype)
        for c0 in range(0, channels, BLOCK_C):
            c = c0 + tl.arange(0, BLOCK_C)[None, :]
            m = active[:, None] & (c < channels)
            c_p = tl.load(color_ptr + p[:, None] * channels + c, mask=m, other=0.0)
            c_q = tl.load(color_ptr + q[:, None] * channels + c, mask=m, other=0.0)
            g_p = tl.load(grad_out_ptr + p[:, None] * channels + c, mask=m, other=0.0)
            g_q = tl.load(grad_out_ptr + q[:, None] * channels + c, mask=m, other=0.0)
            bp, bq = blend_p[:, None], blend_q[:, None]
            tl.atomic_add(grad_color_ptr + p[:, None] * channels + c, bq * g_q - bp * g_p, mask=m)
            tl.atomic_add(grad_color_ptr + q[:, None] * channels + c, bp * g_p - bq * g_q, mask=m)
            # d/dt of (0.5 - t)(c_q - c_p) below the midpoint and (t - 0.5)(c_p - c_q) above.
            d_t = tl.where(below[:, None], -g_p * (c_q - c_p), g_q * (c_p - c_q))
            grad_t += tl.sum(tl.where(m, d_t, 0.0), axis=1)

        grad_t = grad_t * boost
        # When the edge belongs to q's triangle, t was measured from q, so t_p = 1 - t_q.
        src_x = tl.where(use_p, x_p, x_q)
        src_y = tl.where(use_p, y_p, y_q)
        dst_x = tl.where(use_p, x_q, x_p)
        dst_y = tl.where(use_p, y_q, y_p)
        grad_owner_t = tl.where(use_p, grad_t, -grad_t)
        _add_edge_gradient(grad_pos_ptr, pos_ptr, tri_ptr, b, tl.maximum(owner, 0), edge,
                           grad_owner_t, src_x, src_y, dst_x, dst_y, num_vert,
                           active & (grad_t != 0))  # fmt: skip


class Antialias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, color: Tensor, rast: Tensor, pos: Tensor, tri: Tensor, boost: float):
        color = color.contiguous()
        rast = rast.detach().contiguous()
        pos_c = pos.detach().contiguous()
        tri = tri.to(torch.int32).contiguous()
        with torch.no_grad():
            silhouette = _silhouette_edges(pos_c, tri.long()).to(torch.int8).contiguous()
        px, py = pixel_centres(rast.shape[1], rast.shape[2], pos_c)
        out = color.clone()
        args = _launch_args(rast, pos_c, tri, color)
        with torch.cuda.device(color.device):
            _forward_kernel[args["grid"]](
                color, rast, pos_c, tri, silhouette, px, py, out, *args["sizes"],
                BLOCK_P=BLOCK_PIXELS, BLOCK_C=BLOCK_CHANNELS,
            )  # fmt: skip
        ctx.save_for_backward(color, rast, pos_c, tri, silhouette, px, py)
        ctx.boost = boost
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        color, rast, pos, tri, silhouette, px, py = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_color = grad_out.clone()
        grad_pos = torch.zeros_like(pos)
        args = _launch_args(rast, pos, tri, color)
        with torch.cuda.device(color.device):
            _backward_kernel[args["grid"]](
                color, rast, pos, tri, silhouette, px, py, grad_out, grad_color, grad_pos,
                ctx.boost, *args["sizes"], BLOCK_P=BLOCK_PIXELS, BLOCK_C=BLOCK_CHANNELS,
            )  # fmt: skip
        return grad_color, None, grad_pos, None, None


def antialias(color: Tensor, rast: Tensor, pos: Tensor, tri: Tensor, boost: float) -> Tensor:
    if tri.shape[0] == 0:
        return color
    return Antialias.apply(color, rast, pos, tri, boost)


def _launch_args(rast: Tensor, pos: Tensor, tri: Tensor, color: Tensor) -> dict:
    batch, height, width, _ = rast.shape
    num_pixels = batch * height * width
    return {
        "grid": (triton.cdiv(num_pixels, BLOCK_PIXELS),),
        "sizes": (num_pixels, height, width, pos.shape[1], tri.shape[0], color.shape[-1]),
    }
