"""Triton kernels for interpolate.

Each program handles a block of pixels and loops over the attribute channels in chunks. The
backward pass accumulates the gradients of ``rast`` and ``rast_db`` per pixel in registers and
scatters the gradients of ``attr`` to the vertices with atomic adds.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK_PIXELS = 64


@triton.jit
def _load_pixel(rast_ptr, tri_ptr, p, valid, pixels_per_image, attr_batch_stride):
    u = tl.load(rast_ptr + p * 4, mask=valid, other=0.0)
    v = tl.load(rast_ptr + p * 4 + 1, mask=valid, other=0.0)
    tri_id = (tl.load(rast_ptr + p * 4 + 3, mask=valid, other=0.0) + 0.5).to(tl.int64) - 1
    covered = valid & (tri_id >= 0)
    i0 = tl.load(tri_ptr + tri_id * 3, mask=covered, other=0).to(tl.int64)
    i1 = tl.load(tri_ptr + tri_id * 3 + 1, mask=covered, other=0).to(tl.int64)
    i2 = tl.load(tri_ptr + tri_id * 3 + 2, mask=covered, other=0).to(tl.int64)
    base = (p // pixels_per_image) * attr_batch_stride
    return u, v, covered, base, i0, i1, i2


@triton.jit
def _forward_kernel(
    attr_ptr,
    rast_ptr,
    db_ptr,
    tri_ptr,
    diff_ptr,
    out_ptr,
    out_da_ptr,
    num_pixels,
    pixels_per_image,
    attr_batch_stride,
    channels,
    num_diff,
    HAS_DA: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    u, v, covered, base, i0, i1, i2 = _load_pixel(
        rast_ptr, tri_ptr, p, valid, pixels_per_image, attr_batch_stride
    )
    u, v = u[:, None], v[:, None]
    row0 = (base + i0 * channels)[:, None]
    row1 = (base + i1 * channels)[:, None]
    row2 = (base + i2 * channels)[:, None]

    for c0 in range(0, channels, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)[None, :]
        mask = covered[:, None] & (c < channels)
        a0 = tl.load(attr_ptr + row0 + c, mask=mask, other=0.0)
        a1 = tl.load(attr_ptr + row1 + c, mask=mask, other=0.0)
        a2 = tl.load(attr_ptr + row2 + c, mask=mask, other=0.0)
        value = u * a0 + v * a1 + (1 - u - v) * a2
        tl.store(out_ptr + p[:, None] * channels + c, value, mask=valid[:, None] & (c < channels))

    if HAS_DA:
        du_dx = tl.load(db_ptr + p * 4, mask=valid, other=0.0)[:, None]
        du_dy = tl.load(db_ptr + p * 4 + 1, mask=valid, other=0.0)[:, None]
        dv_dx = tl.load(db_ptr + p * 4 + 2, mask=valid, other=0.0)[:, None]
        dv_dy = tl.load(db_ptr + p * 4 + 3, mask=valid, other=0.0)[:, None]
        for k0 in range(0, num_diff, BLOCK_C):
            k = k0 + tl.arange(0, BLOCK_C)[None, :]
            c = tl.load(diff_ptr + k, mask=k < num_diff, other=0)
            mask = covered[:, None] & (k < num_diff)
            a0 = tl.load(attr_ptr + row0 + c, mask=mask, other=0.0)
            a1 = tl.load(attr_ptr + row1 + c, mask=mask, other=0.0)
            a2 = tl.load(attr_ptr + row2 + c, mask=mask, other=0.0)
            da_du, da_dv = a0 - a2, a1 - a2
            out = out_da_ptr + p[:, None] * (2 * num_diff) + 2 * k
            store_mask = valid[:, None] & (k < num_diff)
            tl.store(out, du_dx * da_du + dv_dx * da_dv, mask=store_mask)
            tl.store(out + 1, du_dy * da_du + dv_dy * da_dv, mask=store_mask)


@triton.jit
def _backward_kernel(
    attr_ptr,
    rast_ptr,
    db_ptr,
    tri_ptr,
    diff_ptr,
    grad_out_ptr,
    grad_out_da_ptr,
    grad_attr_ptr,
    grad_rast_ptr,
    grad_db_ptr,
    num_pixels,
    pixels_per_image,
    attr_batch_stride,
    channels,
    num_diff,
    HAS_DA: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    u, v, covered, base, i0, i1, i2 = _load_pixel(
        rast_ptr, tri_ptr, p, valid, pixels_per_image, attr_batch_stride
    )
    row0 = (base + i0 * channels)[:, None]
    row1 = (base + i1 * channels)[:, None]
    row2 = (base + i2 * channels)[:, None]

    grad_u = tl.zeros((BLOCK_P,), dtype=u.dtype)
    grad_v = tl.zeros((BLOCK_P,), dtype=u.dtype)
    for c0 in range(0, channels, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)[None, :]
        mask = covered[:, None] & (c < channels)
        g = tl.load(grad_out_ptr + p[:, None] * channels + c, mask=mask, other=0.0)
        a0 = tl.load(attr_ptr + row0 + c, mask=mask, other=0.0)
        a1 = tl.load(attr_ptr + row1 + c, mask=mask, other=0.0)
        a2 = tl.load(attr_ptr + row2 + c, mask=mask, other=0.0)
        grad_u += tl.sum(g * (a0 - a2), axis=1)
        grad_v += tl.sum(g * (a1 - a2), axis=1)
        tl.atomic_add(grad_attr_ptr + row0 + c, g * u[:, None], mask=mask)
        tl.atomic_add(grad_attr_ptr + row1 + c, g * v[:, None], mask=mask)
        tl.atomic_add(grad_attr_ptr + row2 + c, g * (1 - u - v)[:, None], mask=mask)

    tl.store(grad_rast_ptr + p * 4, grad_u, mask=valid)
    tl.store(grad_rast_ptr + p * 4 + 1, grad_v, mask=valid)
    tl.store(grad_rast_ptr + p * 4 + 2, tl.zeros_like(grad_u), mask=valid)
    tl.store(grad_rast_ptr + p * 4 + 3, tl.zeros_like(grad_u), mask=valid)

    if HAS_DA:
        du_dx = tl.load(db_ptr + p * 4, mask=valid, other=0.0)[:, None]
        du_dy = tl.load(db_ptr + p * 4 + 1, mask=valid, other=0.0)[:, None]
        dv_dx = tl.load(db_ptr + p * 4 + 2, mask=valid, other=0.0)[:, None]
        dv_dy = tl.load(db_ptr + p * 4 + 3, mask=valid, other=0.0)[:, None]
        grad_du_dx = tl.zeros((BLOCK_P,), dtype=u.dtype)
        grad_du_dy = tl.zeros((BLOCK_P,), dtype=u.dtype)
        grad_dv_dx = tl.zeros((BLOCK_P,), dtype=u.dtype)
        grad_dv_dy = tl.zeros((BLOCK_P,), dtype=u.dtype)
        for k0 in range(0, num_diff, BLOCK_C):
            k = k0 + tl.arange(0, BLOCK_C)[None, :]
            c = tl.load(diff_ptr + k, mask=k < num_diff, other=0)
            mask = covered[:, None] & (k < num_diff)
            g_ptr = grad_out_da_ptr + p[:, None] * (2 * num_diff) + 2 * k
            gx = tl.load(g_ptr, mask=mask, other=0.0)
            gy = tl.load(g_ptr + 1, mask=mask, other=0.0)
            a0 = tl.load(attr_ptr + row0 + c, mask=mask, other=0.0)
            a1 = tl.load(attr_ptr + row1 + c, mask=mask, other=0.0)
            a2 = tl.load(attr_ptr + row2 + c, mask=mask, other=0.0)
            da_du, da_dv = a0 - a2, a1 - a2
            grad_du_dx += tl.sum(gx * da_du, axis=1)
            grad_du_dy += tl.sum(gy * da_du, axis=1)
            grad_dv_dx += tl.sum(gx * da_dv, axis=1)
            grad_dv_dy += tl.sum(gy * da_dv, axis=1)
            g_a0 = gx * du_dx + gy * du_dy
            g_a1 = gx * dv_dx + gy * dv_dy
            tl.atomic_add(grad_attr_ptr + row0 + c, g_a0, mask=mask)
            tl.atomic_add(grad_attr_ptr + row1 + c, g_a1, mask=mask)
            tl.atomic_add(grad_attr_ptr + row2 + c, -(g_a0 + g_a1), mask=mask)

        tl.store(grad_db_ptr + p * 4, grad_du_dx, mask=valid)
        tl.store(grad_db_ptr + p * 4 + 1, grad_du_dy, mask=valid)
        tl.store(grad_db_ptr + p * 4 + 2, grad_dv_dx, mask=valid)
        tl.store(grad_db_ptr + p * 4 + 3, grad_dv_dy, mask=valid)


class Interpolate(torch.autograd.Function):
    """Autograd wrapper around the kernels. ``out_da`` has zero channels when none are requested."""

    @staticmethod
    def forward(ctx, attr: Tensor, rast: Tensor, rast_db: Tensor | None, tri: Tensor, diff: Tensor):
        batch, height, width, _ = rast.shape
        attr_3d = attr if attr.ndim == 3 else attr[None]
        attr_3d = attr_3d.contiguous()
        channels = attr_3d.shape[2]
        num_diff = diff.shape[0]
        rast = rast.contiguous()
        tri = tri.to(torch.int32).contiguous()
        has_da = num_diff > 0
        # Unused pointer arguments get a placeholder; the code reading them is compiled out.
        db = rast_db.contiguous() if has_da else rast
        diff = diff if has_da else tri

        out = attr.new_empty(batch, height, width, channels)
        out_da = attr.new_empty(batch, height, width, 2 * num_diff)
        args = _launch_args(rast, attr_3d, channels, num_diff)
        with torch.cuda.device(rast.device):
            _forward_kernel[args["grid"]](
                attr_3d, rast, db, tri, diff, out, out_da,
                args["num_pixels"], args["pixels_per_image"], args["batch_stride"],
                channels, num_diff,
                HAS_DA=has_da, BLOCK_P=BLOCK_PIXELS, BLOCK_C=args["block_c"],
            )  # fmt: skip
        ctx.save_for_backward(attr_3d, rast, db, tri, diff)
        ctx.attr_shape = attr.shape
        ctx.has_da = has_da
        ctx.rast_db_given = rast_db is not None
        return out, out_da

    @staticmethod
    def backward(ctx, grad_out: Tensor | None, grad_out_da: Tensor | None):
        attr_3d, rast, db, tri, diff = ctx.saved_tensors
        channels = attr_3d.shape[2]
        num_diff = diff.shape[0] if ctx.has_da else 0
        has_da = ctx.has_da and grad_out_da is not None
        if grad_out is None:
            grad_out = rast.new_zeros(*rast.shape[:3], channels)
        grad_out = grad_out.contiguous()
        grad_out_da = grad_out_da.contiguous() if has_da else grad_out

        grad_attr = torch.zeros_like(attr_3d)
        grad_rast = torch.empty_like(rast)
        grad_db = torch.zeros_like(rast) if has_da else grad_rast
        args = _launch_args(rast, attr_3d, channels, num_diff)
        with torch.cuda.device(rast.device):
            _backward_kernel[args["grid"]](
                attr_3d, rast, db, tri, diff, grad_out, grad_out_da,
                grad_attr, grad_rast, grad_db,
                args["num_pixels"], args["pixels_per_image"], args["batch_stride"],
                channels, num_diff,
                HAS_DA=has_da, BLOCK_P=BLOCK_PIXELS, BLOCK_C=args["block_c"],
            )  # fmt: skip

        grad_db_out = None
        if ctx.rast_db_given:
            grad_db_out = grad_db if has_da else torch.zeros_like(rast)
        return grad_attr.view(ctx.attr_shape), grad_rast, grad_db_out, None, None


def _launch_args(rast: Tensor, attr_3d: Tensor, channels: int, num_diff: int) -> dict:
    batch, height, width, _ = rast.shape
    num_vert = attr_3d.shape[1]
    num_pixels = batch * height * width
    return {
        "grid": (triton.cdiv(num_pixels, BLOCK_PIXELS),),
        "num_pixels": num_pixels,
        "pixels_per_image": height * width,
        "batch_stride": 0 if attr_3d.shape[0] == 1 else num_vert * channels,
        "block_c": min(triton.next_power_of_2(max(channels, num_diff, 1)), 16),
    }
