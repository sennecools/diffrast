"""Triton kernels for texture.

The mip chain is built with PyTorch ops (as in the reference backend) and packed into one buffer of
shape ``[tex_batch, texels, channels]``, so autograd carries gradients from every level back to the
base texture. The kernels sample that buffer: each program handles a block of pixels and loops
over channels in chunks. The backward pass scatters texel gradients with atomic adds and sums the
gradients of ``uv`` and the level of detail per pixel in registers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from diffrast.backends.reference import _mip_chain

BLOCK_PIXELS = 64
# Rounds of in-register reduction before the remaining texel gradients go out as atomics.
SCATTER_ROUNDS = 2
FILTER_MODES = ("nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear")
BOUNDARY_MODES = ("wrap", "clamp", "zero")


@triton.jit
def _texel_index(i, size, BOUNDARY: tl.constexpr):
    """Index into a dimension of ``size`` texels, and whether the texel exists (``zero`` mode)."""
    if BOUNDARY == 0:
        j = i % size
        j = tl.where(j < 0, j + size, j)
        return j, j == j
    elif BOUNDARY == 1:
        return tl.minimum(tl.maximum(i, 0), size - 1), i == i
    else:
        return tl.minimum(tl.maximum(i, 0), size - 1), (i >= 0) & (i < size)


@triton.jit
def _corners(u, v, height, width, BOUNDARY: tl.constexpr):
    """Bilinear footprint: blend factors, the four texel indices and their validity."""
    x = u * width.to(u.dtype) - 0.5
    y = v * height.to(u.dtype) - 0.5
    x0 = tl.math.floor(x)
    y0 = tl.math.floor(y)
    fx, fy = x - x0, y - y0
    col0, col_ok0 = _texel_index(x0.to(tl.int32), width, BOUNDARY)
    col1, col_ok1 = _texel_index(x0.to(tl.int32) + 1, width, BOUNDARY)
    row0, row_ok0 = _texel_index(y0.to(tl.int32), height, BOUNDARY)
    row1, row_ok1 = _texel_index(y0.to(tl.int32) + 1, height, BOUNDARY)
    i00, i01 = row0 * width + col0, row0 * width + col1
    i10, i11 = row1 * width + col0, row1 * width + col1
    ok00, ok01 = row_ok0 & col_ok0, row_ok0 & col_ok1
    ok10, ok11 = row_ok1 & col_ok0, row_ok1 & col_ok1
    return fx, fy, i00, i01, i10, i11, ok00, ok01, ok10, ok11


@triton.jit
def _level_of_detail(da_ptr, p, valid, height, width):
    du_dx = tl.load(da_ptr + p * 4, mask=valid, other=0.0)
    du_dy = tl.load(da_ptr + p * 4 + 1, mask=valid, other=0.0)
    dv_dx = tl.load(da_ptr + p * 4 + 2, mask=valid, other=0.0)
    dv_dy = tl.load(da_ptr + p * 4 + 3, mask=valid, other=0.0)
    w2 = (width * width).to(du_dx.dtype)
    h2 = (height * height).to(du_dx.dtype)
    len_x = du_dx * du_dx * w2 + dv_dx * dv_dx * h2
    len_y = du_dy * du_dy * w2 + dv_dy * dv_dy * h2
    longest = tl.maximum(tl.maximum(len_x, len_y), 1e-30)
    lod = 0.5 * tl.math.log2(longest)
    return lod, len_x, len_y, longest, du_dx, du_dy, dv_dx, dv_dy, w2, h2


@triton.jit
def _levels(da_ptr, bias_ptr, offs_ptr, hs_ptr, ws_ptr, p, valid, num_levels,
            MODE: tl.constexpr, HAS_DA: tl.constexpr, HAS_BIAS: tl.constexpr):  # fmt: skip
    """Per pixel: the two levels to blend, the blend factor, and the unclamped level of detail."""
    base_h = tl.load(hs_ptr)
    base_w = tl.load(ws_ptr)
    if MODE >= 2:
        lod_raw = tl.zeros(p.shape, dtype=tl.float32)
        if HAS_DA:
            lod_raw += _level_of_detail(da_ptr, p, valid, base_h, base_w)[0]
        if HAS_BIAS:
            lod_raw += tl.load(bias_ptr + p, mask=valid, other=0.0)
        lod = tl.minimum(tl.maximum(lod_raw, 0.0), (num_levels - 1) * 1.0)
        if MODE == 2:
            lower = tl.math.floor(lod + 0.5)
            frac = lod * 0.0
        else:
            lower = tl.math.floor(lod)
            frac = lod - lower
        lo = lower.to(tl.int32)
        hi = tl.minimum(lo + 1, num_levels - 1)
    else:
        lod_raw = tl.zeros(p.shape, dtype=tl.float32)
        frac = lod_raw
        lo = tl.zeros(p.shape, dtype=tl.int32)
        hi = lo
    return lo, hi, frac, lod_raw


@triton.jit
def _forward_kernel(
    flat_ptr,
    offs_ptr,
    hs_ptr,
    ws_ptr,
    uv_ptr,
    da_ptr,
    bias_ptr,
    out_ptr,
    num_pixels,
    pixels_per_image,
    tex_batch_stride,
    channels,
    num_levels,
    MODE: tl.constexpr,
    BOUNDARY: tl.constexpr,
    HAS_DA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    u = tl.load(uv_ptr + p * 2, mask=valid, other=0.0)
    v = tl.load(uv_ptr + p * 2 + 1, mask=valid, other=0.0)
    image = (p // pixels_per_image) * tex_batch_stride
    lo, hi, frac, _ = _levels(da_ptr, bias_ptr, offs_ptr, hs_ptr, ws_ptr, p, valid, num_levels,
                              MODE, HAS_DA, HAS_BIAS)  # fmt: skip

    h_lo, w_lo = tl.load(hs_ptr + lo), tl.load(ws_ptr + lo)
    base_lo = image + tl.load(offs_ptr + lo)
    if MODE == 0:
        col, col_ok = _texel_index(tl.math.floor(u * w_lo.to(u.dtype)).to(tl.int32), w_lo, BOUNDARY)
        row, row_ok = _texel_index(tl.math.floor(v * h_lo.to(u.dtype)).to(tl.int32), h_lo, BOUNDARY)
        texel = (base_lo + row * w_lo + col)[:, None]
        ok = (valid & col_ok & row_ok)[:, None]
    else:
        fx, fy, i00, i01, i10, i11, ok00, ok01, ok10, ok11 = _corners(u, v, h_lo, w_lo, BOUNDARY)
        fx, fy = fx[:, None], fy[:, None]
    if MODE == 3:
        h_hi, w_hi = tl.load(hs_ptr + hi), tl.load(ws_ptr + hi)
        base_hi = image + tl.load(offs_ptr + hi)
        gx, gy, j00, j01, j10, j11, k00, k01, k10, k11 = _corners(u, v, h_hi, w_hi, BOUNDARY)
        gx, gy = gx[:, None], gy[:, None]

    for c0 in range(0, channels, BLOCK_C):
        c = c0 + tl.arange(0, BLOCK_C)[None, :]
        cm = valid[:, None] & (c < channels)
        if MODE == 0:
            value = tl.load(flat_ptr + texel * channels + c, mask=cm & ok, other=0.0)
        else:
            value = _bilinear(flat_ptr, base_lo, i00, i01, i10, i11, ok00, ok01, ok10, ok11,
                              fx, fy, c, channels, cm)  # fmt: skip
            if MODE == 3:
                upper = _bilinear(flat_ptr, base_hi, j00, j01, j10, j11, k00, k01, k10, k11,
                                  gx, gy, c, channels, cm)  # fmt: skip
                value = value * (1 - frac[:, None]) + upper * frac[:, None]
        tl.store(out_ptr + p[:, None] * channels + c, value, mask=cm)


@triton.jit
def _bilinear(flat_ptr, base, i00, i01, i10, i11, ok00, ok01, ok10, ok11, fx, fy, c, channels, cm):
    t00, t01, t10, t11 = _load_corners(
        flat_ptr, base, i00, i01, i10, i11, ok00, ok01, ok10, ok11, c, channels, cm
    )
    return (1 - fy) * ((1 - fx) * t00 + fx * t01) + fy * ((1 - fx) * t10 + fx * t11)


@triton.jit
def _load_corners(flat_ptr, base, i00, i01, i10, i11, ok00, ok01, ok10, ok11, c, channels, cm):
    t00 = tl.load(
        flat_ptr + (base + i00)[:, None] * channels + c, mask=cm & ok00[:, None], other=0.0
    )
    t01 = tl.load(
        flat_ptr + (base + i01)[:, None] * channels + c, mask=cm & ok01[:, None], other=0.0
    )
    t10 = tl.load(
        flat_ptr + (base + i10)[:, None] * channels + c, mask=cm & ok10[:, None], other=0.0
    )
    t11 = tl.load(
        flat_ptr + (base + i11)[:, None] * channels + c, mask=cm & ok11[:, None], other=0.0
    )
    return t00, t01, t10, t11


@triton.jit
def _scatter(grad_ptr, texel, ok, values, cc, channels, ROUNDS: tl.constexpr):
    """Atomically add ``values`` (pixels x channels) to the rows ``texel`` of ``grad_ptr``.

    Pixels in a block often share a texel (background pixels with equal ``uv``, or a magnified
    texture), and atomics on one address serialize. So for a few rounds, the pixels sharing the
    lowest remaining address are summed in registers and written with a single atomic. Zero
    values, such as gradients of masked-out background, are not written at all.
    """
    empty = 0x7FFFFFFFFFFFFFFF
    c_ok = cc < channels
    key = tl.where(ok, texel, empty)
    for _ in tl.static_range(ROUNDS):
        ref = tl.min(key, axis=0)
        group = key == ref
        total = tl.sum(tl.where(group[:, None], values, 0.0), axis=0)
        tl.atomic_add(
            grad_ptr + ref * channels + cc, total, mask=c_ok & (ref != empty) & (total != 0)
        )
        key = tl.where(group, empty, key)
    rest = (key != empty)[:, None] & c_ok[None, :] & (values != 0)
    tl.atomic_add(grad_ptr + texel[:, None] * channels + cc[None, :], values, mask=rest)


@triton.jit
def _scatter_bilinear(grad_ptr, base, i00, i01, i10, i11, ok00, ok01, ok10, ok11,
                      fx, fy, g, cc, channels, valid, ROUNDS: tl.constexpr):  # fmt: skip
    _scatter(grad_ptr, base + i00, valid & ok00, g * (1 - fx) * (1 - fy), cc, channels, ROUNDS)
    _scatter(grad_ptr, base + i01, valid & ok01, g * fx * (1 - fy), cc, channels, ROUNDS)
    _scatter(grad_ptr, base + i10, valid & ok10, g * (1 - fx) * fy, cc, channels, ROUNDS)
    _scatter(grad_ptr, base + i11, valid & ok11, g * fx * fy, cc, channels, ROUNDS)


@triton.jit
def _backward_kernel(
    flat_ptr,
    offs_ptr,
    hs_ptr,
    ws_ptr,
    uv_ptr,
    da_ptr,
    bias_ptr,
    grad_out_ptr,
    grad_flat_ptr,
    grad_uv_ptr,
    grad_da_ptr,
    grad_bias_ptr,
    num_pixels,
    pixels_per_image,
    tex_batch_stride,
    channels,
    num_levels,
    MODE: tl.constexpr,
    BOUNDARY: tl.constexpr,
    HAS_DA: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    u = tl.load(uv_ptr + p * 2, mask=valid, other=0.0)
    v = tl.load(uv_ptr + p * 2 + 1, mask=valid, other=0.0)
    image = (p // pixels_per_image) * tex_batch_stride
    lo, hi, frac, lod_raw = _levels(da_ptr, bias_ptr, offs_ptr, hs_ptr, ws_ptr, p, valid,
                                    num_levels, MODE, HAS_DA, HAS_BIAS)  # fmt: skip

    h_lo, w_lo = tl.load(hs_ptr + lo), tl.load(ws_ptr + lo)
    base_lo = image + tl.load(offs_ptr + lo)
    weight_lo = (1 - frac)[:, None]
    if MODE == 0:
        col, col_ok = _texel_index(tl.math.floor(u * w_lo.to(u.dtype)).to(tl.int32), w_lo, BOUNDARY)
        row, row_ok = _texel_index(tl.math.floor(v * h_lo.to(u.dtype)).to(tl.int32), h_lo, BOUNDARY)
        texel = base_lo + row * w_lo + col
        ok = valid & col_ok & row_ok
    else:
        fx, fy, i00, i01, i10, i11, ok00, ok01, ok10, ok11 = _corners(u, v, h_lo, w_lo, BOUNDARY)
        fx, fy = fx[:, None], fy[:, None]
    if MODE == 3:
        h_hi, w_hi = tl.load(hs_ptr + hi), tl.load(ws_ptr + hi)
        base_hi = image + tl.load(offs_ptr + hi)
        gx, gy, j00, j01, j10, j11, k00, k01, k10, k11 = _corners(u, v, h_hi, w_hi, BOUNDARY)
        gx, gy = gx[:, None], gy[:, None]

    grad_u = tl.zeros((BLOCK_P,), dtype=u.dtype)
    grad_v = tl.zeros((BLOCK_P,), dtype=u.dtype)
    grad_frac = tl.zeros((BLOCK_P,), dtype=u.dtype)
    for c0 in range(0, channels, BLOCK_C):
        cc = c0 + tl.arange(0, BLOCK_C)
        c = cc[None, :]
        cm = valid[:, None] & (c < channels)
        g = tl.load(grad_out_ptr + p[:, None] * channels + c, mask=cm, other=0.0)
        if MODE == 0:
            _scatter(grad_flat_ptr, texel, ok, g, cc, channels, ROUNDS)
        else:
            t00, t01, t10, t11 = _load_corners(
                flat_ptr, base_lo, i00, i01, i10, i11, ok00, ok01, ok10, ok11, c, channels, cm
            )
            g_lo = g * weight_lo
            _scatter_bilinear(grad_flat_ptr, base_lo, i00, i01, i10, i11, ok00, ok01, ok10, ok11,
                              fx, fy, g_lo, cc, channels, valid, ROUNDS)  # fmt: skip
            d_fx = (1 - fy) * (t01 - t00) + fy * (t11 - t10)
            d_fy = (1 - fx) * (t10 - t00) + fx * (t11 - t01)
            grad_u += tl.sum(g_lo * d_fx, axis=1) * w_lo.to(u.dtype)
            grad_v += tl.sum(g_lo * d_fy, axis=1) * h_lo.to(u.dtype)
            if MODE == 3:
                s00, s01, s10, s11 = _load_corners(
                    flat_ptr, base_hi, j00, j01, j10, j11, k00, k01, k10, k11, c, channels, cm
                )
                g_hi = g * frac[:, None]
                _scatter_bilinear(grad_flat_ptr, base_hi, j00, j01, j10, j11, k00, k01, k10, k11,
                                  gx, gy, g_hi, cc, channels, valid, ROUNDS)  # fmt: skip
                e_fx = (1 - gy) * (s01 - s00) + gy * (s11 - s10)
                e_fy = (1 - gx) * (s10 - s00) + gx * (s11 - s01)
                grad_u += tl.sum(g_hi * e_fx, axis=1) * w_hi.to(u.dtype)
                grad_v += tl.sum(g_hi * e_fy, axis=1) * h_hi.to(u.dtype)
                lower = (1 - fy) * ((1 - fx) * t00 + fx * t01) + fy * ((1 - fx) * t10 + fx * t11)
                upper = (1 - gy) * ((1 - gx) * s00 + gx * s01) + gy * ((1 - gx) * s10 + gx * s11)
                grad_frac += tl.sum(g * (upper - lower), axis=1)

    tl.store(grad_uv_ptr + p * 2, grad_u, mask=valid)
    tl.store(grad_uv_ptr + p * 2 + 1, grad_v, mask=valid)

    if MODE == 3:
        # The level of detail is clamped to the available levels.
        in_range = (lod_raw >= 0) & (lod_raw <= (num_levels - 1) * 1.0)
        grad_lod = tl.where(in_range, grad_frac, 0.0)
        if HAS_BIAS:
            tl.store(grad_bias_ptr + p, grad_lod, mask=valid)
    if (MODE == 3) and HAS_DA:
        base_h = tl.load(hs_ptr)
        base_w = tl.load(ws_ptr)
        _, len_x, len_y, longest, du_dx, du_dy, dv_dx, dv_dy, w2, h2 = _level_of_detail(
            da_ptr, p, valid, base_h, base_w
        )
        # lod = log2(longest) / 2, with the footprint clamped from below.
        grad_longest = tl.where(
            longest > 1e-30, grad_lod * 0.5 / (longest * 0.6931471805599453), 0.0
        )
        share_x = tl.where(len_x > len_y, 1.0, tl.where(len_x == len_y, 0.5, 0.0))
        grad_x = grad_longest * share_x
        grad_y = grad_longest * (1 - share_x)
        tl.store(grad_da_ptr + p * 4, grad_x * 2 * du_dx * w2, mask=valid)
        tl.store(grad_da_ptr + p * 4 + 1, grad_y * 2 * du_dy * w2, mask=valid)
        tl.store(grad_da_ptr + p * 4 + 2, grad_x * 2 * dv_dx * h2, mask=valid)
        tl.store(grad_da_ptr + p * 4 + 3, grad_y * 2 * dv_dy * h2, mask=valid)


class Texture(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat, uv, uv_da, bias, offsets, heights, widths, mode, boundary):
        batch, height, width, _ = uv.shape
        channels = flat.shape[-1]
        uv = uv.contiguous()
        # Unused pointer arguments get a placeholder; the code reading them is compiled out.
        has_da = mode >= 2 and uv_da is not None
        has_bias = mode >= 2 and bias is not None
        da = uv_da.contiguous() if has_da else uv
        bias = bias.contiguous() if has_bias else uv
        out = uv.new_empty(batch, height, width, channels)
        args = _launch_args(flat, uv, offsets)
        with torch.cuda.device(uv.device):
            _forward_kernel[args["grid"]](
                flat, offsets, heights, widths, uv, da, bias, out,
                args["num_pixels"], args["pixels_per_image"], args["batch_stride"],
                channels, offsets.shape[0],
                MODE=mode, BOUNDARY=boundary, HAS_DA=has_da, HAS_BIAS=has_bias,
                BLOCK_P=BLOCK_PIXELS, BLOCK_C=args["block_c"],
            )  # fmt: skip
        ctx.save_for_backward(flat, uv, da, bias, offsets, heights, widths)
        ctx.mode, ctx.boundary = mode, boundary
        ctx.has_da, ctx.has_bias = has_da, has_bias
        return out

    @staticmethod
    def backward(ctx, grad_out):
        flat, uv, da, bias, offsets, heights, widths = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_flat = torch.zeros_like(flat)
        grad_uv = torch.empty_like(uv)
        # Only trilinear filtering depends smoothly on the level of detail.
        smooth = ctx.mode == 3
        grad_da = torch.zeros_like(da) if smooth and ctx.has_da else grad_uv
        grad_bias = torch.zeros_like(bias) if smooth and ctx.has_bias else grad_uv
        args = _launch_args(flat, uv, offsets)
        with torch.cuda.device(uv.device):
            _backward_kernel[args["grid"]](
                flat, offsets, heights, widths, uv, da, bias, grad_out, grad_flat, grad_uv,
                grad_da, grad_bias, args["num_pixels"], args["pixels_per_image"],
                args["batch_stride"], flat.shape[-1], offsets.shape[0],
                MODE=ctx.mode, BOUNDARY=ctx.boundary, HAS_DA=ctx.has_da, HAS_BIAS=ctx.has_bias,
                BLOCK_P=BLOCK_PIXELS, BLOCK_C=args["block_c"], ROUNDS=SCATTER_ROUNDS,
            )  # fmt: skip
        grad_uv_da = grad_da if smooth and ctx.has_da else None
        grad_mip_bias = grad_bias if smooth and ctx.has_bias else None
        return grad_flat, grad_uv, grad_uv_da, grad_mip_bias, None, None, None, None, None


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None,
    filter_mode: str,
    boundary_mode: str,
    max_mip_level: int | None,
    mip_level_bias: Tensor | None = None,
) -> Tensor:
    mode = FILTER_MODES.index(filter_mode)
    levels = _mip_chain(tex, max_mip_level) if mode >= 2 else [tex]
    channels = tex.shape[-1]
    flat = torch.cat([level.reshape(level.shape[0], -1, channels) for level in levels], 1)
    sizes = torch.tensor([level.shape[1:3] for level in levels], dtype=torch.int32)
    counts = sizes[:, 0] * sizes[:, 1]
    offsets = (counts.cumsum(0) - counts).to(torch.int32)
    device = uv.device
    return Texture.apply(
        flat.contiguous(),
        uv,
        uv_da,
        mip_level_bias,
        offsets.to(device),
        sizes[:, 0].contiguous().to(device),
        sizes[:, 1].contiguous().to(device),
        mode,
        BOUNDARY_MODES.index(boundary_mode),
    )


def _launch_args(flat: Tensor, uv: Tensor, offsets: Tensor) -> dict:
    batch, height, width, _ = uv.shape
    num_pixels = batch * height * width
    return {
        "grid": (triton.cdiv(num_pixels, BLOCK_PIXELS),),
        "num_pixels": num_pixels,
        "pixels_per_image": height * width,
        "batch_stride": 0 if flat.shape[0] == 1 else flat.shape[1],
        "block_c": min(triton.next_power_of_2(flat.shape[-1]), 16),
    }
