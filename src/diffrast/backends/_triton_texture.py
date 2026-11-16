"""Triton kernels for texture.

The mip levels are built with PyTorch ops (as in the reference backend), or given by the caller,
and packed into one buffer of shape ``[tex_batch, texels, channels]``, so autograd carries
gradients from every level back to the tensor it came from. The kernels sample that buffer: each
program handles a block of pixels and loops over channels in chunks. The backward pass scatters
texel gradients with atomic adds and sums the gradients of ``uv`` and the level of detail per
pixel in registers.

Coordinates and levels of detail that are out of range or not finite are handled as in the
reference backend, with the same limits, so both give the same result for any input.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl
from torch import Tensor

from diffrast.backends import reference

BLOCK_PIXELS = 64
# Rounds of in-register reduction before the remaining texel gradients go out as atomics.
SCATTER_ROUNDS = 2
# Without fused multiply-adds the kernels round like the PyTorch code of the reference backend.
# Ties in the level of detail, and the texel a far-out coordinate lands in, are then decided the
# same way on both backends, whatever the default of the installed Triton version.
FP_FUSION = False
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
def _texel_coordinate(coord, size, shift, BOUNDARY: tl.constexpr):
    """Continuous texel coordinate ``coord * size - shift``, kept within reach of the texture.

    Mirrors ``reference._texel_coordinate``. Returns the coordinate, whether it was not finite
    (it is then replaced), and its derivative with respect to ``coord * size``.
    """
    limit = size.to(coord.dtype)
    if BOUNDARY == 0:
        # The fractional part of an infinite coordinate is NaN, so NaN is all that can go wrong.
        x = (coord - tl.math.floor(coord)) * limit - shift
        bad = x != x
        slope = tl.where(bad, 0.0, 1.0)
    else:
        x = coord * limit - shift
        # Taken before clamping, which would turn NaN into a bound.
        bad = x != x
        slope = tl.where((x >= -1.0) & (x <= limit), 1.0, 0.0)
        x = tl.minimum(tl.maximum(x, -1.0), limit)
    return tl.where(bad, 0.0, x), bad, slope


@triton.jit
def _nearest(u, v, height, width, BOUNDARY: tl.constexpr):
    """Index and validity of the texel containing ``(u, v)``, and whether it was not finite."""
    x, bad_x, _ = _texel_coordinate(u, width, 0.0, BOUNDARY)
    y, bad_y, _ = _texel_coordinate(v, height, 0.0, BOUNDARY)
    col, col_ok = _texel_index(tl.math.floor(x).to(tl.int32), width, BOUNDARY)
    row, row_ok = _texel_index(tl.math.floor(y).to(tl.int32), height, BOUNDARY)
    return row * width + col, col_ok & row_ok, bad_x | bad_y


@triton.jit
def _corners(u, v, height, width, BOUNDARY: tl.constexpr):
    """Bilinear footprint: blend factors, the four texel indices and their validity.

    Also returns the derivatives of the texel coordinates with respect to ``u * width`` and
    ``v * height``, and whether the coordinates were not finite.
    """
    x, bad_x, slope_x = _texel_coordinate(u, width, 0.5, BOUNDARY)
    y, bad_y, slope_y = _texel_coordinate(v, height, 0.5, BOUNDARY)
    x0 = tl.math.floor(x)
    y0 = tl.math.floor(y)
    fx, fy = x - x0, y - y0
    xi, yi = x0.to(tl.int32), y0.to(tl.int32)
    col0, col_ok0 = _texel_index(xi, width, BOUNDARY)
    col1, col_ok1 = _texel_index(xi + 1, width, BOUNDARY)
    row0, row_ok0 = _texel_index(yi, height, BOUNDARY)
    row1, row_ok1 = _texel_index(yi + 1, height, BOUNDARY)
    i00, i01 = row0 * width + col0, row0 * width + col1
    i10, i11 = row1 * width + col0, row1 * width + col1
    ok00, ok01 = row_ok0 & col_ok0, row_ok0 & col_ok1
    ok10, ok11 = row_ok1 & col_ok0, row_ok1 & col_ok1
    bad = bad_x | bad_y
    return fx, fy, i00, i01, i10, i11, ok00, ok01, ok10, ok11, slope_x, slope_y, bad


@triton.jit
def _footprint(da_ptr, p, valid, size):
    """One component of ``uv_da`` in texels, saturated as in the reference backend.

    Returns it with half the derivative of its square with respect to the component, and whether
    the component was NaN. The size is converted to floating point before it is multiplied: the
    square of a size overflows int32 from 46341 texels on.
    """
    component = tl.load(da_ptr + p, mask=valid, other=0.0)
    scale = size.to(component.dtype)
    raw = component * scale
    texels = tl.minimum(tl.maximum(raw, -1e18), 1e18)
    slope = tl.where((raw >= -1e18) & (raw <= 1e18), texels * scale, 0.0)
    return texels, slope, raw != raw


@triton.jit
def _level_of_detail(da_ptr, p, valid, height, width):
    """Level of detail from ``uv_da``, with the terms its backward pass needs."""
    ux, slope_ux, bad_ux = _footprint(da_ptr, p * 4, valid, width)
    uy, slope_uy, bad_uy = _footprint(da_ptr, p * 4 + 1, valid, width)
    vx, slope_vx, bad_vx = _footprint(da_ptr, p * 4 + 2, valid, height)
    vy, slope_vy, bad_vy = _footprint(da_ptr, p * 4 + 3, valid, height)
    len_x = ux * ux + vx * vx
    len_y = uy * uy + vy * vy
    longest = tl.maximum(tl.maximum(len_x, len_y), 1e-30)
    # The saturation dropped any NaN, which has to reach the output.
    bad = bad_ux | bad_uy | bad_vx | bad_vy
    lod = tl.where(bad, float("nan"), 0.5 * tl.math.log2(longest))
    return lod, len_x, len_y, longest, slope_ux, slope_uy, slope_vx, slope_vy


@triton.jit
def _levels(da_ptr, bias_ptr, offs_ptr, hs_ptr, ws_ptr, p, valid, num_levels,
            MODE: tl.constexpr, HAS_DA: tl.constexpr, HAS_BIAS: tl.constexpr):  # fmt: skip
    """Per pixel: the two levels to blend, the blend factor, and the unclamped level of detail.

    Also returns whether the level of detail was NaN; it is then replaced by zero.
    """
    base_h = tl.load(hs_ptr)
    base_w = tl.load(ws_ptr)
    if MODE >= 2:
        lod_raw = tl.zeros(p.shape, dtype=tl.float32)
        if HAS_DA:
            lod_raw += _level_of_detail(da_ptr, p, valid, base_h, base_w)[0]
        if HAS_BIAS:
            lod_raw += tl.load(bias_ptr + p, mask=valid, other=0.0)
        bad = lod_raw != lod_raw
        lod_raw = tl.where(bad, 0.0, lod_raw)
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
        bad = p != p
    return lo, hi, frac, lod_raw, bad


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
    lo, hi, frac, _, bad = _levels(da_ptr, bias_ptr, offs_ptr, hs_ptr, ws_ptr, p, valid,
                                   num_levels, MODE, HAS_DA, HAS_BIAS)  # fmt: skip

    h_lo, w_lo = tl.load(hs_ptr + lo), tl.load(ws_ptr + lo)
    base_lo = image + tl.load(offs_ptr + lo)
    if MODE == 0:
        index, ok, bad = _nearest(u, v, h_lo, w_lo, BOUNDARY)
        texel = (base_lo + index)[:, None]
        ok = (valid & ok)[:, None]
    else:
        fx, fy, i00, i01, i10, i11, ok00, ok01, ok10, ok11, _, _, bad_lo = _corners(
            u, v, h_lo, w_lo, BOUNDARY
        )
        fx, fy = fx[:, None], fy[:, None]
        bad = bad | bad_lo
    if MODE == 3:
        h_hi, w_hi = tl.load(hs_ptr + hi), tl.load(ws_ptr + hi)
        base_hi = image + tl.load(offs_ptr + hi)
        gx, gy, j00, j01, j10, j11, k00, k01, k10, k11, _, _, bad_hi = _corners(
            u, v, h_hi, w_hi, BOUNDARY
        )
        gx, gy = gx[:, None], gy[:, None]
        bad = bad | bad_hi

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
        value = tl.where(bad[:, None], float("nan"), value)
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
    COORD_GRADS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    """Scatter the texel gradients; with ``COORD_GRADS`` also those of ``uv`` and the level.

    Only the latter read texel values, so without them ``flat_ptr`` is never dereferenced.
    """
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < num_pixels
    u = tl.load(uv_ptr + p * 2, mask=valid, other=0.0)
    v = tl.load(uv_ptr + p * 2 + 1, mask=valid, other=0.0)
    image = (p // pixels_per_image) * tex_batch_stride
    lo, hi, frac, lod_raw, bad = _levels(da_ptr, bias_ptr, offs_ptr, hs_ptr, ws_ptr, p, valid,
                                         num_levels, MODE, HAS_DA, HAS_BIAS)  # fmt: skip

    h_lo, w_lo = tl.load(hs_ptr + lo), tl.load(ws_ptr + lo)
    base_lo = image + tl.load(offs_ptr + lo)
    weight_lo = (1 - frac)[:, None]
    if MODE == 0:
        index, ok, bad = _nearest(u, v, h_lo, w_lo, BOUNDARY)
        texel = base_lo + index
        ok = valid & ok
    else:
        fx, fy, i00, i01, i10, i11, ok00, ok01, ok10, ok11, sx_lo, sy_lo, bad_lo = _corners(
            u, v, h_lo, w_lo, BOUNDARY
        )
        fx, fy = fx[:, None], fy[:, None]
        bad = bad | bad_lo
    if MODE == 3:
        h_hi, w_hi = tl.load(hs_ptr + hi), tl.load(ws_ptr + hi)
        base_hi = image + tl.load(offs_ptr + hi)
        gx, gy, j00, j01, j10, j11, k00, k01, k10, k11, sx_hi, sy_hi, bad_hi = _corners(
            u, v, h_hi, w_hi, BOUNDARY
        )
        gx, gy = gx[:, None], gy[:, None]
        bad = bad | bad_hi

    grad_u = tl.zeros((BLOCK_P,), dtype=u.dtype)
    grad_v = tl.zeros((BLOCK_P,), dtype=u.dtype)
    grad_frac = tl.zeros((BLOCK_P,), dtype=u.dtype)
    for c0 in range(0, channels, BLOCK_C):
        cc = c0 + tl.arange(0, BLOCK_C)
        c = cc[None, :]
        cm = valid[:, None] & (c < channels)
        g = tl.load(grad_out_ptr + p[:, None] * channels + c, mask=cm, other=0.0)
        # A pixel that is NaN in the output passes no gradient.
        g = tl.where(bad[:, None], 0.0, g)
        if MODE == 0:
            _scatter(grad_flat_ptr, texel, ok, g, cc, channels, ROUNDS)
        else:
            g_lo = g * weight_lo
            _scatter_bilinear(grad_flat_ptr, base_lo, i00, i01, i10, i11, ok00, ok01, ok10, ok11,
                              fx, fy, g_lo, cc, channels, valid, ROUNDS)  # fmt: skip
            if COORD_GRADS:
                t00, t01, t10, t11 = _load_corners(
                    flat_ptr, base_lo, i00, i01, i10, i11, ok00, ok01, ok10, ok11, c, channels, cm
                )
                d_fx = (1 - fy) * (t01 - t00) + fy * (t11 - t10)
                d_fy = (1 - fx) * (t10 - t00) + fx * (t11 - t01)
                grad_u += tl.sum(g_lo * d_fx, axis=1) * w_lo.to(u.dtype) * sx_lo
                grad_v += tl.sum(g_lo * d_fy, axis=1) * h_lo.to(u.dtype) * sy_lo
            if MODE == 3:
                g_hi = g * frac[:, None]
                _scatter_bilinear(grad_flat_ptr, base_hi, j00, j01, j10, j11, k00, k01, k10, k11,
                                  gx, gy, g_hi, cc, channels, valid, ROUNDS)  # fmt: skip
            if (MODE == 3) and COORD_GRADS:
                s00, s01, s10, s11 = _load_corners(
                    flat_ptr, base_hi, j00, j01, j10, j11, k00, k01, k10, k11, c, channels, cm
                )
                e_fx = (1 - gy) * (s01 - s00) + gy * (s11 - s10)
                e_fy = (1 - gx) * (s10 - s00) + gx * (s11 - s01)
                grad_u += tl.sum(g_hi * e_fx, axis=1) * w_hi.to(u.dtype) * sx_hi
                grad_v += tl.sum(g_hi * e_fy, axis=1) * h_hi.to(u.dtype) * sy_hi
                lower = (1 - fy) * ((1 - fx) * t00 + fx * t01) + fy * ((1 - fx) * t10 + fx * t11)
                upper = (1 - gy) * ((1 - gx) * s00 + gx * s01) + gy * ((1 - gx) * s10 + gx * s11)
                grad_frac += tl.sum(g * (upper - lower), axis=1)

    if COORD_GRADS:
        tl.store(grad_uv_ptr + p * 2, grad_u, mask=valid)
        tl.store(grad_uv_ptr + p * 2 + 1, grad_v, mask=valid)

    if (MODE == 3) and COORD_GRADS:
        # The level of detail is clamped to the available levels.
        in_range = (lod_raw >= 0) & (lod_raw <= (num_levels - 1) * 1.0)
        grad_lod = tl.where(in_range, grad_frac, 0.0)
        if HAS_BIAS:
            tl.store(grad_bias_ptr + p, grad_lod, mask=valid)
    if (MODE == 3) and HAS_DA and COORD_GRADS:
        base_h = tl.load(hs_ptr)
        base_w = tl.load(ws_ptr)
        _, len_x, len_y, longest, slope_ux, slope_uy, slope_vx, slope_vy = _level_of_detail(
            da_ptr, p, valid, base_h, base_w
        )
        # lod = log2(longest) / 2, with the footprint clamped from below.
        grad_longest = tl.where(
            longest > 1e-30, grad_lod * 0.5 / (longest * 0.6931471805599453), 0.0
        )
        share_x = tl.where(len_x > len_y, 1.0, tl.where(len_x == len_y, 0.5, 0.0))
        grad_x = grad_longest * share_x
        grad_y = grad_longest * (1 - share_x)
        tl.store(grad_da_ptr + p * 4, grad_x * 2 * slope_ux, mask=valid)
        tl.store(grad_da_ptr + p * 4 + 1, grad_y * 2 * slope_uy, mask=valid)
        tl.store(grad_da_ptr + p * 4 + 2, grad_x * 2 * slope_vx, mask=valid)
        tl.store(grad_da_ptr + p * 4 + 3, grad_y * 2 * slope_vy, mask=valid)


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
        args = _launch_args(flat.shape, uv)
        with torch.cuda.device(uv.device):
            _forward_kernel[args["grid"]](
                flat, offsets, heights, widths, uv, da, bias, out,
                args["num_pixels"], args["pixels_per_image"], args["batch_stride"],
                channels, offsets.shape[0],
                MODE=mode, BOUNDARY=boundary, HAS_DA=has_da, HAS_BIAS=has_bias,
                BLOCK_P=BLOCK_PIXELS, BLOCK_C=args["block_c"], enable_fp_fusion=FP_FUSION,
            )  # fmt: skip
        # Texel values are needed only for the gradients of uv and the level of detail. Unless
        # one of those is wanted they are not kept, so a texture that is sampled directly may
        # be modified in place before the backward pass.
        coord_grads = mode > 0 and any(ctx.needs_input_grad[1:4])
        ctx.save_for_backward(flat if coord_grads else None, uv, da, bias, offsets, heights, widths)
        ctx.settings = (flat.shape, mode, boundary, has_da, has_bias)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grads = TextureGradients.apply(grad_out, *ctx.saved_tensors, *ctx.settings)
        return (*grads, None, None, None, None, None)


class TextureGradients(torch.autograd.Function):
    """The backward pass of :class:`Texture`.

    It is a function of its own so that differentiating its results raises. They depend on the
    texture, ``uv`` and the level of detail as well as on the incoming gradient, and no kernel
    computes those second-order terms.
    """

    @staticmethod
    def forward(
        ctx, grad_out, flat, uv, da, bias, offsets, heights, widths,
        flat_shape, mode, boundary, has_da, has_bias,
    ):  # fmt: skip
        coord_grads = flat is not None
        grad_out = grad_out.contiguous()
        grad_flat = uv.new_zeros(flat_shape)
        # Only trilinear filtering depends smoothly on the level of detail.
        smooth = mode == 3
        grad_uv = torch.empty_like(uv) if coord_grads else None
        grad_da = torch.zeros_like(da) if coord_grads and smooth and has_da else None
        grad_bias = torch.zeros_like(bias) if coord_grads and smooth and has_bias else None
        # Pointers the kernel does not use get a placeholder, as in the forward pass.
        unused = grad_flat
        args = _launch_args(flat_shape, uv)
        with torch.cuda.device(uv.device):
            _backward_kernel[args["grid"]](
                flat if coord_grads else unused, offsets, heights, widths, uv, da, bias, grad_out,
                grad_flat,
                unused if grad_uv is None else grad_uv,
                unused if grad_da is None else grad_da,
                unused if grad_bias is None else grad_bias,
                args["num_pixels"], args["pixels_per_image"], args["batch_stride"],
                flat_shape[-1], offsets.shape[0],
                MODE=mode, BOUNDARY=boundary, HAS_DA=has_da, HAS_BIAS=has_bias,
                COORD_GRADS=coord_grads, BLOCK_P=BLOCK_PIXELS, BLOCK_C=args["block_c"],
                ROUNDS=SCATTER_ROUNDS, enable_fp_fusion=FP_FUSION,
            )  # fmt: skip
        return grad_flat, grad_uv, grad_da, grad_bias

    @staticmethod
    def backward(ctx, *grads):
        raise RuntimeError(
            "The Triton backend of texture provides first-order gradients only. Select the "
            'reference backend, diffrast.set_backend("reference"), for higher-order gradients.'
        )


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
    if uv.numel() == 0:
        # There is nothing to launch. The PyTorch code returns an empty result within the graph.
        return reference.texture(
            tex, uv, uv_da, filter_mode, boundary_mode, max_mip_level, mip_level_bias, mip, packed
        )
    mode = FILTER_MODES.index(filter_mode)
    # Inputs the output does not depend on stay out of the graph, as in the reference backend.
    if mode == 0:
        uv = uv.detach()
    if mode < 3:
        uv_da = None if uv_da is None or mode < 2 else uv_da.detach()
        mip_level_bias = None if mip_level_bias is None or mode < 2 else mip_level_bias.detach()

    if mode >= 2:
        flat, heights, widths, offsets = reference._pack_levels(tex, mip, max_mip_level, packed)
    else:
        flat, heights, widths, offsets = reference._pack([tex])
    return Texture.apply(
        flat.contiguous(),
        uv,
        uv_da,
        mip_level_bias,
        offsets.int(),
        heights.int(),
        widths.int(),
        mode,
        BOUNDARY_MODES.index(boundary_mode),
    )


def _launch_args(flat_shape: Sequence[int], uv: Tensor) -> dict:
    """Launch parameters for a packed buffer of shape ``flat_shape`` sampled at ``uv``."""
    batch, height, width, _ = uv.shape
    num_pixels = batch * height * width
    return {
        "grid": (triton.cdiv(num_pixels, BLOCK_PIXELS),),
        "num_pixels": num_pixels,
        "pixels_per_image": height * width,
        "batch_stride": 0 if flat_shape[0] == 1 else flat_shape[1],
        "block_c": min(triton.next_power_of_2(flat_shape[-1]), 16),
    }
