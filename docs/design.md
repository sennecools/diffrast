# Design

This document describes how diffrast is structured and how each op is implemented. It is updated
as the implementation lands.

## Layout

```
src/diffrast/
  ops.py              public ops: validation and dispatch
  context.py          RasterizeContext
  backends/
    reference.py      pure PyTorch, ground truth for tests
    triton.py         Triton backend for CUDA and ROCm
    _triton_*.py      Triton kernels, imported only when used
benchmarks/           timing scripts
```

`ops.py` owns the public signatures and all input checks. Backends receive validated arguments
and can assume shapes, dtypes and devices are correct. Inputs shared across the batch (a 2D
`attr`, a single-texture `tex`) are passed through unexpanded.

## Conventions

- Positions are in clip space, `[batch, vertices, 4]`. A vertex is inside the view volume when
  `-w <= x, y, z <= w`.
- Pixel `(row, col)` covers `[col, col + 1) x [row, row + 1)` in screen space and is sampled at its
  centre. Row 0 is at clip-space `y = -1`.
- `rast[..., 3]` stores `triangle_id + 1`, so zero means background.
- Edges are inclusive: a pixel centre exactly on an edge is covered, so shared edges never leave
  cracks. When several triangles cover a pixel at the same depth, the lowest triangle id wins.
- A pixel is drawn only where the interpolated `w` is positive and `-1 <= z / w <= 1`.

## Reference backend

`backends/reference.py` rasterizes in homogeneous coordinates (Olano and Greer, 1997). For a pixel
at NDC `(px, py)`, the perspective-correct barycentrics `b` of a triangle satisfy

```
sum(b * (x - px * w)) = 0
sum(b * (y - py * w)) = 0
sum(b) = 1
```

so `b` is the cross product of the two vectors `x - px * w` and `y - py * w`, divided by the sum
of its components. Because the test is done on the interpolated `w` and `z / w` per pixel,
triangles crossing the camera plane need no clipping.

Visibility is resolved without autograd, evaluating triangles in chunks to bound memory. The
barycentrics of the winning triangle are then recomputed per pixel with autograd enabled, so
gradients reach exactly the vertices of the visible triangle. `rast_db` uses the analytic
derivatives of `b` with respect to `(px, py)`.

### texture

- Texel `[row, col]` of a `[height, width]` texture is centred at
  `uv = ((col + 0.5) / width, (row + 0.5) / height)`. `nearest` reads the texel containing `uv`;
  `linear` blends the four texels around it.
- Out-of-range texels are wrapped (`wrap`), clamped to the edge (`clamp`), or read as zero
  (`zero`); the boundary mode applies per texel, so `zero` fades out over half a texel.
- Mip level `i + 1` is a 2 x 2 box filter of level `i`, with each size halved and rounded up until
  both are 1. An odd edge is averaged over the texels it covers.
- The level of detail is `log2` of the longer screen-space footprint axis measured in base-level
  texels, `max(|d(u, v)/dx|, |d(u, v)/dy|)`, clamped to the available levels.
  `linear-mipmap-nearest` rounds it to the nearest level; `linear-mipmap-linear` blends the two
  neighbouring levels, so gradients also flow into `uv_da`.
- The mip chain is rebuilt on every call from `tex`, so gradients reach the base texture through
  all levels.

### antialias

Visibility is a step function of the vertex positions, so a plain render has no gradient that
moves a silhouette. `antialias` replaces the step with a ramp:

1. **Silhouette edges.** Edges are matched between triangles by their vertex indices. An edge is a
   silhouette when it belongs to one triangle only, to more than two, or when the two triangles
   sharing it project onto the same side of it (the surface folds over there). Edges inside a
   continuous surface are left alone even if the colour changes across them.
2. **Crossings.** For each pair of horizontally or vertically adjacent pixels showing different
   triangles, each pixel's triangle is tested at the other pixel's centre. A triangle that does
   not cover it is exited through one of its edges; since the homogeneous edge functions are
   affine in screen space, the crossing point `t` along the segment between the centres follows
   by linear interpolation. If both triangles qualify, the nearer one owns the edge. An edge is
   only handled by the pixel pairs along the axis it crosses more steeply, so diagonal edges are
   not blended twice.
3. **Blending.** Each pixel is treated as a unit box along that axis. The pixel whose half contains
   the crossing takes `|t - 0.5|` of the other pixel's colour, which is its exact covered fraction
   for a straight edge. Summed over a row, this recovers the crossing position exactly, so the
   total area of a straight edge is exact at any slope; only corners where two silhouettes meet
   are approximated.

The result is continuous in the vertex positions: when an edge passes a pixel centre and the
pixel changes owner, the blended colour is the same on both sides. `pos_gradient_boost` scales
the gradient through `t` without changing the image.

## Triton rasterize

The Triton backend replaces only the visibility pass; `rast` and `rast_db` are then built by the
same differentiable code as the reference backend (`reference.rasterize_from_ids`), so gradients
are identical by construction.

All paths write to one buffer of 64-bit keys per pixel. A key packs the depth (as an
order-preserving integer) in the high 32 bits and the triangle index in the low 32 bits, so a
single `atomic_min` keeps the nearest triangle and resolves equal depths to the lower index.

1. **Setup** (`_setup_kernel`, one lane per triangle). A triangle is culled when all its vertices
   have `w <= 0`, or all lie outside the same clip plane (`x > w`, `x < -w`, and so on), since
   every point on it is a non-negative combination of its vertices. Triangles entirely in front of
   the camera are bounded by the pixel centres inside their projected bounding box; triangles
   crossing the camera plane get the whole screen. Back-facing and degenerate triangles are not
   culled.
2. **Small triangles** (same kernel). A triangle whose box fits in 4 x 4 pixels is rasterized
   right away: 16 lanes test those pixels and write hits with `atomic_min`. Dense meshes at
   moderate resolutions are mostly such triangles, and they skip binning entirely.
3. **Binning** (PyTorch, larger triangles only). The image is split into 16 x 16 tiles. Each
   triangle is added to every tile its box overlaps, the pairs are sorted by tile, and each tile's
   list is cut into work items of up to 64 triangles. Tile ranges come from a binary search on
   the sorted list; `bincount` was much slower on ROCm because its atomics contend when many
   triangles share few tiles.
4. **Coverage and depth** (`_tile_kernel`). One program per work item evaluates the homogeneous
   edge functions for its 256 pixels, keeps the nearest hit in registers, and merges it into the
   buffer with one `atomic_min` per pixel.

Both kernels read vertices through `tri` directly from `pos`, so no per-triangle copy of the
vertex data is made. The coverage and depth tests use the same formulas as the reference backend,
and on the test scenes both backends pick the same triangle for every pixel.

## Gradients

- Through `rasterize`, gradients flow from `(u, v, z / w)` and `rast_db` into the vertex positions
  of the visible triangle. Which triangle is visible is a discrete choice and has no gradient.
- `antialias` provides the missing visibility gradient: along silhouette edges it blends the
  colours on both sides by the edge's coverage of the pixel, so moving a vertex changes the image
  continuously.
- `interpolate` passes gradients to `attr`, `rast` and `rast_db`; `texture` to `tex`, `uv` and
  `uv_da`.

## Testing strategy

- **Analytic scenes:** a single triangle, overlapping triangles with known depth order, a cube.
  Coverage, ids and barycentrics are compared against values computed by hand.
- **Backend parity:** every Triton kernel is compared against the reference backend on random
  meshes.
- **Gradients:** `torch.autograd.gradcheck` in float64 on tiny scenes; finite differences for
  `antialias`.
- **Edge cases:** triangles behind the camera or crossing the near plane, degenerate and
  sub-pixel triangles, shared edges.
