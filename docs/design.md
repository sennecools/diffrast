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
    triton.py         Triton kernels for CUDA and ROCm
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

## rasterize

1. **Setup.** Per triangle: clip against the near plane, project to screen space, compute the
   screen-space bounding box and edge-function coefficients. Back-facing and degenerate triangles
   are kept; culling is the caller's choice.
2. **Binning.** The image is split into tiles (16 x 16 pixels to start). Each triangle is appended
   to the list of every tile its bounding box overlaps.
3. **Coverage and depth.** One program per tile walks its triangle list, evaluates the edge
   functions at each pixel centre, and keeps the nearest hit. Two strategies are being compared:
   a 64-bit atomic min on `(depth, triangle_id)` packed into one integer, and a per-tile sort by
   depth. 64-bit atomics behave differently across AMD and NVIDIA, so both are measured.
4. **Barycentrics.** For the winning triangle, perspective-correct `(u, v)` and `z / w` are
   computed in a second pass, along with their screen-space derivatives for `rast_db`.

## Gradients

- Through `rasterize`, gradients flow from `(u, v, z / w)` and `rast_db` into the vertex positions
  of the visible triangle. Which triangle is visible is a discrete choice and has no gradient.
- `antialias` provides the missing visibility gradient: along silhouette edges it blends the
  colours on both sides by the edge's analytic coverage of the pixel, so moving a vertex changes
  the image smoothly.
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
