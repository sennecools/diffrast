# Design

This document describes how diffrast is structured and how each op is implemented. It is updated
as the implementation lands.

## Layout

```
src/diffrast/
  ops.py              public ops: validation and dispatch
  context.py          RasterizeContext, RasterizeGLContext
  peeling.py          DepthPeeler
  topology.py         edge adjacency for antialias, and its cache
  _cache.py           values derived from a tensor's contents, cached per tensor
  torch.py            the public API again, under nvdiffrast's module path
  compat.py           install_as_nvdiffrast / uninstall_as_nvdiffrast
  log.py              log level
  backends/
    reference.py      pure PyTorch, ground truth for tests
    _cube.py          cube-map face selection and lookups across face edges
    triton.py         Triton backend for CUDA and ROCm
    _triton_*.py      Triton kernels, imported only when used
    _autograd.py      guard for backward passes that cannot be differentiated again
benchmarks/           timing scripts
examples/             example programs and the helpers they share
```

`ops.py` owns the public signatures and all input checks. Backends receive validated arguments
and can assume shapes, dtypes and devices are correct. Inputs shared across the batch (a 2D
`attr`, a single-texture `tex`) are passed through unexpanded.

`interpolate` and `antialias` check that every index in `tri` refers to an existing vertex.
Finding the smallest and largest index takes a reduction and, on a GPU, a synchronisation, so the
result is cached per tensor. The cache keys an entry on the tensor object, holds it by weak
reference and stores its version counter, data address and shape, so an in-place edit (also
through a view) or new storage is checked again. Writes that bypass the version counter, such as
through `tensor.data` or memory shared with NumPy, are not seen. Triangle ids in `rast` are not
validated, since that would synchronise on every call; ids past the end of `tri` count as
background.

## Conventions

- Positions are in clip space, `[batch, vertices, 4]`. A vertex is inside the view volume when
  `-w <= x, y, z <= w`.
- Pixel `(row, col)` covers `[col, col + 1) x [row, row + 1)` in screen space and is sampled at its
  centre. Row 0 is at clip-space `y = -1`.
- `rast[..., 3]` stores `triangle_id + 1`, so zero means background.
- A pixel is covered by a triangle when its centre is inside all three edges. A centre exactly on
  an edge is inside only if the triangle's interior lies in the +x direction of screen space from
  that edge, or in the +y direction for a horizontal edge. Row 0 is at `y = -1`, so a shape owns
  the pixel centres on its left and top sides but not on its right and bottom sides, like the
  top-left rule of hardware rasterizers. Two triangles on opposite sides of a shared edge
  therefore never both cover a pixel and never leave one empty, whatever their winding.
- Visibility compares the depth each triangle computes at the pixel; exactly equal depths go to
  the lowest triangle index. Depths of different triangles in one plane are equal only up to
  rounding, so the lower index is guaranteed only for triangles with identical vertex data.
- Triangles with zero area (collinear vertices, repeated indices) and triangles with a NaN or
  infinite vertex coordinate are not drawn.
- A pixel is drawn only where the interpolated `w` is positive and `-1 <= z / w <= 1`.
- `rast[..., 3]` is float32, so at most `2**24` triangles can be drawn; `rasterize` raises above
  that.

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

### Coverage

`_coverage` evaluates, per triangle and pixel, `a_i = x_i - px w_i`, `c_i = y_i - py w_i`,
`e = a x c`, `s = sum(e)` and `depth = (e . z) / (e . w)`, and tests:

1. **Edges.** For the edge from vertex i to vertex j, `e_k = a_i c_j - a_j c_i`. Swapping i and j
   only swaps the order of one subtraction, so in floating point the two triangles sharing an
   edge compute exact negatives of each other (or, with opposite winding, equal values and
   opposite `s`), and exactly one of them accepts a pixel on the edge. Where `e_k` is exactly
   zero, the sign of its screen-space gradient decides: `gx = c_i w_j - c_j w_i`, or
   `gy = a_j w_i - a_i w_j` when `gx` is zero. These use the coordinates relative to the pixel,
   which are small and often exact where `e_k` is zero; at a pixel centre on a vertex the
   gradients of the edges through it are single products, so the triangles of a closed fan agree
   on which of them owns the pixel.
2. **Area.** `s` is the signed area seen from the pixel. Rounding of the vertex coordinates and of
   `a` and `c` can change it by about `eps (Rx sum|c| + sum|a| (Ry + sum|c|))`, with
   `Rx = max_i(|x_i| + |w_i|)` and `Ry` likewise. A triangle is skipped where `|s|` is at most
   twice that (`AREA_TOLERANCE`, shared with the Triton kernels). The test is invariant to the
   scale of clip space. Without it, a zero-area triangle is drawn wherever the rounding residue of
   its edge functions agrees in sign. Measured zero-area triangles stay at or below about one
   epsilon; the price is that a triangle within two epsilons of zero area at a pixel, about
   `4e-4` pixels thick at 1000 pixels, is not drawn there even if it is not degenerate.
3. **Depth.** The interpolated `w` is positive (`e . w` has the sign of `s`), `-1 <= depth <= 1`,
   and all vertex coordinates are finite.

The rule is exact only for the pixel centres a device computes: `pixel_centres` can round
differently on the CPU and on a GPU, so two devices may give a pixel on a shared edge to
different triangles. Coverage stays exactly once on each.

### texture

- Texel `[row, col]` of a `[height, width]` texture is centred at
  `uv = ((col + 0.5) / width, (row + 0.5) / height)`. `nearest` reads the texel containing `uv`;
  `linear` blends the four texels around it.
- Out-of-range texels are wrapped (`wrap`), clamped to the edge (`clamp`), or read as zero
  (`zero`); the boundary mode applies per texel, so `zero` fades out over half a texel.
- Mip level `i + 1` splits each dimension of level `i` into half as many equal bins, rounding up,
  until both sizes are 1. A coarse texel is the area-weighted mean of the texels its bin overlaps.
  For an even size that is the 2 x 2 box filter; an odd size needs fractional weights, three taps
  per coarse texel along that axis. Every base texel then has the same total weight in every
  level, so the 1 x 1 level is the texture mean, and the levels stay aligned in `uv`.
- The level of detail is `log2` of the longer screen-space footprint axis measured in base-level
  texels, `max(|d(u, v)/dx|, |d(u, v)/dy|)`, plus `mip_level_bias`, clamped to the available
  levels. `linear-mipmap-nearest` rounds it to the nearest level; `linear-mipmap-linear` blends
  the two neighbouring levels, so gradients also flow into `uv_da`. At the coarsest level both
  lookups read the same level, so the level of detail has no gradient there.
- Mip levels come from one of three places. Without `mip` they are built from `tex` on every call,
  and gradients reach `tex` through all levels. `texture_construct_mip(tex)` builds the same
  levels once. A list of tensors gives levels 1 and up directly, each with its own size and its
  own gradients.
- All levels are packed into one `[tex_batch, texels, channels]` buffer with the size and offset
  of each level, so a mipmapped lookup is two bilinear lookups however many levels the pixels use.
  A stack from `texture_construct_mip` keeps that buffer for a texture that does not require
  gradients, and packs afresh when `tex` is another tensor or was modified in place.
- Out-of-range input: wrapping reduces `uv` to its fractional part first; with `clamp` and `zero`
  far-out and infinite coordinates behave like any coordinate outside. NaN inputs give a NaN pixel
  that passes no gradient, and the footprint saturates, so huge `uv_da` selects the coarsest level.

### Cube maps

A cube map has shape `[batch, 6, size, size, channels]` with faces `+x, -x, +y, -y, +z, -z`,
oriented as in the OpenGL specification. `_cube.project` picks the face from the component of
largest magnitude and divides the direction by it, so face coordinates do not depend on the length
of the direction. Their screen-space derivatives follow from the six values of `uv_da` by the
quotient rule. From there a cube lookup is the 2D code on face coordinates; mip levels are the
faces filtered one by one.

Bilinear taps one texel outside a face are folded around the edge onto the neighbouring face. In
units of half a texel, texel centres are integer points on the surface of the cube
`[-size, size]^3`, so the fold is integer arithmetic and needs no table of edge cases. Beyond a
corner no face continues, and the missing tap counts as the mean of the other three. Each level is
therefore continuous across every edge and corner. The level of detail is measured on the plane
of the selected face, so a level chosen from `uv_da` can change from one face to the next.

Directions with a non-finite component, the zero vector, and vectors whose largest component is
subnormal select no point on the cube: they read zeros and pass no gradient. Cube lookups are
written in PyTorch with gradients from autograd; the Triton backend dispatches them to this code.

### antialias

Visibility is a step function of the vertex positions, so a plain render has no gradient that
moves a silhouette. `antialias` replaces the step with a ramp:

1. **Silhouette edges.** Edges are matched between triangles by their vertex indices. An edge is a
   silhouette when it belongs to one triangle only, to more than two, or when the two triangles
   sharing it project onto the same side of it (the surface folds over there). Edges inside a
   continuous surface are left alone even if the colour changes across them. Matching edges takes
   a sort over all of them, more than half the forward time for a mesh of some ten thousand
   triangles, and depends on `tri` only, so the adjacency is computed once per index tensor:
   explicitly with `antialias_construct_topology_hash(tri)`, or through the per-tensor cache
   described above. The test against `pos` is cheap and runs on every call.
2. **Crossings.** For each pair of horizontally or vertically adjacent pixels showing different
   triangles, each pixel's triangle is tested at the other pixel's centre. A triangle that does
   not cover it is exited through one of its edges; since the homogeneous edge functions are
   affine in screen space, the crossing point `t` along the segment between the centres follows
   by linear interpolation. If both triangles qualify, the nearer one owns the edge. An edge is
   only handled by the pixel pairs along the axis it crosses more steeply, taken from the edge's
   two vertices so every pair along it decides the same way, and edges within a relative `1e-4`
   of 45 degrees go to the horizontal pairs. Diagonal edges are therefore not blended twice.

   `rast` already says which triangle a pixel shows, so antialias does not decide coverage again
   from the sign of its own edge functions. A destination centre that lies on an edge up to a
   bound on the rounding error of the edge function counts as beyond it, with the crossing at that
   centre. A pixel centred on a silhouette is then half covered whichever side the rasterizer's
   fill rule or rounding put it on. No division is made where its result is discarded, since a
   division by zero there would still put NaN into the gradient.
3. **Blending.** Each pixel is treated as a unit box along that axis. The pixel whose half contains
   the crossing takes `|t - 0.5|` of the other pixel's colour, which is its exact covered fraction
   for a straight edge. Summed over a row, this recovers the crossing position exactly, so the
   total area of a straight edge is exact at any slope; only corners where two silhouettes meet
   are approximated.

The result is continuous in the vertex positions: when an edge passes a pixel centre and the
pixel changes owner, the blended colour is the same on both sides. `pos_gradient_boost` scales
the gradient through `t` without changing the image.

Because only the triangle a pixel shows is examined, a silhouette is found only where the
triangle at the outline is visible in a pixel next to the background. On a coarse mesh that is
nearly everywhere. On a finely tessellated smooth mesh the triangles at the outline are often
sub-pixel slivers hidden behind their neighbours, and a share of the outline pixels is left
unblended: in `examples/antialias_gradients.py`, 86 % of the outline pairs of a 42-vertex sphere
at 154 pixels are blended, against about 15 % for a 642-vertex sphere. Gradients then reach fewer
outline vertices per step, which optimisation tolerates but which is worth knowing.

## Triton rasterize

Rasterization runs in two passes: a visibility pass that finds the nearest triangle per pixel,
and a resolve pass that builds `rast` and `rast_db` for it and carries gradients back to `pos`.
`benchmarks/bench_rasterize.py --stages` times each stage separately.

### Visibility

All paths write to one buffer of 64-bit keys per pixel. A key packs the depth (as an
order-preserving integer) in the high 32 bits and the triangle index in the low 32 bits, so a
single `atomic_min` keeps the nearest triangle and resolves equal depths to the lower index.

1. **Setup** (`_setup_kernel`, one lane per triangle). A triangle is culled when all its vertices
   have `w <= 0`, or all lie outside the same clip plane (`x > w`, `x < -w`, and so on), since
   every point on it is a non-negative combination of its vertices. Triangles entirely in front of
   the camera are bounded by the pixel centres inside their projected bounding box; triangles
   crossing the camera plane get the whole screen. A triangle with a NaN or infinite coordinate is
   culled too; it would fail every comparison and be binned into all tiles. Back-facing triangles
   are not culled, and zero-area ones are rejected per pixel by the area test. The bounding box
   gets a margin of `1e-3 + size * 1e-6` pixels, since its float32 error grows with the image.
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
vertex data is made. They are launched with `enable_fp_fusion=False`: a fused multiply-add
computes `a_i c_j - a_j c_i` with one rounding instead of two, which breaks the exact antisymmetry
the fill rule relies on, and quads split along a diagonal had holes on it. Unfused, the kernels
perform the reference backend's operations one for one, so on the same device both backends pick
the same triangle and the same depth bits for every pixel.

### Resolve

`_triton_resolve.py` reads the visibility key of each pixel and evaluates the reference formulas
for the visible triangle in one kernel: the edge functions `e = a x c`, the barycentrics
`e / sum(e)`, the depth `(e . z) / (e . w)`, and their screen-space derivatives through
`de/dpx = c x w` and `de/dpy = w x a`. The backward kernel runs the same steps in reverse, using
`v x G` and `G x u` as the gradients of `u x v` for an upstream gradient `G`.

Each program handles a 16 x 8 tile of pixels, which mostly show the same few triangles. Before
vertex gradients are written with atomics, the pixels sharing a vertex are summed in registers,
round by round in order of vertex address. The rounds stop once they merge fewer than four pixels
each on average, which happens quickly for dense meshes where few pixels share a vertex, and
continue for coarse meshes where thousands of pixels contend for each vertex.

In float32 the kernels round differently from the reference code. Compared with a float64
computation, both are accurate to about `1e-5` relative, including for tiny triangles whose edge
functions cancel. Unlike the visibility kernels, the resolve kernels are compiled with fusion,
which makes the barycentrics of small triangles more accurate; which triangle a pixel shows is
already decided by then.

## Range mode

The rank of `pos` selects the mode, as in nvdiffrast. With `pos` of shape `[vertices, 4]`,
`ranges` is an int32 `[batch, 2]` tensor of `(start, count)` into `tri`, and batch item `i` draws
`tri[start_i : start_i + count_i]` from the shared vertex buffer. `ops.py` implements it above the
backends: each item is rasterized as an instanced call with a batch of one, and the results are
stacked with the triangle ids shifted by `start`, so they index the whole `tri` and
`interpolate` and `antialias` work unchanged with a shared `attr` and `pos`.

## Triton interpolate

`_triton_interpolate.py` has a forward and a backward kernel joined by a `torch.autograd.Function`.
Each program handles a block of 64 pixels and loops over the attribute channels in chunks of up
to 16, reading the visible triangle's three vertex rows directly from `attr`. Attributes shared
across the batch are read in place rather than expanded.

The backward kernel sums the gradients of `u`, `v` and the four `rast_db` entries over channels
in registers, so they need no atomics. Gradients to `attr` are scattered to the three vertices of
each pixel's triangle with `atomic_add`.

The interpolate, texture and antialias kernels, like the visibility kernels, are launched with
`enable_fp_fusion=False`, so they round like the PyTorch code. Every launch sets the option
explicitly because Triton 3.9 fuses by default and 3.10 will not, so results would otherwise
depend on the installed version.

The Triton backward passes are kernels and cannot be differentiated again. They are wrapped so
that running them with `create_graph=True` raises, rather than silently returning gradients that
lack their second-order terms. Higher-order gradients need the reference backend.

## Triton texture

`_triton_texture.py` builds the mip chain with the same PyTorch code as the reference backend and
packs every level into one `[tex_batch, texels, channels]` buffer, with per-level offsets and
sizes. Autograd carries gradients from that buffer back through the box filters to the base
texture, so the kernels only deal with sampling. Each program handles 64 pixels and loops over
channels in chunks of up to 16; filter and boundary modes are compile-time constants.

The backward kernel sums the gradients of `uv` and of the blend factor between mip levels in
registers, and chains the latter analytically through the level-of-detail formula into `uv_da`.
Texel gradients are scattered with `atomic_add`. Atomics on one address serialize, and many
pixels often share a texel: background pixels all sample the same `uv`, and magnified textures
cover many pixels per texel. So before scattering, each program sums the gradients of pixels
that share the lowest remaining texel address, for two rounds, and writes each group with one
atomic. Zero gradients, such as from masked-out background, are not written.

## Triton antialias

`_triton_antialias.py` classifies silhouette edges with the reference code (it is per triangle and
cheap), then runs one lane per pixel, handling the pairs the pixel forms with its right and lower
neighbours with the same crossing, ownership and blending rules as the reference backend. A
pixel is touched by at most four pairs, so blended colours and colour gradients are added with
atomics without meaningful contention.

The backward kernel chains the gradient of the crossing point `t = e_s / (e_s - e_d)` into the two
vertices of the crossed edge analytically. With `e = a_i c_j - a_j c_i`, `a = x - px w` and
`c = y - py w` evaluated at both pixel centres, the partial derivatives with respect to each
vertex's `x`, `y` and `w` are short closed forms; `z` does not affect `t`. In float64 the Triton
and reference gradients agree to about `1e-11`; in float32 they differ only by rounding, which is
largest for vertices close to the camera plane, whose gradients are very large.

## Depth peeling

`DepthPeeler` rasterizes the same geometry repeatedly. Each layer keeps, per pixel, the nearest
triangle ordered after the previous layer's triangle, where triangles are ordered by depth and
then by index. Ordering by both, rather than by depth alone, means that triangles at exactly the
same depth each get their own layer instead of being skipped or repeated.

The comparison has to use the visibility pass's own depth: `rast[..., 2]` is computed separately
and can differ in the last bit. So each backend carries its visibility result from one layer to
the next. The reference backend keeps the depth and index per pixel; the Triton backend keeps
its packed `(depth, index)` keys, so both kernels of the visibility pass only need one extra
comparison, `key > previous_key`. The resolve pass and the other ops are unchanged, because each
layer is an ordinary `rast` tensor.

Because a pixel on a shared edge belongs to one triangle only, a single surface gives a single
layer. The peeler keeps its own copy of `pos`, `tri` and `ranges`, so in-place edits made between
layers cannot mix two geometries, and it raises when used after its `with` block has exited.

## Gradients

- Through `rasterize`, gradients flow from `(u, v, z / w)` and `rast_db` into the vertex positions
  of the visible triangle. Which triangle is visible is a discrete choice and has no gradient.
- `antialias` provides the missing visibility gradient: along silhouette edges it blends the
  colours on both sides by the edge's coverage of the pixel, so moving a vertex changes the image
  continuously.
- `interpolate` passes gradients to `attr`, `rast` and `rast_db`; `texture` to `tex`, `uv`,
  `uv_da`, `mip_level_bias` and the tensors of a custom mip list.

## Compatibility with nvdiffrast

`RasterizeGLContext` is a `RasterizeContext` that stores `output_db` and `mode` and has no-op
`set_context()` and `release_context()`; there is no OpenGL rasterizer behind it. With
`output_db=False`, `rasterize` returns an empty `[batch, height, width, 0]` tensor in place of
`rast_db`. `diffrast/torch.py` star-imports `diffrast`, so `import diffrast.torch as dr` gives
every public name. `install_as_nvdiffrast()` puts `diffrast` and `diffrast.torch` into
`sys.modules` as `nvdiffrast` and `nvdiffrast.torch`, and refuses to shadow an installed
nvdiffrast unless `force=True`.

## Testing strategy

- **Analytic scenes:** a single triangle, overlapping triangles with known depth order, a cube.
  Coverage, ids and barycentrics are compared against values computed by hand.
- **Backend parity:** every Triton kernel is compared against the reference backend on random
  meshes.
- **Gradients:** `torch.autograd.gradcheck` in float64 on tiny scenes; finite differences and
  gradcheck with a fixed rasterization for `antialias`.
- **Edge cases:** triangles behind the camera or crossing the near plane, degenerate and
  sub-pixel triangles, shared edges, silhouettes exactly through pixel centres.
- **Fill rule:** meshes with vertices on the pixel-centre lattice, also under integer
  homographies, are compared with an exact rational rasterizer; flat sheets must have an empty
  second depth layer, and Triton ids must equal reference ids on every pixel of every layer.
- **Cube maps:** compared with a brute-force float64 sampler written from the OpenGL face table,
  which finds neighbouring texels by projection rather than by the integer fold.
- **Examples:** every example runs with small sizes on the CPU in `tests/test_examples.py`.
