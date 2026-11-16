# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Public API for `rasterize`, `interpolate`, `texture` and `antialias`, with input validation.
- Backend selection between `triton` and `reference`.
- Reference implementations of `rasterize` and `interpolate`, with gradients to positions,
  attributes and barycentrics, and screen-space derivatives (`rast_db`, `out_da`).
- Reference `texture` with `nearest`, `linear`, `linear-mipmap-nearest` and `linear-mipmap-linear`
  filtering and `wrap`, `clamp` and `zero` boundaries, with gradients to `tex`, `uv` and `uv_da`.
- Reference `antialias`: blends colours across silhouette edges by where each edge crosses
  between pixel centres, giving continuous images and gradients to vertex positions.
- Triton `rasterize`: tile binning and a coverage kernel that merges depth with 64-bit atomics.
  Matches the reference backend pixel for pixel on the test scenes.
- Triton `rasterize` draws triangles with a pixel bounding box of at most 4 x 4 directly, without
  tile binning. On an RX 9070 XT, a 1M-triangle mesh at 8 views and 512 x 512 goes from about
  220 ms to 6 ms forward.
- Triton `interpolate`, forward and backward, including attribute derivatives.
- Triton `texture` for all filter and boundary modes, forward and backward. Texel gradients are
  pre-reduced in registers where pixels share texels before they are written with atomics.
- `DepthPeeler` for depth peeling, with nvdiffrast's `rasterize_next_layer()` interface, on both
  backends. Triangles at equal depth are returned in turn, in index order.
- `texture` accepts `mip_level_bias`, a per-pixel offset to the mip level of detail that also
  works without `uv_da` and receives gradients with trilinear filtering.
- Triton `antialias`, forward and backward, with position gradients chained analytically through
  the edge functions.
- Benchmarks for each op in `benchmarks/`, and `benchmarks/bench_pipeline.py`, which times a full
  differentiable render, with `plot_pipeline.py` to chart its results.
- Cube-map textures in `texture`: `tex` of shape `[batch, 6, size, size, channels]`, direction
  vectors as `uv`, six-channel `uv_da` and `boundary_mode="cube"`, with every filter mode. Lookups
  are filtered across face edges and corners without seams. They run in PyTorch on both backends.
- `texture(mip=...)` accepts a list of tensors as custom mip levels, each with its own gradients.
- `texture_construct_mip(tex, max_mip_level=None, cube_mode=False)` builds the mip levels once for
  several lookups; for a constant texture it also keeps the packed buffer the kernels sample.
- `antialias_construct_topology_hash(tri)`. Without it, `antialias` caches the edge adjacency of a
  `tri` tensor between calls for as long as the tensor is not modified.
- `antialias` accepts `pos` of shape `[vertices, 4]`, shared by all batch items.
- `RasterizeGLContext(output_db=True, mode="automatic", device=None)`, so code written for
  nvdiffrast's OpenGL context runs unchanged. Contexts also accept a GPU index as `device`.
- `diffrast.torch`, which re-exports the public API, so `import nvdiffrast.torch as dr` ports to
  `import diffrast.torch as dr`; and `install_as_nvdiffrast()` / `uninstall_as_nvdiffrast()`, which
  register diffrast under the module name `nvdiffrast` so third-party code runs without edits.
- `get_log_level()` and `set_log_level(level)`.
- Range mode: `rasterize` and `DepthPeeler` accept `pos` of shape `[vertices, 4]` with `ranges`,
  an int32 `[batch, 2]` tensor of `(start, count)` into `tri`.
- For a context whose `output_db` is false, `rasterize` and `DepthPeeler` return an empty
  `rast_db` of shape `[batch, height, width, 0]`.
- Examples: `texture_fitting.py` (rewritten on a torus knot), `shape_fitting.py`,
  `pose_fitting.py`, `depth_peeling.py`, `gbuffer.py` and `antialias_gradients.py`, with shared
  helpers in `examples/common.py`.

### Changed

- Triton `rasterize` builds `rast` and `rast_db` and their gradients in fused kernels instead of
  PyTorch ops. At 2048 x 2048 with 8 views on an RX 9070 XT, this stage takes 31 to 43 ms forward
  and backward instead of 128 to 219 ms.
- `benchmarks/bench_rasterize.py --stages` times each stage of the Triton rasterizer.
- `auto` backend selection now routes each op to Triton only if Triton implements it.
- The reference backend computes barycentrics only for covered pixels, which removes a slow
  gradient accumulation through background pixels and lowers memory use.
- A pixel centre exactly on an edge belongs to one triangle only: the one whose interior lies in
  the +x direction, or +y for a horizontal edge (a top-left rule). Triangles sharing an edge no
  longer both cover its pixels, so `DepthPeeler` no longer returns a second layer along the
  edges of a flat surface. Both backends pick the same triangle for every pixel on one device.
- The first parameter of `rasterize` and `DepthPeeler` is named `glctx`, as in nvdiffrast's
  documentation.
- `rasterize` raises `ValueError` for more than `2**24` triangles, the largest id `rast` holds
  exactly, and for indices in `tri` past the end of `pos`.
- `DepthPeeler` keeps its own copy of the geometry, so in-place edits between layers do not mix
  two meshes, and raises when used after its `with` block has exited instead of restarting.
- The Triton `interpolate` backward pass skips the attribute gradient when `attr` does not
  require one. Those writes are atomics that many pixels aim at the same vertex, so for a
  10,000-triangle mesh at 8 x 2048 x 2048 this pass goes from about 160 ms to 5 ms.
- `interpolate` always returns two tensors. Without `diff_attrs` the second is empty, of shape
  `[batch, height, width, 0]`, instead of `None`; `texture` and `interpolate` accept it as "no
  derivatives".
- `interpolate` and `antialias` raise `ValueError` when `tri` refers to a vertex that does not
  exist. The check is cached per index tensor.
- Mip levels of odd-sized textures are built by exact area weighting, so every texel has the same
  weight in every level and the coarsest level is the texture mean. Even sizes are unchanged.
- `texture` defines its behaviour for out-of-range input identically on both backends: far-out and
  infinite coordinates count as outside, NaN inputs give a NaN pixel with no gradient, and huge
  `uv_da` selects the coarsest level.
- Every Triton kernel sets floating-point fusion explicitly (off everywhere except the rasterizer's
  resolve pass), so results no longer depend on the installed Triton's default and Triton 3.9's
  warning about that default is gone.
- On the Triton backend, differentiating the gradients of any op a second time raises an error instead of silently dropping second-order terms. Higher-order
  gradients need the reference backend.
- Error messages for wrong shapes, devices and argument types name the argument and the expected
  layout.

### Fixed

- The Triton rasterizer left background holes on shared edges through pixel centres, such as quad
  diagonals and pixel-aligned grids, because fused multiply-adds broke the exact antisymmetry of
  the edge functions. The visibility kernels are compiled without fusion.
- Zero-area triangles were drawn as one-pixel lines with huge `rast_db` and gradients; triangles
  whose area at a pixel is within rounding noise of zero are now skipped on both backends.
- Triangles with a NaN or infinite vertex are culled; on the Triton backend they were binned into
  every screen tile.
- `rasterize` with a batch of zero raised `ZeroDivisionError`.
- The Triton rasterizer missed edge pixels in images about 60000 pixels wide, and its tile offsets
  could overflow 32 bits for very large batches.
- Float64 `rast_db` on the Triton backend was limited to float32 accuracy.
- The reference `antialias` returned NaN position gradients for exactly horizontal or vertical
  screen-space edges, such as an upright cube.
- `antialias` blends a pixel whose centre lies exactly on a silhouette edge by one half, whichever
  side the rasterizer gave it to. Before, such pixels could be left unblended.
- A texture with a zero-sized dimension raises `ValueError` instead of aborting the process on the
  Triton backend.
- The reference `texture` returned a gradient for `mip_level_bias` and `uv_da` at the coarsest mip
  level.
- The Triton `texture` computed the wrong level of detail for textures 46341 texels or more wide.
