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
- Triton `rasterize`: tile binning and a coverage kernel that merges depth with 64-bit atomics.
  Matches the reference backend pixel for pixel on the test scenes.
- Triton `rasterize` draws triangles with a pixel bounding box of at most 4 x 4 directly, without
  tile binning. On an RX 9070 XT, a 1M-triangle mesh at 8 views and 512 x 512 goes from about
  220 ms to 6 ms forward.
- `benchmarks/bench_rasterize.py`.

### Changed

- `auto` backend selection now routes each op to Triton only if Triton implements it.
- The reference backend computes barycentrics only for covered pixels, which removes a slow
  gradient accumulation through background pixels and lowers memory use.
