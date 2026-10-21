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
