# diffrast

[![CI](https://github.com/sennecools/diffrast/actions/workflows/ci.yml/badge.svg)](https://github.com/sennecools/diffrast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A differentiable mesh rasterizer for PyTorch that runs on **AMD (ROCm) and NVIDIA (CUDA)** GPUs,
under the **MIT license**, with an API that mirrors [nvdiffrast](https://github.com/NVlabs/nvdiffrast).

> **Status: early development.** `rasterize` and `interpolate` work on the reference backend;
> `texture`, `antialias` and the Triton kernels are in progress. See the [roadmap](#roadmap).

## Why

Most image-to-3D, texturing and inverse-rendering pipelines render a mesh from several views and
backpropagate through the result. The standard tool for this, nvdiffrast, has two limitations:

- **It needs an NVIDIA GPU.** Pipelines built on it do not run on AMD hardware.
- **It is licensed for non-commercial use only** (NVIDIA Source Code License), and so is any
  derivative of its code.

diffrast is a clean-room implementation written in [Triton](https://github.com/triton-lang/triton),
so a single codebase targets both vendors, plus a pure-PyTorch reference backend that runs anywhere
and serves as ground truth for testing.

## Ops

| Op | Forward | Gradients to |
| --- | --- | --- |
| `rasterize(ctx, pos, tri, resolution)` | Per-pixel barycentrics, depth, triangle id; screen-space barycentric derivatives | `pos` |
| `interpolate(attr, rast, tri)` | Per-pixel vertex attributes (UVs, normals, colours) | `attr`, `rast` |
| `texture(tex, uv, uv_da=None)` | Nearest, bilinear and mipmapped lookups | `tex`, `uv` |
| `antialias(color, rast, pos, tri)` | Analytic edge antialiasing, giving gradients at silhouettes | `color`, `pos` |

All ops take a leading batch dimension, so several views render in one call.

## Usage

```python
import torch
import diffrast as dr

ctx = dr.RasterizeContext()  # current GPU, or CPU if there is none

rast, rast_db = dr.rasterize(ctx, pos_clip, tri, resolution=(512, 512))
uv, uv_da = dr.interpolate(uv_attr, rast, tri, rast_db=rast_db, diff_attrs="all")
color = dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear")
color = dr.antialias(color, rast, pos_clip, tri)

loss = torch.nn.functional.mse_loss(color, target)
loss.backward()  # gradients reach tex and pos_clip
```

Function names, argument names and argument order follow nvdiffrast's documented API, and
`RasterizeCudaContext` is an alias of `RasterizeContext`, so most code ports by changing the
import to `import diffrast as dr`. Not yet supported: the OpenGL context, range mode, depth
peeling, cube-map textures, `mip_level_bias` and precomputed `mip` stacks.

### Backends

| Backend | Devices | Notes |
| --- | --- | --- |
| `triton` | CUDA and ROCm GPUs | Default on GPU when Triton is installed |
| `reference` | Any PyTorch device | Pure PyTorch; slow, used for testing and as a fallback |

Select one with `dr.set_backend("reference")` or the `DIFFRAST_BACKEND` environment variable.

## Installation

diffrast is not on PyPI yet. Install PyTorch for your platform first
([CUDA or ROCm build](https://pytorch.org/get-started/locally/)), then:

```bash
pip install git+https://github.com/sennecools/diffrast
```

Linux builds of PyTorch ship with Triton. On Windows, only the reference backend is currently
available.

## Roadmap

- [x] Public API and input validation
- [x] Reference backend: `rasterize`, `interpolate`
- [ ] Reference backend: `texture`, `antialias`
- [ ] Triton backend: `rasterize`, `interpolate`
- [ ] Triton backend: `texture`, `antialias`
- [ ] Texture-fitting example
- [ ] Benchmarks on AMD and NVIDIA
- [ ] PyPI release

## Development

```bash
git clone https://github.com/sennecools/diffrast
cd diffrast
python -m pip install --upgrade pip  # --group needs pip 25.1+
pip install -e . --group dev
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and [docs/design.md](docs/design.md) for how
the rasterizer works.

## License

MIT. See [LICENSE](LICENSE).

diffrast is an independent project and is not affiliated with or endorsed by NVIDIA.
