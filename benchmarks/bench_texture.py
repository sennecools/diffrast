"""Time texture lookups of a 1024 x 1024 RGB texture mapped onto a rasterized grid mesh.

Usage: python benchmarks/bench_texture.py [--backends triton reference] [--views 1 8]
"""

from __future__ import annotations

import argparse

import torch

import diffrast as dr
from bench_rasterize import grid_mesh, time_ms


def measure(ctx, pos, tri, res, filter_mode, repeats, warmup_s) -> tuple[float, float]:
    """Return forward and forward + backward time in milliseconds."""
    gen = torch.Generator(device=pos.device).manual_seed(0)
    # Planar mapping: the grid's vertices are laid out row by row, (n + 1) per row.
    n = round(pos.shape[1] ** 0.5) - 1
    steps = torch.linspace(0, 1, n + 1, device=pos.device)
    uv_attr = torch.stack(torch.meshgrid(steps, steps, indexing="xy"), -1).reshape(-1, 2)
    with torch.no_grad():
        rast, rast_db = dr.rasterize(ctx, pos, tri, (res, res))
        uv, uv_da = dr.interpolate(uv_attr, rast, tri, rast_db=rast_db, diff_attrs="all")
    tex = torch.rand(1, 1024, 1024, 3, device=pos.device, generator=gen, requires_grad=True)
    uv.requires_grad_(True)
    uv_da.requires_grad_(True)

    def forward():
        with torch.no_grad():
            dr.texture(tex, uv, uv_da, filter_mode=filter_mode)

    def forward_backward():
        dr.texture(tex, uv, uv_da, filter_mode=filter_mode).sum().backward()

    return time_ms(forward, repeats, warmup_s), time_ms(forward_backward, repeats, warmup_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["triton", "reference"])
    parser.add_argument("--filters", nargs="+", default=["linear", "linear-mipmap-linear"])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--views", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmup", type=float, default=4.0, help="warmup seconds per case")
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    ctx = dr.RasterizeContext()
    print(f"device: {torch.cuda.get_device_name(ctx.device)}, torch {torch.__version__}")
    row = "{:<10} {:<21} {:>5} {:>5} {:>9} {:>11}"
    print(row.format("backend", "filter", "res", "views", "fwd ms", "fwd+bwd ms"))

    for views in args.views:
        pos, tri = grid_mesh(100_000, views, ctx.device)
        for res in args.resolutions:
            for filter_mode in args.filters:
                for backend in args.backends:
                    dr.set_backend(backend)
                    try:
                        fwd, fwd_bwd = measure(
                            ctx, pos, tri, res, filter_mode, args.repeats, args.warmup
                        )
                        times = (f"{fwd:.2f}", f"{fwd_bwd:.2f}")
                    except torch.OutOfMemoryError:
                        times = ("OOM", "OOM")
                        torch.cuda.empty_cache()
                    print(row.format(backend, filter_mode, res, views, *times))


if __name__ == "__main__":
    main()
