"""Time antialias of a shaded grid mesh, whose silhouette runs across the image.

Usage: python benchmarks/bench_antialias.py [--backends triton reference] [--views 1 8]
"""

from __future__ import annotations

import argparse

import torch
from bench_rasterize import grid_mesh, time_ms

import diffrast as dr


def measure(ctx, pos, tri, res, repeats, warmup_s) -> tuple[float, float]:
    """Return forward and forward + backward time in milliseconds."""
    gen = torch.Generator(device=pos.device).manual_seed(0)
    shade = torch.rand(pos.shape[1], 3, device=pos.device, generator=gen)
    with torch.no_grad():
        rast, _ = dr.rasterize(ctx, pos, tri, (res, res))
        color, _ = dr.interpolate(shade, rast, tri)
    pos = pos.clone().requires_grad_(True)
    color.requires_grad_(True)

    def forward():
        with torch.no_grad():
            dr.antialias(color, rast, pos, tri)

    def forward_backward():
        dr.antialias(color, rast, pos, tri).sum().backward()

    return time_ms(forward, repeats, warmup_s), time_ms(forward_backward, repeats, warmup_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["triton", "reference"])
    parser.add_argument("--triangles", nargs="+", type=int, default=[100_000])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--views", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=float, default=3.0, help="warmup seconds per case")
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    ctx = dr.RasterizeContext()
    print(f"device: {torch.cuda.get_device_name(ctx.device)}, torch {torch.__version__}")
    row = "{:<10} {:>10} {:>5} {:>5} {:>9} {:>11}"
    print(row.format("backend", "triangles", "res", "views", "fwd ms", "fwd+bwd ms"))

    for num_tri in args.triangles:
        for views in args.views:
            pos, tri = grid_mesh(num_tri, views, ctx.device)
            for res in args.resolutions:
                for backend in args.backends:
                    dr.set_backend(backend)
                    try:
                        fwd, fwd_bwd = measure(ctx, pos, tri, res, args.repeats, args.warmup)
                        times = (f"{fwd:.2f}", f"{fwd_bwd:.2f}")
                    except torch.OutOfMemoryError:
                        times = ("OOM", "OOM")
                        torch.cuda.empty_cache()
                    print(row.format(backend, tri.shape[0], res, views, *times))


if __name__ == "__main__":
    main()
