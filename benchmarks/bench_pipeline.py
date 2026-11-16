"""Time a full differentiable render: rasterize, interpolate, mipmapped texture and antialias.

The scene is the displaced grid of ``bench_rasterize.py`` with a 1024 x 1024 RGB texture. The
backward pass carries gradients to both the vertex positions and the texture.

Usage: python benchmarks/bench_pipeline.py [--backends triton reference] [--json results.json]
"""

from __future__ import annotations

import argparse
import json
import platform

import torch

import diffrast as dr
from bench_rasterize import grid_mesh, time_ms


def measure(ctx, pos, tri, res: int, repeats: int, warmup_s: float) -> tuple[float, float]:
    """Return forward and forward + backward time in milliseconds."""
    gen = torch.Generator(device=pos.device).manual_seed(0)
    # Planar mapping: the grid's vertices are laid out row by row, (n + 1) per row.
    n = round(pos.shape[1] ** 0.5) - 1
    steps = torch.linspace(0, 1, n + 1, device=pos.device)
    uv_attr = torch.stack(torch.meshgrid(steps, steps, indexing="xy"), -1).reshape(-1, 2)
    tex = torch.rand(1, 1024, 1024, 3, device=pos.device, generator=gen, requires_grad=True)
    pos = pos.clone().requires_grad_(True)

    def render():
        rast, rast_db = dr.rasterize(ctx, pos, tri, (res, res))
        uv, uv_da = dr.interpolate(uv_attr, rast, tri, rast_db=rast_db, diff_attrs="all")
        color = dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear")
        return dr.antialias(color, rast, pos, tri)

    def forward():
        with torch.no_grad():
            render()

    def forward_backward():
        render().sum().backward()

    return time_ms(forward, repeats, warmup_s), time_ms(forward_backward, repeats, warmup_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["triton", "reference"])
    parser.add_argument("--triangles", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--views", nargs="+", type=int, default=[8])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmup", type=float, default=8.0, help="warmup seconds per case")
    parser.add_argument("--memory-fraction", type=float, default=0.6)
    parser.add_argument("--max-reference-work", type=float, default=2e10)
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    ctx = dr.RasterizeContext()
    device = torch.cuda.get_device_name(ctx.device)
    print(f"device: {device}, torch {torch.__version__}")
    row = "{:<10} {:>10} {:>5} {:>5} {:>9} {:>11}"
    print(row.format("backend", "triangles", "res", "views", "fwd ms", "fwd+bwd ms"))

    results = []
    for num_tri in args.triangles:
        for views in args.views:
            pos, tri = grid_mesh(num_tri, views, ctx.device)
            for res in args.resolutions:
                for backend in args.backends:
                    work = tri.shape[0] * views * res * res
                    if backend == "reference" and work > args.max_reference_work:
                        continue
                    dr.set_backend(backend)
                    try:
                        fwd, fwd_bwd = measure(ctx, pos, tri, res, args.repeats, args.warmup)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(row.format(backend, tri.shape[0], res, views, "OOM", "OOM"))
                        continue
                    times = (f"{fwd:.2f}", f"{fwd_bwd:.2f}")
                    print(row.format(backend, tri.shape[0], res, views, *times))
                    results.append(
                        {
                            "backend": backend,
                            "triangles": tri.shape[0],
                            "resolution": res,
                            "views": views,
                            "forward_ms": round(fwd, 3),
                            "forward_backward_ms": round(fwd_bwd, 3),
                        }
                    )

    if args.json:
        import triton

        report = {
            "device": device,
            "torch": torch.__version__,
            "triton": triton.__version__,
            "diffrast": dr.__version__,
            "platform": platform.platform(),
            "results": results,
        }
        with open(args.json, "w") as file:
            json.dump(report, file, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
