"""Time rasterize on a perspective view of a displaced grid mesh.

Usage: python benchmarks/bench_rasterize.py [--backends triton reference] [--views 1 8]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import torch

import diffrast as dr


def grid_mesh(num_tri: int, views: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(1, round(math.sqrt(num_tri / 2)))
    u = torch.linspace(-1, 1, n + 1, device=device)
    gx, gy = torch.meshgrid(u, u, indexing="xy")
    height = 0.1 * torch.sin(6 * gx) * torch.cos(5 * gy)
    world = torch.stack([gx, height, gy], -1).reshape(-1, 3)

    idx = torch.arange((n + 1) * (n + 1), device=device).view(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tri = torch.cat([torch.stack([a, b, d], -1), torch.stack([a, d, c], -1)]).reshape(-1, 3)

    views_pos = []
    for i in range(views):
        angle = 2 * math.pi * i / max(views, 1)
        rot = torch.tensor(
            [
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ],
            device=device,
        )
        cam = world @ rot.T + torch.tensor([0.0, -0.6, 2.2], device=device)
        f, near, far = 1.5, 0.1, 10.0
        clip = torch.stack(
            [
                f * cam[:, 0],
                f * cam[:, 1],
                (far + near) / (far - near) * cam[:, 2] - 2 * far * near / (far - near),
                cam[:, 2],
            ],
            -1,
        )
        views_pos.append(clip)
    return torch.stack(views_pos), tri.to(torch.int32)


def time_ms(fn, repeats: int, warmup_s: float) -> float:
    # Warm up for a fixed time rather than a fixed count: on some drivers (ROCm on Windows), newly
    # allocated memory stays slow for several seconds before it settles.
    start = time.perf_counter()
    while True:
        fn()
        torch.cuda.synchronize()
        if time.perf_counter() - start > warmup_s:
            break
    # Median of individually timed calls, so occasional stalls do not skew the result.
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000


def measure(ctx, pos, tri, res: int, repeats: int, warmup_s: float) -> tuple[float, float]:
    """Return forward and forward + backward time in milliseconds."""
    pos = pos.clone().requires_grad_(True)

    def forward():
        with torch.no_grad():
            dr.rasterize(ctx, pos, tri, (res, res))

    def forward_backward():
        rast, rast_db = dr.rasterize(ctx, pos, tri, (res, res))
        (rast[..., :3].sum() + rast_db.sum()).backward()

    return time_ms(forward, repeats, warmup_s), time_ms(forward_backward, repeats, warmup_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["triton", "reference"])
    parser.add_argument("--triangles", nargs="+", type=int, default=[10_000, 100_000, 1_000_000])
    parser.add_argument("--resolutions", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--views", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=float, default=15.0, help="warmup seconds per case")
    parser.add_argument("--memory-fraction", type=float, default=0.5)
    parser.add_argument("--max-reference-work", type=float, default=2e10)
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
                    if backend == "reference" and tri.shape[0] * views * res * res > (
                        args.max_reference_work
                    ):
                        continue
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
