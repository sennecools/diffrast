"""Draw the results of ``bench_pipeline.py --json`` as an SVG chart.

Usage: python benchmarks/plot_pipeline.py results.json chart.svg
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from html import escape

SURFACE = "#121110"
GRID = "#2a2725"
TEXT = "#f0eadf"
MUTED = "#a09b93"
# One hue, lighter for more triangles; the reference backend is the grey baseline.
TRITON = ["#1c5cab", "#5598e7", "#cde2fb"]
REFERENCE = "#8a8580"

WIDTH, HEIGHT = 830, 452
PLOT_TOP, PLOT_BOTTOM = 132, 372
PANELS = (
    ("forward_ms", "forward", 64, 372),
    ("forward_backward_ms", "forward + backward", 474, 782),
)
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def count_label(triangles: int) -> str:
    """Triangle count rounded to a short label; the grid meshes are a little off round numbers."""
    if triangles >= 950_000:
        return f"{triangles / 1e6:.0f}M"
    return f"{round(triangles / 1000, -1 if triangles > 20_000 else 0):.0f}k"


def duration(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f} s"
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"


def text(x, y, body, size=12, fill=MUTED, anchor="start", weight=400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(body)}</text>'
    )


def chart(report: dict, views: int) -> str:
    rows = [r for r in report["results"] if r["views"] == views]
    resolutions = sorted({r["resolution"] for r in rows})
    counts = sorted({r["triangles"] for r in rows if r["backend"] == "triton"})
    series = [("triton", n, TRITON[-len(counts) :][i]) for i, n in enumerate(counts)]
    baseline = sorted({r["triangles"] for r in rows if r["backend"] == "reference"})
    series += [("reference", n, REFERENCE) for n in baseline[:1]]

    values = [r[key] for r in rows for key, *_ in PANELS]
    low = 10 ** math.floor(math.log10(min(values)))
    high = 10 ** math.ceil(math.log10(max(values)))

    def y_of(ms: float) -> float:
        t = (math.log10(ms) - math.log10(low)) / (math.log10(high) - math.log10(low))
        return PLOT_BOTTOM - t * (PLOT_BOTTOM - PLOT_TOP)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}" role="img" '
        f'aria-label="Time per full differentiable render by resolution and triangle count">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="{SURFACE}"/>',
        text(24, 36, f"Full differentiable render, {views} views per call", 16, TEXT, weight=600),
        text(
            24,
            56,
            "rasterize, interpolate, mipmapped texture, antialias. Milliseconds per call, "
            "log scale; lower is better.",
        ),
    ]

    # Legend: a line key and a dot beside each name, in text colours.
    x = 24
    for backend, n, color in series:
        name = (
            f"{count_label(n)} triangles"
            if backend == "triton"
            else (f"reference backend, {count_label(n)}")
        )
        out.append(
            f'<line x1="{x}" y1="80" x2="{x + 22}" y2="80" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round"/><circle cx="{x + 11}" cy="80" r="4" fill="{color}"/>'
        )
        out.append(text(x + 30, 84, name, 12, TEXT))
        x += 52 + 6.6 * len(name)

    for key, title, left, right in PANELS:
        out.append(text(left, PLOT_TOP - 16, title, 13, TEXT, weight=600))
        step = (right - left - 40) / max(len(resolutions) - 1, 1)

        def x_of(res: int, left=left, step=step) -> float:
            return left + 8 + resolutions.index(res) * step

        decade = low
        while decade <= high * 1.0001:
            y = y_of(decade)
            out.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>'
            )
            label = f"{decade:g} ms" if decade < 1000 else f"{decade / 1000:g} s"
            out.append(text(left - 8, y + 4, label, 11, MUTED, "end"))
            decade *= 10
        for res in resolutions:
            out.append(text(x_of(res), PLOT_BOTTOM + 20, f"{res}", 11, MUTED, "middle"))
        centre = (left + right) / 2
        out.append(
            text(centre, PLOT_BOTTOM + 40, "resolution (pixels per side)", 11, MUTED, "middle")
        )

        ends = []
        for backend, n, color in series:
            points = sorted(
                (r["resolution"], r[key])
                for r in rows
                if r["backend"] == backend and r["triangles"] == n
            )
            if not points:
                continue
            path = " ".join(f"{x_of(res):.1f},{y_of(ms):.1f}" for res, ms in points)
            out.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for res, ms in points:
                out.append(
                    f'<circle cx="{x_of(res):.1f}" cy="{y_of(ms):.1f}" r="4" fill="{color}" '
                    f'stroke="{SURFACE}" stroke-width="2"><title>{backend}, {n:,} triangles, '
                    f"{res} px: {ms:.2f} ms</title></circle>"
                )
            res, ms = points[-1]
            ends.append([y_of(ms) + 4, x_of(res) + 10, duration(ms)])

        # Value labels at the line ends, kept a line height apart where the lines come close.
        ends.sort()
        for above, below in itertools.pairwise(ends):
            below[0] = max(below[0], above[0] + 13)
        for y, x, value in ends:
            out.append(text(x, y, value, 11, TEXT))

    footer = f"{report['device']} · torch {report['torch']} · triton {report['triton']}"
    out.append(text(24, HEIGHT - 12, footer, 11, MUTED))
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    with open(args.results) as file:
        report = json.load(file)
    with open(args.output, "w", encoding="utf-8") as file:
        file.write(chart(report, args.views))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
