"""Show what each diffrast op returns, on one object.

A torus knot is rendered with one chain of ops, and every intermediate result becomes a panel of
the figure: triangle ids, barycentrics and depth from ``rasterize``; texture coordinates, normals
and screen-space derivatives (shown as the mip level they select) from ``interpolate``; the
sampled colours from ``texture``; and the shaded image after ``antialias``. The mesh is coarse on
purpose, so that single triangles stay visible.

    python examples/gbuffer.py --out gbuffer.png

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

import diffrast as dr
from common import (
    ACCENT,
    PANEL,
    Mesh,
    downsample,
    look_at,
    orbit,
    perspective,
    titled_panels,
    to_clip,
    to_image,
    to_srgb,
    torus_knot,
    vertex_normals,
)

MIP_COLOURS = ((0.12, 0.16, 0.30), (0.20, 0.50, 0.55), (0.85, 0.76, 0.52), (0.97, 0.94, 0.88))


class Buffers(NamedTuple):
    """Everything the chain of ops produces for one view, each ``[1, height, width, channels]``."""

    rast: Tensor
    uv: Tensor
    uv_da: Tensor
    normal: Tensor
    albedo: Tensor
    image: Tensor


def render(
    ctx: dr.RasterizeContext, mesh: Mesh, tex: Tensor, eye: Tensor, resolution: tuple[int, int]
) -> Buffers:
    """Run the whole chain once and keep every intermediate result."""
    height, width = resolution
    mvp = perspective(30, width / height, device=eye.device) @ look_at(eye)
    pos = centre_in_frame(to_clip(mesh.positions, mvp))
    normals = vertex_normals(mesh.positions, mesh.tri)

    rast, rast_db = dr.rasterize(ctx, pos, mesh.tri, resolution)
    uv, uv_da = dr.interpolate(mesh.uvs, rast, mesh.tri_uv, rast_db=rast_db, diff_attrs="all")
    normal, _ = dr.interpolate(normals, rast, mesh.tri)
    albedo = dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear")

    normal = F.normalize(normal, dim=-1)
    shaded = on_panel(to_srgb(albedo * irradiance(normal)), rast)
    image = dr.antialias(shaded, rast, pos, mesh.tri)
    return Buffers(rast, uv, uv_da, normal, albedo, image)


def centre_in_frame(pos: Tensor, fill: float = 0.82) -> Tensor:
    """Shift and scale clip-space positions so the object spans ``fill`` of the image."""
    ndc = pos[..., :2] / pos[..., 3:]
    low, high = ndc.amin(1, keepdim=True), ndc.amax(1, keepdim=True)
    scale = 2 * fill / (high - low).amax()
    xy = (pos[..., :2] - (low + high) / 2 * pos[..., 3:]) * scale
    return torch.cat([xy, pos[..., 2:]], -1)


def irradiance(normal: Tensor) -> Tensor:
    """Linear light on the surface: a warm key from the upper front, ambient and a cool fill."""
    key = F.normalize(normal.new_tensor([0.45, 0.85, 0.55]), dim=0)
    fill = F.normalize(normal.new_tensor([-0.8, 0.2, 0.4]), dim=0)
    warm = normal.new_tensor([1.0, 0.96, 0.9])
    cool = normal.new_tensor([0.55, 0.65, 1.0])
    diffuse = (normal @ key).clamp(min=0)[..., None]
    wrap = (0.5 + 0.5 * (normal @ fill))[..., None]
    return 0.22 + 0.75 * warm * diffuse + 0.08 * cool * wrap


def on_panel(colours: Tensor, rast: Tensor) -> Tensor:
    """``colours`` where a triangle was drawn and the panel colour elsewhere."""
    return torch.where(rast[..., 3:] > 0, colours, colours.new_tensor(PANEL))


def stripe_texture(size: int, device: torch.device) -> Tensor:
    """Linear-colour texture, ``8 * size`` wide: a striped band between two thin lines on cream."""
    v, u = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, 8 * size, device=device),
        indexing="ij",
    )
    cream = torch.tensor((0.85, 0.79, 0.70), device=device)
    orange = torch.tensor(ACCENT, device=device) ** 2.2
    charcoal = torch.tensor((0.05, 0.046, 0.043), device=device)

    across = (v - 0.5).abs()[..., None]
    stripes = ((u * 48 + v * 2) % 1 < 0.5)[..., None]
    colour = torch.where(across < 0.11, torch.where(stripes, orange, charcoal), cream)
    colour = torch.where((across > 0.15) & (across < 0.175), charcoal, colour)
    return colour[None].contiguous()


def id_colours(rast: Tensor, triangles: int, seed: int) -> Tensor:
    """A fixed pseudo-random hue per triangle. ``rast[..., 3]`` holds the triangle id plus one."""
    generator = torch.Generator().manual_seed(seed)
    hue = torch.rand(triangles + 1, 1, generator=generator)
    value = 0.75 + 0.25 * torch.rand(triangles + 1, 1, generator=generator)
    phase = torch.tensor([0.0, 1 / 3, 2 / 3])
    table = value * (0.62 + 0.3 * torch.cos(2 * math.pi * (hue + phase)))
    return table.to(rast.device)[rast[..., 3].long()]


def barycentric_colours(rast: Tensor) -> Tensor:
    """The weights of the three corners as red, green and blue; ``rast`` stores the first two."""
    u, v = rast[..., 0], rast[..., 1]
    return torch.stack([u, v, 1 - u - v], -1)


def depth_shades(rast: Tensor) -> Tensor:
    """``z / w`` stretched over the range the object covers, with near surfaces bright."""
    z = rast[..., 2:3]
    drawn = z[rast[..., 3:] > 0]
    near, far = drawn.min(), drawn.max()
    return (0.12 + 0.88 * (far - z) / (far - near)).expand(-1, -1, -1, 3)


def mip_level(uv_da: Tensor, tex: Tensor) -> Tensor:
    """The level of detail that ``texture`` derives from ``uv_da``.

    ``uv_da`` holds how far ``uv`` moves from one pixel to the next along x and along y. Measured
    in texels, the longer of those two steps is the footprint of the pixel, and its base-two
    logarithm is the mip level whose texels are about as large as the pixel.
    """
    du_dx, du_dy, dv_dx, dv_dy = uv_da.unbind(-1)
    height, width = tex.shape[1:3]
    step_x = torch.hypot(du_dx * width, dv_dx * height)
    step_y = torch.hypot(du_dy * width, dv_dy * height)
    return torch.log2(torch.maximum(step_x, step_y).clamp(min=1e-8)).clamp(min=0)


def mip_colours(level: Tensor) -> Tensor:
    """Blend through ``MIP_COLOURS``, which has one colour per whole level."""
    stops = level.new_tensor(MIP_COLOURS)
    level = level.clamp(0, len(MIP_COLOURS) - 1 - 1e-4)
    lower = level.floor().long()
    blend = (level - lower)[..., None]
    return torch.lerp(stops[lower], stops[lower + 1], blend)


def figure_cells(
    buffers: Buffers, tex: Tensor, triangles: int, seed: int
) -> list[tuple[str, str, Tensor]]:
    """Title, origin and display colours of every panel, in the order of the pipeline."""
    rast, uv = buffers.rast, buffers.uv
    uv_colours = torch.cat([uv, torch.full_like(uv[..., :1], 0.25)], -1)
    levels = mip_colours(mip_level(buffers.uv_da, tex))
    cells = [
        ("triangle id", "rasterize · rast[..., 3]", id_colours(rast, triangles, seed)),
        ("barycentrics", "rasterize · rast[..., :2]", barycentric_colours(rast)),
        ("depth, near is bright", "rasterize · rast[..., 2]", depth_shades(rast)),
        ("texture coordinates", "interpolate · uv", uv_colours),
        ("normals", "interpolate · normal", buffers.normal * 0.5 + 0.5),
        (f"mip level, 0 dark to {len(MIP_COLOURS) - 1} pale", "interpolate · uv_da", levels),
        ("texture", "texture · albedo", to_srgb(buffers.albedo)),
    ]
    cells = [(title, origin, on_panel(colours, rast)) for title, origin, colours in cells]
    return [*cells, ("shaded and antialiased", "antialias · image", buffers.image)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--panel", type=int, default=390, help="panel size in pixels")
    parser.add_argument(
        "--supersample", type=int, default=2, help="render at this multiple and average down"
    )
    parser.add_argument("--segments", type=int, default=112, help="segments along the knot")
    parser.add_argument("--sides", type=int, default=10, help="segments around the tube")
    parser.add_argument("--texture-size", type=int, default=512, help="texture height")
    parser.add_argument("--seed", type=int, default=0, help="picks the triangle id colours")
    parser.add_argument("--out", help="write the figure to this PNG")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ctx = dr.RasterizeContext(device)
    mesh = torus_knot(segments=args.segments, sides=args.sides, device=device)
    tex = stripe_texture(args.texture_size, device)
    eye = orbit(torch.tensor([0.55], device=device), torch.tensor([0.36], device=device), 4.1)
    side = args.panel * args.supersample

    with torch.no_grad():
        buffers = render(ctx, mesh, tex, eye, (side, side))
        cells = figure_cells(buffers, tex, mesh.tri.shape[0], args.seed)

    drawn = buffers.rast[..., 3] > 0
    visible = buffers.rast[..., 3][drawn].unique().numel()
    level = mip_level(buffers.uv_da, tex)[drawn]
    print(f"{mesh.tri.shape[0]} triangles, {visible} of them visible, at {side} x {side} px")
    print(f"the knot covers {drawn.float().mean().item():.1%} of the image")
    print(f"mip level: median {level.median().item():.2f}, highest {level.max().item():.2f}")

    if args.out:
        titles, origins, colours = zip(*cells, strict=True)
        # Panels are rendered larger and averaged down, which also smooths the data panels.
        pictures = [to_image(downsample(panel, args.supersample)[0]) for panel in colours]
        titled_panels(pictures, titles, origins, columns=4).save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
