"""Recover a texture from rendered images of a mesh.

A target texture is rendered onto a torus knot from random viewpoints. A second texture, starting
out flat grey, is optimised until its renders match. Only images are compared: the gradient of
their difference reaches the texels through antialiasing and mipmapped texture sampling.

    python examples/texture_fitting.py --gif texture_fitting.gif --out texture_fitting.png

The animation shows the target and the fit from a camera that goes around the knot, a close-up of
the fit, and below them the texture as it fills in. That texture is a strip, with ``u`` along the
knot and ``v`` around the tube. Other meshes work too: ``--mesh sphere``, or ``--mesh model.obj
--texture model.png`` for an OBJ file with texture coordinates.

The targets come from the same renderer and carry no noise, and the cameras reach every part of
the knot, so the fit ends up close to exact. Where a mesh hides part of its surface, or an atlas
has unused space, the texels that no view samples stay grey, apart from a faint tint that reaches
them through the coarser mip levels. The share of texels that came back is printed at the end.

``--gif`` writes the format that its file name ends in: ``.gif``, ``.webp`` or, with ffmpeg on the
PATH, ``.mp4`` and ``.webm``.

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import diffrast as dr
from common import (
    CLAY,
    INK,
    PANEL,
    Mesh,
    downsample,
    from_srgb,
    layout,
    lighting,
    load_obj,
    look_at,
    orbit,
    panels,
    perspective,
    psnr,
    save_animation,
    stack,
    text_mask,
    to_clip,
    to_image,
    to_srgb,
    torus_knot,
    uv_sphere,
    vertex_normals,
)

# Linear albedo of the stripes, and the grey that the fitted texture starts from.
ORANGE = (0.95, 0.07, 0.02)
GREY = 0.18

# Directions towards the lights, in camera space.
KEY_LIGHT = (0.4, 0.7, 0.6)
FILL_LIGHT = (-0.7, -0.3, 0.4)

FOV = 30.0
# From this far away the unit sphere, which bounds every mesh, fits in a square image.
CAMERA_DISTANCE = 4.2
# Texture width in heights, for texels that come out roughly square on the surface.
TEXTURE_ASPECT = {"knot": 8, "sphere": 2}

# The figures look down on the mesh from a camera that goes around it. The knot lies flat, so
# that camera can come closer; raising the knot a little centres it in the view.
FIGURE_ASPECT = 1.5
FIGURE_YAW, FIGURE_PITCH = 0.5, 0.45
FIGURE_DISTANCE = {"knot": 3.2}
KNOT_LIFT = 0.12
# The close-up looks at the surface point with these texture coordinates: on the knot, a spot
# next to the middle of a word of the built-in texture.
CLOSE_UP_UV, CLOSE_UP_ZOOM = (0.37, 0.7), 1.7


def load_mesh(name: str, segments: int, device: torch.device) -> Mesh:
    """The knot, a sphere or the OBJ file ``name``, with texture coordinates."""
    if name == "knot":
        # flip_v keeps the lettering of the texture from showing as its mirror image.
        knot = torus_knot(
            segments=segments, sides=max(3, segments // 8), device=device, flat=True, flip_v=True
        )
        lift = knot.positions.new_tensor([0.0, KNOT_LIFT, 0.0])
        return knot._replace(positions=knot.positions + lift)
    if name == "sphere":
        return uv_sphere(max(2, segments // 2), segments, device)
    mesh = load_obj(name, device)
    if mesh.uvs is None:
        raise SystemExit(f"{name} has no texture coordinates.")
    return mesh


def target_texture(height: int, width: int, device: torch.device) -> Tensor:
    """Diagonal stripes, a fine checker and the word "diffrast", in linear colour.

    The stripes are readable long before the optimisation has converged; the checker and the
    letters show whether fine detail comes back. Returns ``[1, height, width, 3]``.
    """
    y, x = torch.meshgrid(
        (torch.arange(height, device=device) + 0.5) / height,
        (torch.arange(width, device=device) + 0.5) / height,
        indexing="ij",
    )
    clay, ink, orange = (torch.tensor(c, device=device) for c in (CLAY, INK, ORANGE))

    def blend(colour: Tensor, other: Tensor, mask: Tensor) -> Tensor:
        return torch.lerp(colour, other, mask[..., None].float())

    # From bottom to top: stripes, a checker, the word on a clay band, a checker, stripes. The
    # band sits above the middle because that side of the knot's tube faces outwards.
    low, high = 16 / 32, 25 / 32
    edge = torch.minimum((y - low).abs(), (y - high).abs())
    stripes = (4 * x + 2 * y) % 1 < 0.5
    checker = ((32 * x).floor() + (32 * y).floor()) % 2 == 0
    colour = blend(ink.expand(height, width, 3), orange, stripes)
    colour = blend(colour, blend(clay.expand_as(colour), ink, checker), edge < 2 / 32)
    colour = blend(colour, clay, (y > low) & (y < high))
    colour = blend(colour, ink, edge < 1 / 64)

    # One word per stretch of texture that is about twice as long as it is high.
    words = max(2, round(width / height / 2))
    rows = slice(round(low * height), round(high * height))
    size = max(4, round(min(0.27 * height, 0.21 * width / words)))
    word = text_mask("diffrast", width // words, rows.stop - rows.start, size, device)
    word = word.repeat(1, words)
    colour[rows, : word.shape[1]] = blend(colour[rows, : word.shape[1]], ink, word)
    return colour[None].contiguous()


def load_texture(path: str, height: int, device: torch.device) -> Tensor:
    """Image file as a linear texture ``[1, height, width, 3]`` with row 0 at the bottom."""
    image = Image.open(path).convert("RGB")
    width = max(1, round(height * image.width / image.height))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    texels = torch.from_numpy(np.array(image)).to(device).float().flip(0) / 255
    return (texels**2.2)[None].contiguous()


def render(
    ctx: dr.RasterizeContext,
    mesh: Mesh,
    tex: Tensor,
    view: Tensor,
    proj: Tensor,
    resolution: tuple[int, int],
) -> Tensor:
    """Linear images ``[batch, height, width, 3]`` of the textured mesh, one per view matrix."""
    pos = to_clip(mesh.positions, proj @ view)
    # The lights are fixed to the camera, so every view looks at a lit surface.
    normals = vertex_normals(mesh.positions, mesh.tri) @ view[:, :3, :3].transpose(1, 2)

    rast, rast_db = dr.rasterize(ctx, pos, mesh.tri, resolution)
    uv, uv_da = dr.interpolate(mesh.uvs, rast, mesh.tri_uv, rast_db=rast_db, diff_attrs="all")
    normal, _ = dr.interpolate(normals, rast, mesh.tri)
    albedo = dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear", boundary_mode="wrap")

    shaded = albedo * lighting(normal, KEY_LIGHT, FILL_LIGHT)
    covered = rast[..., 3:] > 0
    colour = torch.where(covered, shaded, from_srgb(PANEL, pos.device))
    return dr.antialias(colour, rast, pos, mesh.tri)


def random_views(
    count: int, generator: torch.Generator, device: torch.device
) -> tuple[Tensor, Tensor]:
    """View matrices ``[count, 4, 4]`` of cameras all around the mesh, and their projection."""
    yaw = torch.rand(count, generator=generator) * 2 * math.pi
    pitch = (torch.rand(count, generator=generator) - 0.5) * 2.4
    view = look_at(orbit(yaw.to(device), pitch.to(device), CAMERA_DISTANCE))
    return view, perspective(FOV, 1.0, device=device)


def figure_views(mesh: Mesh, turn: float, distance: float) -> tuple[Tensor, Tensor]:
    """View and projection matrices ``[2, 4, 4]`` for the figures: an overview and a close-up.

    ``turn`` is how far the overview camera is on its way around the mesh, from 0 to 1. The
    close-up stays where that round starts, turned towards the surface point at ``CLOSE_UP_UV``
    and zoomed in.
    """
    device = mesh.positions.device
    # Positions and texture coordinates pair up corner by corner of the triangles.
    offset = mesh.uvs[mesh.tri_uv.long()] - mesh.uvs.new_tensor(CLOSE_UP_UV)
    focus = mesh.positions[mesh.tri.flatten()[offset.norm(dim=-1).argmin()]]
    yaw = FIGURE_YAW + 2 * math.pi * torch.tensor([turn, 0.0], device=device)
    eye = orbit(yaw, torch.full_like(yaw, FIGURE_PITCH), distance)
    view = look_at(eye, torch.stack([torch.zeros_like(focus), focus]))
    fovs = (FOV, FOV / CLOSE_UP_ZOOM)
    proj = torch.stack([perspective(fov, FIGURE_ASPECT, device=device) for fov in fovs])
    return view, proj


def figure(
    ctx: dr.RasterizeContext,
    mesh: Mesh,
    target: Tensor,
    tex: Tensor,
    distance: float,
    step: int,
    turn: float,
    width: int,
    compare: bool = False,
) -> Image.Image:
    """The target, the fit and a close-up of the fit in a row, above the texture being fitted.

    ``compare`` adds the target texture, for a still that has to stand on its own.
    """
    gap, panel, label_size = layout(width, 3)
    # Rendered at twice the size of a panel and averaged down.
    resolution = (2 * round(panel / FIGURE_ASPECT), 2 * panel)
    view, proj = figure_views(mesh, turn, distance)
    with torch.no_grad():
        wanted = render(ctx, mesh, target, view[:1], proj[:1], resolution)
        fitted = render(ctx, mesh, tex, view, proj, resolution)
    renders = torch.cat([wanted, fitted])
    images = [to_image(image) for image in downsample(to_srgb(renders), 2)]
    labels = ["target", f"fitted, step {step}", "fitted, close-up"]
    rows = [panels(images, labels, gap, label_size)]

    textures = [(tex, f"fitted texture, psnr {psnr(tex, target):.1f} dB")]
    if compare:
        textures.insert(0, (target, "target texture"))
    # A long strip spans the figure; a squarer texture is as high as the renders.
    aspect = tex.shape[2] / tex.shape[1]
    strip = min(rows[0].width - 2 * gap, round(images[0].height * aspect))
    for texels, label in textures:
        rows.append(panels([to_image(to_srgb(texels[0]), strip)], [label], gap, label_size))
    return stack(rows, overlap=gap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", default="knot", help="knot, sphere or the path of an OBJ file")
    parser.add_argument("--texture", help="image to recover instead of the built-in texture")
    parser.add_argument("--iterations", type=int, default=250, help="optimisation steps")
    parser.add_argument("--views", type=int, default=4, help="random views per iteration")
    parser.add_argument("--resolution", type=int, default=1024, help="size of the views")
    parser.add_argument("--texture-size", type=int, default=256, help="texture height")
    parser.add_argument("--segments", type=int, default=384, help="segments of the knot or sphere")
    parser.add_argument("--lr", type=float, default=0.03, help="learning rate")
    parser.add_argument("--seed", type=int, default=0, help="picks the random views")
    parser.add_argument(
        "--gif",
        help="write the optimisation progress to this animation; "
        "the extension picks the format: .gif, .webp, .mp4 or .webm",
    )
    parser.add_argument("--frames", type=int, default=48, help="frames of the animation")
    parser.add_argument("--gif-width", type=int, default=960, help="width of the animation")
    parser.add_argument("--out", help="write the result and the target to this PNG")
    parser.add_argument("--out-width", type=int, default=1660, help="width of the PNG")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    ctx = dr.RasterizeContext(device)
    mesh = load_mesh(args.mesh, args.segments, device)
    if args.texture:
        target = load_texture(args.texture, args.texture_size, device)
    else:
        width = TEXTURE_ASPECT.get(args.mesh, 1) * args.texture_size
        target = target_texture(args.texture_size, width, device)
    tex = torch.full_like(target, GREY, requires_grad=True)
    optimiser = torch.optim.Adam([tex], lr=args.lr)
    resolution = (args.resolution, args.resolution)
    distance = FIGURE_DISTANCE.get(args.mesh, CAMERA_DISTANCE)
    draw = partial(figure, ctx, mesh, target, tex, distance)

    frames = []
    # Most of the change happens early, so the frames crowd towards the first steps, while the
    # camera goes around the mesh at an even pace.
    last = max(1, args.frames - 1)
    capture = [round(args.iterations * (i / last) ** 3) for i in range(args.frames)]
    for step in range(args.iterations + 1):
        while args.gif and len(frames) < args.frames and capture[len(frames)] == step:
            frames.append(draw(step, len(frames) / last, args.gif_width))
        if step == args.iterations:
            break

        view, proj = random_views(args.views, generator, device)
        with torch.no_grad():
            reference = render(ctx, mesh, target, view, proj, resolution)

        optimiser.zero_grad()
        fitted = render(ctx, mesh, tex, view, proj, resolution)
        loss = F.mse_loss(fitted, reference)
        loss.backward()
        optimiser.step()
        with torch.no_grad():
            tex.clamp_(0, 1)

        if step % 100 == 0 or step == args.iterations - 1:
            decibel = psnr(fitted, reference)
            print(f"step {step:5d}  loss {loss.item():.6f}  psnr {decibel:.2f} dB")

    view, proj = random_views(args.views, generator, device)
    with torch.no_grad():
        fitted = render(ctx, mesh, tex, view, proj, resolution)
        reference = render(ctx, mesh, target, view, proj, resolution)
        close = ((tex - target).abs().amax(-1) < 0.02).float().mean().item()
    print(f"renders  psnr {psnr(fitted, reference):.2f} dB on {args.views} new views")
    print(f"texture  psnr {psnr(tex, target):.2f} dB, {close:.1%} of its texels within 0.02")

    if args.gif:
        # Most of every frame changes from one to the next. Dithering makes a GIF of them about
        # a fifth larger, for little gain on a surface this busy.
        save_animation(frames, args.gif, frame_ms=80, dither=False)
        print(f"wrote {args.gif}")
    if args.out:
        draw(args.iterations, 0.0, args.out_width, compare=True).save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
