"""Recover a texture from rendered images of a sphere.

A target texture is rendered onto a shaded sphere from random viewpoints. A second texture, starting
out flat grey, is optimized so that its renders match, using gradients that flow back through
antialiasing, texture sampling with mipmaps, interpolation and rasterization.

    python examples/texture_fitting.py --gif docs/images/texture_fitting.gif

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import diffrast as dr


def uv_sphere(rings: int, segments: int, device: torch.device):
    """Sphere with separate position and texture-coordinate index buffers.

    Positions are shared across the texture seam and at the poles, so ``tri_pos`` describes a
    closed surface; ``tri_uv`` indexes a grid of texture coordinates that is cut along the seam.
    """
    theta = torch.linspace(0, math.pi, rings + 1, device=device)
    phi = torch.linspace(0, 2 * math.pi, segments + 1, device=device)

    inner = theta[1:-1, None]
    ring_xyz = torch.stack(
        [
            torch.sin(inner) * torch.cos(phi[None, :-1]),
            torch.cos(inner).expand(-1, segments),
            -torch.sin(inner) * torch.sin(phi[None, :-1]),
        ],
        -1,
    ).reshape(-1, 3)
    poles = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], device=device)
    positions = torch.cat([poles[:1], ring_xyz, poles[1:]])

    u, v = torch.meshgrid(phi / (2 * math.pi), 1 - theta / math.pi, indexing="xy")
    uvs = torch.stack([u, v], -1).reshape(-1, 2)

    def pos_index(r, s):
        if r == 0:
            return 0
        if r == rings:
            return positions.shape[0] - 1
        return 1 + (r - 1) * segments + s % segments

    def uv_index(r, s):
        return r * (segments + 1) + s

    tri_pos, tri_uv = [], []
    for r in range(rings):
        for s in range(segments):
            a, b, c, d = (r, s), (r + 1, s), (r + 1, s + 1), (r, s + 1)
            if r == 0:
                cell = [(a, b, c)]  # a and d are the north pole
            elif r == rings - 1:
                cell = [(a, b, d)]  # b and c are the south pole
            else:
                cell = [(a, b, c), (a, c, d)]
            for corners in cell:
                tri_pos.append([pos_index(*corner) for corner in corners])
                tri_uv.append([uv_index(*corner) for corner in corners])
    tri_pos = torch.tensor(tri_pos, dtype=torch.int32, device=device)
    tri_uv = torch.tensor(tri_uv, dtype=torch.int32, device=device)
    return positions, tri_pos, uvs, tri_uv


def target_texture(size: int, device: torch.device) -> torch.Tensor:
    """Colourful bands, a checkerboard and a word, so that progress is easy to see."""
    v, u = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, 2 * size, device=device),
        indexing="ij",
    )
    hue = 2 * math.pi * u
    color = torch.stack(
        [
            0.5 + 0.5 * torch.cos(hue),
            0.5 + 0.5 * torch.cos(hue - 2.1),
            0.5 + 0.5 * torch.cos(hue - 4.2),
        ],
        -1,
    )
    checker = ((u * 24).floor() + (v * 12).floor()) % 2
    color = torch.where(
        ((v - 0.5).abs() > 0.22)[..., None], color * (0.55 + 0.45 * checker[..., None]), color
    )

    image = Image.new("L", (2 * size, size), 0)
    font = ImageFont.load_default(size=size // 5)
    draw = ImageDraw.Draw(image)
    for offset in (0.25, 0.75):
        draw.text((offset * 2 * size, size / 2), "diffrast", fill=255, font=font, anchor="mm")
    text = torch.from_numpy(np.asarray(image)).to(device).float().flip(0) / 255
    color = color * (1 - text[..., None]) + 0.05 * text[..., None]
    return color[None].contiguous()


def rotation(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Rotation matrices that turn the sphere by ``yaw`` around y and ``pitch`` around x."""
    cy, sy, cp, sp = torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).view(-1, 3, 3)
    rx = torch.stack([one, zero, zero, zero, cp, -sp, zero, sp, cp], -1).view(-1, 3, 3)
    return rx @ ry


def render(ctx, tex, mesh, turn, resolution, light):
    """Shaded, antialiased renders of the textured sphere, one per rotation."""
    positions, tri_pos, uvs, tri_uv = mesh
    camera = positions @ turn.transpose(1, 2)
    distance, focal, near, far = 3.0, 2.2, 0.1, 10.0
    z = camera[..., 2] - distance
    pos = torch.stack(
        [
            focal * camera[..., 0],
            focal * camera[..., 1],
            -(far + near) / (far - near) * z - 2 * far * near / (far - near),
            -z,
        ],
        -1,
    )

    rast, rast_db = dr.rasterize(ctx, pos, tri_pos, resolution)
    uv, uv_da = dr.interpolate(uvs, rast, tri_uv, rast_db=rast_db, diff_attrs="all")
    normal, _ = dr.interpolate(camera, rast, tri_pos)
    albedo = dr.texture(tex, uv, uv_da, filter_mode="linear-mipmap-linear", boundary_mode="wrap")

    normal = torch.nn.functional.normalize(normal, dim=-1)
    shade = 0.35 + 0.65 * (normal @ light).clamp(min=0)[..., None]
    mask = (rast[..., 3:] > 0).float()
    color = albedo * shade * mask + 0.92 * (1 - mask)
    return dr.antialias(color.contiguous(), rast, pos, tri_pos)


def to_image(image: torch.Tensor, size: int) -> Image.Image:
    """Tensor in ``[height, width, 3]`` with row 0 at the bottom to a PIL image."""
    array = (image.detach().flip(0).clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(array).resize((size, round(size * array.shape[0] / array.shape[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--views", type=int, default=6, help="random views per iteration")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--texture-size", type=int, default=256, help="texture height")
    parser.add_argument("--segments", type=int, default=96, help="sphere subdivisions around")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--gif", help="write the optimization progress to this GIF")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    ctx = dr.RasterizeContext(device)
    mesh = uv_sphere(args.segments // 2, args.segments, device)
    light = torch.nn.functional.normalize(torch.tensor([0.4, 0.6, 0.7], device=device), dim=0)
    target = target_texture(args.texture_size, device)
    tex = torch.full_like(target, 0.5, requires_grad=True)
    optimizer = torch.optim.Adam([tex], lr=args.lr)
    resolution = (args.resolution, args.resolution)

    frames = []
    # Most of the change happens early, so frames are spaced quadratically in time.
    capture = {round(args.iterations * (i / args.frames) ** 2) for i in range(args.frames + 1)}
    for step in range(args.iterations + 1):
        if args.gif and step in capture:
            frames.append(progress_frame(ctx, target, tex, mesh, step, args, light))
        if step == args.iterations:
            break

        yaw = torch.rand(args.views, device=device) * 2 * math.pi
        pitch = (torch.rand(args.views, device=device) - 0.5) * 1.6
        turn = rotation(yaw, pitch)
        with torch.no_grad():
            reference = render(ctx, target, mesh, turn, resolution, light)

        optimizer.zero_grad()
        fitted = render(ctx, tex, mesh, turn, resolution, light)
        loss = torch.nn.functional.mse_loss(fitted, reference)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            tex.clamp_(0, 1)

        if step % 100 == 0 or step == args.iterations - 1:
            psnr = -10 * math.log10(loss.item())
            print(f"step {step:5d}  loss {loss.item():.6f}  psnr {psnr:.2f} dB")

    if args.gif:
        save_gif(frames, args.gif)
        print(f"wrote {args.gif}")


def save_gif(frames: list[Image.Image], path: str) -> None:
    """One palette for all frames, built from a sample of them, with dithering against it."""
    sample = frames[:: max(1, len(frames) // 6)]
    montage = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, frame in enumerate(sample):
        montage.paste(frame, (0, i * frame.height))
    palette = montage.quantize(colors=256)
    frames = [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for frame in frames
    ]
    durations = [80] * (len(frames) - 1) + [2500]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def progress_frame(ctx, target, tex, mesh, step, args, light) -> Image.Image:
    """Target render, current render and current texture side by side, from a turning view."""
    yaw = torch.tensor([0.6 + step / args.iterations * 2 * math.pi], device=tex.device)
    turn = rotation(yaw, torch.tensor([0.35], device=tex.device))
    with torch.no_grad():
        images = [render(ctx, t, mesh, turn, (256, 256), light)[0] for t in (target, tex)]
    panel = 200
    canvas = Image.new("RGB", (panel * 4 + 40, panel + 36), (255, 255, 255))
    canvas.paste(to_image(images[0], panel), (10, 26))
    canvas.paste(to_image(images[1], panel), (panel + 20, 26))
    canvas.paste(to_image(tex[0], 2 * panel), (2 * panel + 30, 26))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((10, 4), "target", fill=(40, 40, 40), font=font)
    draw.text((panel + 20, 4), f"fitted, step {step}", fill=(40, 40, 40), font=font)
    draw.text((2 * panel + 30, 4), "fitted texture", fill=(40, 40, 40), font=font)
    return canvas


if __name__ == "__main__":
    main()
