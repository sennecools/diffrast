"""Recover an environment map from renders of a mirror-like object.

A glossy torus knot reflects its surroundings, which are stored in a cube map: six square faces
that are looked up with a direction instead of texture coordinates. A second cube map starts out
flat grey and is optimised until renders of the knot from random viewpoints match renders made
with the first. The fit sees the knot on black, so everything it learns about the environment
comes from reflections.

    python examples/env_lighting.py --gif env_lighting.gif --out env_lighting.png
    python examples/env_lighting.py --iterations 0 --cover env_lighting_cover.png

The animation shows the target and the fit from a slowly turning camera, and next to them both
cube maps unfolded as a cross: the four sides in a row (-x, +z, +x, -z), with +y above and -y
below +z. Behind the knot each panel shows its own cube map, out of focus; the fit never saw that
backdrop. The still adds the fitted map on a rougher knot. The cover is the knot alone, in a finer
version of the target environment.

``render`` uses the cube map three times. The mirror image comes from a mipmapped lookup along the
reflected view direction, whose screen-space derivatives pick the mip level; ``mip_level_bias``
shifts that level, which blurs the reflection like a rougher surface would. Diffuse light and the
backdrop are lookups that name their mip level directly.

The reflected direction is computed per vertex and interpolated, which is an approximation: on
the default mesh it stays within half a degree of the exact per-pixel direction, on a coarse mesh
reflections bend visibly. The knot does not reflect itself.

The targets come from the same renderer and carry no noise, so the fit ends up close to exact. A
curved mirror shows nearly all of its surroundings from any side, and pixels near its outline
read coarse mip levels that average whole faces, so no texel is left without gradient. What
limits the sharpness of the result is how much of the environment one pixel of the mirror covers.
The fit therefore looks from close by, which halves that footprint to between one and two texels.

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

import diffrast as dr
from common import (
    BACKGROUND,
    MUTED,
    Mesh,
    font,
    look_at,
    orbit,
    panels,
    perspective,
    psnr,
    rgb,
    save_animation,
    text_mask,
    to_clip,
    to_image,
    to_srgb,
    torus_knot,
    vertex_normals,
)

# Per cube face, in the order diffrast stores them: the direction through its centre, and the
# directions in which its columns and rows run (the OpenGL convention).
FACES = (
    ((1, 0, 0), (0, 0, -1), (0, -1, 0)),
    ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    ((0, 0, 1), (1, 0, 0), (0, -1, 0)),
    ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
)
# Row and column of each face in the unfolded cross.
CROSS = ((1, 2), (1, 0), (0, 1), (2, 1), (1, 1), (1, 3))

# The knot mirrors most of the light and scatters a little, tinted like warm clay.
MIRROR = 0.9
DIFFUSE = (0.040, 0.038, 0.036)
# Mip level bias of the rougher knot in the still.
ROUGH = 2.0
# The backdrop is read from the mip level whose faces are this many texels wide.
BACKDROP_FACE = 32
# Radiance of every texel when the fit starts.
GREY = 0.02

# Directions towards the lights of the studio.
SOFTBOX = (0.55, 0.75, 0.55)
STRIP = (-1.0, 0.1, 0.25)
ACCENT_PANEL = (-0.35, -0.6, 0.7)
BACKLIGHT = (-0.3, -0.25, -0.9)

FOV = 30.0
# From this far away the unit sphere, which bounds the knot, fits in a square image.
CAMERA_DISTANCE = 4.2
# One pixel of a curved mirror covers less of the environment from close by.
FIT_DISTANCE = 2.4
HELD_OUT_VIEWS = 16

FIGURE_YAW, FIGURE_PITCH, FIGURE_SWEEP = 0.05, 0.25, 0.7
COVER_SIZE = (1600, 1000)
COVER_YAW, COVER_PITCH, COVER_DISTANCE = 0.3, 0.25, 4.4
# The cover uses an environment with this many times the texels of the fitted one along a side.
COVER_DETAIL = 8


class Camera(NamedTuple):
    """A batch of views: eye positions ``[batch, 3]``, view matrices and one projection."""

    eye: Tensor
    view: Tensor
    proj: Tensor


def render(
    ctx: dr.RasterizeContext,
    mesh: Mesh,
    env: Tensor,
    camera: Camera,
    resolution: tuple[int, int],
    roughness: float = 0.0,
    backdrop: bool = True,
) -> Tensor:
    """Linear images ``[batch, height, width, 3]`` of the knot lit by the cube map ``env``.

    ``roughness`` is added to the mip level of the reflection. Without ``backdrop`` the pixels
    around the knot are black instead of showing the environment.
    """
    pos = to_clip(mesh.positions, camera.proj @ camera.view)
    normals = vertex_normals(mesh.positions, mesh.tri)
    # The direction a mirror sends the eye's ray in, per view and per vertex.
    towards_eye = F.normalize(camera.eye[:, None] - mesh.positions, dim=-1)
    mirrored = 2 * (towards_eye * normals).sum(-1, keepdim=True) * normals - towards_eye
    mip = dr.texture_construct_mip(env, cube_mode=True)

    rast, rast_db = dr.rasterize(ctx, pos, mesh.tri, resolution)
    # A three-channel attribute with all its derivatives is what a cube lookup takes.
    direction, direction_da = dr.interpolate(
        mirrored, rast, mesh.tri, rast_db=rast_db, diff_attrs="all"
    )
    level = torch.ones_like(rast[..., 0])
    reflection = dr.texture(
        env, direction, direction_da, roughness * level, mip=mip, boundary_mode="cube"
    )

    # Diffuse light changes slowly over the surface. It is read once per vertex, with the normals
    # laid out as an image one pixel wide, from the level that has two texels along a face.
    coarse = normals.new_full((1, normals.shape[0], 1), math.log2(env.shape[2]) - 1)
    soft = dr.texture(
        env, normals[None, :, None], mip_level_bias=coarse, mip=mip, boundary_mode="cube"
    )
    diffuse, _ = dr.interpolate(soft[0, :, 0], rast, mesh.tri)
    colour = MIRROR * reflection + diffuse * diffuse.new_tensor(DIFFUSE)

    if backdrop:
        blur = math.log2(env.shape[2] / BACKDROP_FACE) * level
        rays = camera_rays(camera, resolution)
        behind = dr.texture(env, rays, mip_level_bias=blur, mip=mip, boundary_mode="cube")
        colour = torch.where(rast[..., 3:] > 0, colour, behind)
    return dr.antialias(colour, rast, pos, mesh.tri)


def camera_rays(camera: Camera, resolution: tuple[int, int]) -> Tensor:
    """World-space direction ``[batch, height, width, 3]`` of the ray through every pixel."""
    height, width = resolution
    device = camera.eye.device
    y, x = torch.meshgrid(
        (torch.arange(height, device=device) + 0.5) / height * 2 - 1,
        (torch.arange(width, device=device) + 0.5) / width * 2 - 1,
        indexing="ij",
    )
    local = torch.stack([x / camera.proj[0, 0], y / camera.proj[1, 1], -torch.ones_like(x)], -1)
    # The rows of the view rotation are the camera's axes in world space.
    return (local @ camera.view[:, None, :3, :3]).contiguous()


def orbit_cameras(yaw: Tensor, pitch: Tensor, distance: float, aspect: float = 1.0) -> Camera:
    """Cameras on a sphere around the origin, looking at it; angles in radians."""
    eye = orbit(yaw, pitch, distance)
    return Camera(eye, look_at(eye), perspective(FOV, aspect, device=eye.device))


def random_cameras(
    count: int, generator: torch.Generator, device: torch.device, distance: float
) -> Camera:
    """Cameras all around the knot."""
    yaw = torch.rand(count, generator=generator) * 2 * math.pi
    pitch = (torch.rand(count, generator=generator) - 0.5) * 2.4
    return orbit_cameras(yaw.to(device), pitch.to(device), distance)


def cube_directions(size: int, device: torch.device) -> Tensor:
    """Unit direction ``[6, size, size, 3]`` through the centre of every texel of a cube map."""
    faces = torch.tensor(FACES, dtype=torch.float32, device=device)
    centre, along_columns, along_rows = faces[:, :, None, None].unbind(1)
    step = (torch.arange(size, device=device) + 0.5) / size * 2 - 1
    row, column = torch.meshgrid(step, step, indexing="ij")
    directions = centre + column[..., None] * along_columns + row[..., None] * along_rows
    return F.normalize(directions, dim=-1)


def rectangle(
    directions: Tensor, centre: tuple[float, ...], half: tuple[float, float], edge: float
) -> Tensor:
    """Mask ``[..., 1]`` of a rectangular light that faces the origin from the direction ``centre``.

    ``half`` is its half width and half height at distance one, ``edge`` the width over which it
    fades out.
    """
    forward = F.normalize(directions.new_tensor(centre), dim=0)
    right = F.normalize(torch.linalg.cross(directions.new_tensor([0.0, 1.0, 0.0]), forward), dim=0)
    up = torch.linalg.cross(forward, right)
    depth = (directions @ forward).clamp(min=1e-6)
    across = (half[0] - (directions @ right / depth).abs()) / edge + 0.5
    along = (half[1] - (directions @ up / depth).abs()) / edge + 0.5
    mask = across.clamp(0, 1) * along.clamp(0, 1) * (directions @ forward > 0)
    return mask[..., None]


def glow(directions: Tensor, centre: tuple[float, ...], width: float) -> Tensor:
    """Soft spot ``[..., 1]`` around the direction ``centre``; ``width`` is roughly in radians."""
    towards = F.normalize(directions.new_tensor(centre), dim=0)
    return torch.exp(-2 * (1 - directions @ towards) / width**2)[..., None]


def studio(size: int, device: torch.device, supersample: int = 4) -> Tensor:
    """The target environment ``[1, 6, size, size, 3]``, in linear radiance: a dark studio.

    A warm softbox, a cool strip light and a small orange panel are far brighter than one, so
    their reflections stay bright after tone mapping. A grid of ceiling lights and the word
    "diffrast" are the fine detail that shows how sharp the recovered map is.
    """
    fine = size * supersample
    d = cube_directions(fine, device)
    x, y, z = d.unbind(-1)

    def colour(*values: float) -> Tensor:
        return d.new_tensor(values)

    floor = ((0.02 - y) / 0.04).clamp(0, 1)[..., None]
    env = torch.lerp(colour(0.0115, 0.0115, 0.0125), colour(0.0147, 0.0132, 0.0125), floor)
    env = env + torch.exp(-((y / 0.09) ** 2))[..., None] * colour(0.005, 0.0035, 0.0025)
    env = env + glow(d, BACKLIGHT, 0.3) * colour(0.010, 0.0075, 0.006)

    env = env + glow(d, SOFTBOX, 0.5) * colour(0.12, 0.09, 0.06)
    env = env + rectangle(d, SOFTBOX, (0.50, 0.34), 0.10) * colour(6.5, 5.2, 3.9)
    env = env + glow(d, STRIP, 0.6) * colour(0.06, 0.10, 0.18)
    env = env + rectangle(d, STRIP, (0.055, 0.75), 0.03) * colour(2.0, 3.4, 5.5)
    env = env + glow(d, ACCENT_PANEL, 0.4) * colour(0.12, 0.012, 0.004)
    env = env + rectangle(d, ACCENT_PANEL, (0.16, 0.16), 0.03) * colour(2.6, 0.075, 0.02)

    # Four by four lights in one corner of the ceiling, at positions on the plane y = 1.
    u, v = x / y.clamp(min=1e-6), z / y.clamp(min=1e-6)
    spacing, radius = 0.2, 0.045
    to_nearest = torch.hypot(
        u - (u / spacing).round() * spacing, v - (v / spacing).round() * spacing
    )
    lights = ((radius - to_nearest) / 0.02).clamp(0, 1)
    lights = lights * (y > 0) * (u > -0.9) * (u < -0.1) * (v > -0.9) * (v < -0.1)
    env = env + lights[..., None] * colour(6.0, 5.5, 5.0)

    # Faces are stored as seen from outside the cube, which is how a mirror shows them, so the
    # word reads the right way round in reflections. Their rows start at the top.
    word = text_mask("diffrast", fine, fine // 4, fine // 6, device).flip(0)
    top = fine * 5 // 8
    env[4, top : top + word.shape[0]] += word[..., None] * colour(2.5, 2.3, 2.0)

    env = F.avg_pool2d(env.permute(0, 3, 1, 2), supersample).permute(0, 2, 3, 1)
    return env[None].contiguous()


def compress(linear: Tensor) -> Tensor:
    """Radiance squeezed into ``[0, 1)``, so that the lights do not drown out the rest."""
    return linear / (1 + linear)


def unfold(env: Tensor, face: int) -> Image.Image:
    """Cube map as a cross of display values on the canvas colour, ``face`` pixels per face."""
    size = env.shape[2]
    cross = env.new_tensor(BACKGROUND).repeat(3 * size, 4 * size, 1)
    for texels, (row, column) in zip(to_srgb(env[0]), CROSS, strict=True):
        cross[row * size : (row + 1) * size, column * size : (column + 1) * size] = texels
    # to_image expects row 0 at the bottom; the rows of a cube face start at its top.
    return to_image(cross.flip(0), 4 * face)


def figure(
    ctx: dr.RasterizeContext,
    mesh: Mesh,
    target: Tensor,
    env: Tensor,
    step: int,
    turn: float,
    width: int,
    still: bool = False,
) -> Image.Image:
    """Target and fit from one camera, next to both cube maps.

    ``turn`` moves the camera along its sweep, from 0 to 1. The animation has the cube maps in a
    column beside the renders. The ``still`` puts them below, larger, and adds the fitted
    environment on a rougher knot.
    """
    gap, label_size = width // 60, width // 53
    if still:
        # A pixel or two more between the panels lets them fill the width exactly.
        gap += (width - 4 * gap) % 3
        side = (width - 4 * gap) // 3
        face = (width - 3 * gap) // 8
    else:
        # The largest faces for which two crosses and a label are as high as a render.
        face = (width - 8 * gap - 2 * label_size) // 16
        side = (width - 4 * gap - 4 * face) // 2

    yaw = torch.tensor([FIGURE_YAW + FIGURE_SWEEP * (turn - 0.5)], device=env.device)
    camera = orbit_cameras(yaw, torch.full_like(yaw, FIGURE_PITCH), CAMERA_DISTANCE)
    # Rendered at twice the size of a panel; to_image averages it down.
    views = [(target, 0.0), (env, 0.0)] + ([(env, ROUGH)] if still else [])
    with torch.no_grad():
        renders = [
            to_image(to_srgb(render(ctx, mesh, cube, camera, (2 * side, 2 * side), rough)[0]), side)
            for cube, rough in views
        ]
        crosses = [unfold(target, face), unfold(env, face)]
        decibel = psnr(to_srgb(env), to_srgb(target))
    labels = ["target", f"fitted, step {step}", f"fitted, mip level bias +{ROUGH:g}"]
    # The column of the animation is too narrow for more words.
    fitted_cube = "fitted cube map" if still else "fitted"
    cube_labels = ["target cube map", f"{fitted_cube}, psnr {decibel:.1f} dB"]

    if still:
        rows = [
            panels(renders, labels, gap, label_size),
            panels(crosses, cube_labels, gap, label_size),
        ]
        # Every row brings its own margin; overlapping them keeps the gaps between rows equal.
        canvas = Image.new("RGB", (width, rows[0].height + rows[1].height - gap), rgb(BACKGROUND))
        canvas.paste(rows[1], (0, rows[0].height - gap))
        canvas.paste(rows[0], (0, 0))
        return canvas

    # The second cross sits under the first, with its label in between like a panel label.
    column = Image.new("RGB", (crosses[0].width, side), rgb(BACKGROUND))
    column.paste(crosses[0], (0, 0))
    column.paste(crosses[1], (0, side - crosses[1].height))
    label_top = side - crosses[1].height - gap - label_size
    ImageDraw.Draw(column).text(
        (0, label_top), cube_labels[1], fill=rgb(MUTED), font=font(label_size)
    )
    return panels([*renders, column], [*labels[:2], cube_labels[0]], gap, label_size)


def cover(ctx: dr.RasterizeContext, mesh: Mesh, env: Tensor) -> Image.Image:
    """The knot in the target environment, composed like a product shot."""
    width, height = COVER_SIZE
    device = env.device
    camera = orbit_cameras(
        torch.tensor([COVER_YAW], device=device),
        torch.tensor([COVER_PITCH], device=device),
        COVER_DISTANCE,
        width / height,
    )
    with torch.no_grad():
        image = to_srgb(render(ctx, mesh, env, camera, (2 * height, 2 * width)))
    image = F.avg_pool2d(image.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)[0].cpu()
    # Eight bits leave visible steps in the dark backdrop; half a step of noise hides them.
    noise = torch.rand(image.shape, generator=torch.Generator().manual_seed(0)) - 0.5
    return to_image(image + noise / 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--views", type=int, default=4, help="random views per iteration")
    parser.add_argument("--resolution", type=int, default=384, help="size of the views")
    parser.add_argument("--env-size", type=int, default=64, help="texels along a cube face")
    parser.add_argument("--segments", type=int, default=768, help="knot subdivisions along")
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", help="write the optimisation progress to this GIF")
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--gif-width", type=int, default=960)
    parser.add_argument("--out", help="write the result and the target to this PNG")
    parser.add_argument("--out-width", type=int, default=1660)
    parser.add_argument("--cover", help="write the knot in the target environment to this PNG")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    ctx = dr.RasterizeContext(device)
    mesh = torus_knot(segments=args.segments, sides=max(3, args.segments // 8), device=device)
    target = studio(args.env_size, device)
    # Radiance spans three orders of magnitude. Adam takes steps of a fixed size, which in the
    # logarithm of the radiance are steps of a fixed ratio, for dim texels and bright ones alike.
    log_env = torch.full_like(target, math.log(GREY), requires_grad=True)
    optimizer = torch.optim.Adam([log_env], lr=args.lr)
    resolution = (args.resolution, args.resolution)
    reached = torch.zeros_like(target[..., 0], dtype=torch.bool)

    frames = []
    # Most of the change happens early, so the frames crowd towards the first steps, while the
    # camera turns at an even pace.
    last = max(1, args.frames - 1)
    capture = [round(args.iterations * (i / last) ** 3) for i in range(args.frames)]
    for step in range(args.iterations + 1):
        while args.gif and len(frames) < args.frames and capture[len(frames)] == step:
            env = log_env.detach().exp()
            frames.append(figure(ctx, mesh, target, env, step, len(frames) / last, args.gif_width))
        if step == args.iterations:
            break

        camera = random_cameras(args.views, generator, device, FIT_DISTANCE)
        with torch.no_grad():
            reference = render(ctx, mesh, target, camera, resolution, backdrop=False)

        optimizer.zero_grad()
        fitted = render(ctx, mesh, log_env.exp(), camera, resolution, backdrop=False)
        loss = F.mse_loss(compress(fitted), compress(reference))
        loss.backward()
        reached |= log_env.grad.abs().sum(-1) > 0
        optimizer.step()

        if step % 100 == 0 or step == args.iterations - 1:
            decibel = psnr(to_srgb(fitted), to_srgb(reference))
            print(f"step {step:5d}  loss {loss.item():.2e}  psnr {decibel:.2f} dB")

    env = log_env.detach().exp()
    camera = random_cameras(HELD_OUT_VIEWS, generator, device, CAMERA_DISTANCE)
    with torch.no_grad():
        fitted = to_srgb(render(ctx, mesh, env, camera, resolution))
        reference = to_srgb(render(ctx, mesh, target, camera, resolution))
        error = (to_srgb(env) - to_srgb(target)).abs().amax(-1)
    print(
        f"renders   psnr {psnr(fitted, reference):.2f} dB on {HELD_OUT_VIEWS} new views with "
        "their backdrop, in display values"
    )
    print(
        f"cube map  psnr {psnr(to_srgb(env), to_srgb(target)):.2f} dB, "
        f"{(error < 0.05).float().mean().item():.1%} of its texels within 0.05; "
        f"{reached.float().mean().item():.1%} of them received gradient"
    )

    if args.gif and frames:
        save_animation(frames, args.gif, frame_ms=70, dither=False)
        print(f"wrote {args.gif}")
    if args.out:
        figure(ctx, mesh, target, env, args.iterations, 0.5, args.out_width, still=True).save(
            args.out
        )
        print(f"wrote {args.out}")
    if args.cover:
        cover(ctx, mesh, studio(COVER_DETAIL * args.env_size, device, 2)).save(args.cover)
        print(f"wrote {args.cover}")


if __name__ == "__main__":
    main()
