"""Order-independent transparency with depth peeling.

A glass torus knot with an opaque sphere at its centre. ``dr.DepthPeeler`` returns one depth layer
per call: first the nearest surface in every pixel, then the surface behind it, and so on. A layer
has the same format as the output of ``dr.rasterize``, so each one is shaded with the ordinary ops,
and the layers are blended front to back. No triangle is ever sorted, and it does not matter that
the sphere pokes through the glass.

    python examples/depth_peeling.py --out glass.png --layers layers.png --gif layers.gif

Without arguments it only prints how much of the image each layer covers. ``--check-gradients``
backpropagates through the blended image and shows how far into the layers the gradients reach.

The glass is a stylised thin-wall model: it mirrors one light and dims what is behind it, more so
at grazing angles, but it does not refract.

``--gif`` writes the format that its file name ends in: ``.gif``, ``.webp`` or, with ffmpeg on the
PATH, ``.mp4`` and ``.webm``.

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import diffrast as dr
from common import (
    ACCENT,
    BACKGROUND,
    PANEL,
    downsample,
    from_srgb,
    icosphere,
    layout,
    look_at,
    orbit,
    panels,
    perspective,
    save_animation,
    to_clip,
    to_image,
    to_srgb,
    torus_knot,
    vertex_normals,
)

CORE_RADIUS = 0.42
# Turn of the knot at which the camera sees its three lobes apart.
KNOT_TURN = 0.65
CAMERA_PITCH = 1.05
CAMERA_DISTANCE = 4.3
# Perspective enlarges the near half of the knot, so the camera aims a little in front of the
# knot's centre to centre it in the image.
CAMERA_TARGET = (0.0, 0.0, 0.1)
FOV = 30.0

# Direction towards the key light, and the light on a surface that is turned away from it.
KEY_LIGHT = (0.55, 0.75, -0.05)
AMBIENT = 0.22
# The key light is a softbox, which reflections show as a disc: its brightness, and the cosines
# of the angle from its centre between which its edge fades in.
SOFTBOX = 18.0
SOFTBOX_EDGE = (0.97, 0.99)
# A smooth surface mirrors this fraction of its surroundings head-on and all of them edge-on.
REFLECTANCE = 0.04
# Cool light from all around, which only shows where the glass mirrors strongly: at its rim.
RIM_LIGHT = (0.32, 0.52, 0.76)
GLASS_TINT = (0.25, 0.55, 1.0)
# Fraction of the light that one wall of the knot holds back when it is seen head-on.
GLASS_OPACITY = 0.025
# Linear brightness that marks the outline of the nearest layer in the panels of later layers.
OUTLINE = 0.006
FRAME_MS = 70


class Scene(NamedTuple):
    """Glass knot and opaque sphere in one mesh; the knot's triangles come first."""

    positions: Tensor
    tri: Tensor
    knot_triangles: int
    core_colour: Tensor


class Camera(NamedTuple):
    """Position of the eye and the matrix that takes world space to clip space."""

    eye: Tensor
    matrix: Tensor


class Render(NamedTuple):
    """Blended image in display values, and per layer its linear RGBA and the fraction it covers."""

    image: Tensor
    layers: list[Tensor]
    coverage: list[float]


def build_scene(segments: int, subdivisions: int, device: torch.device) -> Scene:
    """Torus knot lying flat around the y axis, with a sphere in the middle."""
    knot = torus_knot(segments=segments, sides=max(3, segments // 8), device=device, flat=True)
    core = icosphere(subdivisions, device)
    positions = torch.cat([knot.positions, CORE_RADIUS * core.positions])
    tri = torch.cat([knot.tri, core.tri + knot.positions.shape[0]])
    core_colour = torch.tensor(ACCENT, device=device) ** 2.2
    return turned(Scene(positions, tri, knot.tri.shape[0], core_colour), KNOT_TURN)


def turned(scene: Scene, angle: float) -> Scene:
    """The scene rotated around the y axis, as on a turntable."""
    c, s = math.cos(angle), math.sin(angle)
    rotation = scene.positions.new_tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return scene._replace(positions=scene.positions @ rotation.T)


def make_camera(aspect: float, device: torch.device) -> Camera:
    """Camera that looks down on the knot from the front."""
    yaw, pitch = torch.zeros(1, device=device), torch.full((1,), CAMERA_PITCH, device=device)
    eye = orbit(yaw, pitch, CAMERA_DISTANCE)
    target = eye.new_tensor(CAMERA_TARGET)
    return Camera(eye[0], perspective(FOV, aspect, device=device) @ look_at(eye, target))


def shade(
    position: Tensor, normal: Tensor, is_core: Tensor, eye: Tensor, core_colour: Tensor
) -> Tensor:
    """Premultiplied RGBA of the surface in one layer, in linear light."""
    view = F.normalize(eye - position, dim=-1)
    normal = F.normalize(normal, dim=-1)
    facing = (normal * view).sum(-1, keepdim=True)
    # Every second layer is the inside of a wall. Its normal points away from the camera, so it
    # is flipped to light the side that is seen.
    normal = torch.where(facing < 0, -normal, normal)
    facing = facing.abs()

    light = F.normalize(position.new_tensor(KEY_LIGHT), dim=0)
    diffuse = AMBIENT + (normal @ light).clamp(min=0)[..., None]

    # Both materials are glossy: they mirror the softbox, weakly where the surface faces the
    # camera and strongly towards its rim.
    grazing = (1 - facing) ** 3
    reflectance = REFLECTANCE + (1 - REFLECTANCE) * grazing
    mirrored = 2 * facing * normal - view
    low, high = SOFTBOX_EDGE
    softbox = (((mirrored @ light)[..., None] - low) / (high - low)).clamp(0, 1)
    highlight = reflectance * SOFTBOX * softbox

    # The path through a thin wall gets longer at grazing angles, which makes the rim denser.
    wall = 1 - (1 - GLASS_OPACITY) ** (1 / facing.clamp(min=0.02))
    # Clear glass mirrors its surroundings as brightly as dense glass does, so the reflections
    # are added to the premultiplied colour without being weighted by the wall's opacity.
    glass = position.new_tensor(GLASS_TINT) * diffuse * wall
    glass = glass + highlight + grazing * position.new_tensor(RIM_LIGHT)
    # Whatever the wall absorbs or reflects does not arrive from behind it.
    glass_alpha = 1 - (1 - wall) * (1 - reflectance)

    core = core_colour * diffuse + highlight
    colour = torch.where(is_core, core, glass)
    alpha = torch.where(is_core, torch.ones_like(facing), glass_alpha)
    return torch.cat([colour, alpha], -1)


def render(
    ctx: dr.RasterizeContext,
    scene: Scene,
    camera: Camera,
    size: tuple[int, int],
    num_layers: int,
    supersample: int,
    background: Tensor,
) -> Render:
    """Peel ``num_layers`` depth layers and blend them front to back over ``background``.

    Going from front to back, every layer adds its colour, dimmed by the layers already passed,
    and then dims what is still to come by its own opacity.
    """
    width, height = size
    resolution = (height * supersample, width * supersample)
    pos = to_clip(scene.positions, camera.matrix)
    normals = vertex_normals(scene.positions, scene.tri)
    attributes = torch.cat([scene.positions, normals], -1)

    colour, transmittance = 0.0, 1.0
    layers, coverage = [], []
    with dr.DepthPeeler(ctx, pos, scene.tri, resolution) as peeler:
        for _ in range(num_layers):
            rast, _ = peeler.rasterize_next_layer()
            surface, _ = dr.interpolate(attributes, rast, scene.tri)

            # The last channel of rast is the triangle index plus one, and zero where empty.
            covered = rast[..., 3:] > 0
            is_core = rast[..., 3:] > scene.knot_triangles
            position, normal = surface[..., :3], surface[..., 3:]
            rgba = shade(position, normal, is_core, camera.eye, scene.core_colour)
            rgba = dr.antialias(rgba * covered, rast, pos, scene.tri)

            colour = colour + transmittance * rgba[..., :3]
            transmittance = transmittance * (1 - rgba[..., 3:])

            # The layers stay linear, because layer_strip still blends them over a backdrop.
            layers.append(downsample(rgba, supersample)[0])
            coverage.append(covered.float().mean().item())

    image = colour + transmittance * background
    return Render(downsample(to_srgb(image), supersample)[0], layers, coverage)


def layer_strip(
    result: Render, background: Tensor, shown: int, gap: int, label_size: int
) -> Image.Image:
    """The first ``shown`` layers on their own, then all of them blended, as labelled panels.

    A layer holds whatever surface comes next, also where the opaque sphere in front of it lets
    nothing through, so the panels show the far side of the sphere and the glass inside it.
    """
    # Later layers are fragments. A faint outline of the nearest layer shows where they belong.
    backdrop = background + OUTLINE * (result.layers[0][..., 3:] > 0)
    alone = [
        to_srgb(layer[..., :3] + (1 - layer[..., 3:]) * backdrop) for layer in result.layers[:shown]
    ]
    labels = ["layer 1 (nearest)", *[f"layer {index}" for index in range(2, len(alone) + 1)]]
    labels.append(f"{len(result.layers)} layers blended")
    return panels([to_image(image) for image in [*alone, result.image]], labels, gap, label_size)


def check_gradients(
    ctx: dr.RasterizeContext,
    scene: Scene,
    camera: Camera,
    size: int,
    num_layers: int,
    background: Tensor,
    seed: int,
) -> None:
    """Backpropagate the mean of the blended image to the vertices and to the sphere's colour.

    Blending more layers lets the gradient reach vertices that the nearest surface hides.
    """
    generator = torch.Generator().manual_seed(seed)
    start = torch.rand(3, generator=generator).to(background.device)

    def mean_brightness(positions: Tensor, core_colour: Tensor, layers: int) -> Tensor:
        changed = scene._replace(positions=positions, core_colour=core_colour)
        return render(ctx, changed, camera, (size, size), layers, 1, background).image.mean()

    print(f"gradients of the mean of a {size}x{size} image, for a sphere of a random colour")
    print("layers   vertices reached   |d/d positions|   |d/d sphere colour|")
    for layers in sorted({1, 2, 3, 4, num_layers}):
        positions = scene.positions.clone().requires_grad_()
        core_colour = start.clone().requires_grad_()
        mean_brightness(positions, core_colour, layers).backward()
        reached = (positions.grad.abs().sum(-1) > 0).float().mean().item()
        print(
            f"{layers:6d}   {100 * reached:14.1f} %   {positions.grad.norm().item():15.3e}"
            f"   {core_colour.grad.norm().item():19.3e}"
        )
    print("the rest are behind the opaque sphere, or in triangles that no pixel centre falls in")

    # The image varies smoothly with the sphere's colour, so a central difference is an
    # independent check on the gradient, part of which arrives through the glass.
    with torch.no_grad():
        step = torch.tensor([0.01, 0.0, 0.0], device=start.device)
        high = mean_brightness(scene.positions, start + step, num_layers)
        low = mean_brightness(scene.positions, start - step, num_layers)
    print(
        f"d/d red of the sphere: autograd {core_colour.grad[0].item():.6f}, "
        f"finite difference {((high - low) / 0.02).item():.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="write the blended image to this PNG")
    parser.add_argument("--width", type=int, default=1600, help="width of the blended image")
    parser.add_argument("--height", type=int, default=1000, help="height of the blended image")
    parser.add_argument("--layers", help="write single layers next to the blend to this PNG")
    parser.add_argument("--panel-size", type=int, default=308, help="panel side in --layers")
    parser.add_argument(
        "--gif",
        help="write a turntable of layers and blend to this animation; "
        "the extension picks the format: .gif, .webp, .mp4 or .webm",
    )
    parser.add_argument("--frames", type=int, default=48, help="frames in a third of a turn")
    parser.add_argument("--gif-width", type=int, default=960, help="width of the animation")
    parser.add_argument(
        "--check-gradients", action="store_true", help="print gradients through the layers"
    )
    parser.add_argument("--check-size", type=int, default=512, help="image side for the check")
    parser.add_argument("--num-layers", type=int, default=8, help="depth layers to peel")
    parser.add_argument(
        "--supersample", type=int, default=2, help="render at this multiple and average down"
    )
    parser.add_argument("--segments", type=int, default=768, help="segments along the knot")
    parser.add_argument("--subdivisions", type=int, default=4, help="icosphere subdivisions")
    parser.add_argument("--seed", type=int, default=0, help="picks the sphere colour of the check")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ctx = dr.RasterizeContext(device)
    scene = build_scene(args.segments, args.subdivisions, device)
    canvas, panel = from_srgb(BACKGROUND, device), from_srgb(PANEL, device)

    def draw(subject: Scene, size: tuple[int, int], background: Tensor) -> Render:
        camera = make_camera(size[0] / size[1], device)
        with torch.no_grad():
            return render(ctx, subject, camera, size, args.num_layers, args.supersample, background)

    still = draw(scene, (args.width, args.height), canvas)
    print(f"{args.width}x{args.height} at {args.supersample}x supersampling on {device}")
    print("layer   pixels covered")
    for index, fraction in enumerate(still.coverage):
        print(f"{index + 1:5d}   {100 * fraction:12.3f} %")
    print("a ray enters and leaves each closed surface, so layers come in pairs and thin out")
    if args.out:
        to_image(still.image).save(args.out)
        print(f"wrote {args.out}")

    if args.layers:
        result = draw(scene, (args.panel_size, args.panel_size), panel)
        # Five panels of the default size and six gaps make a figure 1660 pixels wide.
        scale = args.panel_size / 308
        gap, label_size = round(20 * scale), round(28 * scale)
        layer_strip(result, panel, shown=4, gap=gap, label_size=label_size).save(args.layers)
        print(f"wrote {args.layers}")

    if args.gif:
        gap, side, label_size = layout(args.gif_width, 4)
        frames = []
        for index in range(args.frames):
            # The knot has three-fold symmetry, so a third of a turn loops without a seam.
            turn = turned(scene, 2 * math.pi / 3 * index / args.frames)
            result = draw(turn, (side, side), panel)
            frames.append(layer_strip(result, panel, shown=3, gap=gap, label_size=label_size))
        save_animation(frames, args.gif, frame_ms=FRAME_MS, seamless=True)
        print(f"wrote {args.gif} ({len(frames)} frames)")

    if args.check_gradients:
        check_gradients(
            ctx, scene, make_camera(1.0, device), args.check_size, args.num_layers, panel, args.seed
        )


if __name__ == "__main__":
    main()
