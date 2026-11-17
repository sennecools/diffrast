"""Recover a mesh from rendered images alone.

A sphere is deformed until its renders match renders of a target shape from random viewpoints.
Which triangle a pixel shows is a step function of the vertex positions, so a plain rasterizer
gives no gradient for moving a silhouette. ``dr.antialias`` turns that step into a ramp, and the
ramp is what lets the outline of the sphere travel to the outline of the target.

    python examples/shape_fitting.py --gif shape_fitting.gif --out shape_fitting.png

The images do not say where each vertex should go, only where the outline and the shading are
wrong, and a naive optimisation folds the surface over itself. Three things prevent that here: the
positions are optimised through a smoothing change of variables, the optimiser keeps the direction
of the smoothed gradient, and shading is compared only where the target is. None of the three is
part of diffrast; they are spelled out below.

The animation shows the target, the mesh on a slow turntable, and four views that are never
trained on, in which the mesh is painted over the target: orange is what the mesh has yet to
cover. The script prints how well the outlines overlap on such held-out views, and at the end the
fraction of triangles that ended up flipped.

The target is a chess pawn, or any OBJ file given with ``--mesh``. The optimisation moves vertices
and cannot change how they are connected, so the target must have the topology of a sphere: a
shape with a hole or a handle will not be recovered.

``--gif`` writes the format that its file name ends in: ``.gif``, ``.webp`` or, with ffmpeg on the
PATH, ``.mp4`` and ``.webm``.

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

import diffrast as dr
from common import (
    BACKGROUND,
    CLAY,
    PANEL,
    Mesh,
    downsample,
    from_srgb,
    icosphere,
    lathe,
    layout,
    load_obj,
    look_at,
    orbit,
    panels,
    perspective,
    rgb,
    save_animation,
    to_clip,
    to_image,
    to_srgb,
    vertex_normals,
)

# From this far away the unit sphere, which bounds every mesh, fits in the image with a margin.
FOV, CAMERA_DISTANCE = 40.0, 3.4
# Camera of the pictures. The views that are trained on are random.
PICTURE_YAW, PICTURE_PITCH = 0.6, 0.3
HELD_OUT_VIEWS = 16

# A small start, so that most of the surface has to grow. Where a wide sphere must shrink onto a
# thin part of the target, its triangles are squeezed into slivers that flip easily.
START_RADIUS = 0.4
# The learning rate ends at this fraction of its peak.
LR_DECAY = 0.05

# Direction towards each light in camera space, and its colour. The lights move with the camera,
# so every side of the shape is lit in some view: a key light from the upper front and a faint
# cool fill. Together they put a lit clay surface just below white after common.to_srgb.
LIGHTS = (
    ((0.65, 0.60, 0.45), (0.70, 0.67, 0.63)),
    ((-0.80, 0.00, 0.30), (0.10, 0.13, 0.18)),
)
AMBIENT = 0.14

# Linear albedo of the target in the pictures, which comes out close to common.ACCENT in the key
# light. The fitted mesh is drawn in common.CLAY.
ORANGE = (1.5, 0.114, 0.014)

Camera = tuple[Tensor, Tensor]  # clip-space matrices and world-to-camera rotations


def pawn(segments: int = 96, device: torch.device | str = "cpu") -> Mesh:
    """Chess pawn as a surface of revolution, centred and scaled to the unit sphere."""
    angle = np.radians(np.linspace(90, -62, 13))
    head = np.stack([0.185 * np.cos(angle), 0.83 + 0.185 * np.sin(angle)], -1)
    head[0, 0] = 0.0
    body = [
        (0.100, 0.670), (0.105, 0.650), (0.170, 0.630), (0.215, 0.605), (0.170, 0.585),
        (0.100, 0.555), (0.095, 0.500), (0.115, 0.370), (0.150, 0.250), (0.200, 0.170),
        (0.250, 0.140), (0.290, 0.110), (0.360, 0.085), (0.385, 0.050), (0.370, 0.015),
        (0.300, 0.000), (0.000, 0.000),
    ]  # fmt: skip
    profile = np.concatenate([head, body])

    # Chaikin corner cutting rounds the polyline; its two ends stay on the axis.
    for _ in range(3):
        cut = np.stack([0.75 * profile[:-1] + 0.25 * profile[1:],
                        0.25 * profile[:-1] + 0.75 * profile[1:]], 1)  # fmt: skip
        profile = np.concatenate([profile[:1], cut.reshape(-1, 2), profile[-1:]])

    # Rings at equal arc length, about as far apart as the vertices of the widest ring.
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(profile, axis=0), axis=1))])
    samples = np.linspace(0, along[-1], segments * 5 // 3)
    radius = torch.from_numpy(np.interp(samples, along, profile[:, 0]))
    height = torch.from_numpy(np.interp(samples, along, profile[:, 1]))

    mesh = lathe(radius, height, segments, device)
    positions = mesh.positions - (mesh.positions.amax(0) + mesh.positions.amin(0)) / 2
    return Mesh(positions / positions.norm(dim=-1).max(), mesh.tri)


def smoothing_matrices(vertices: int, tri: Tensor, weight: float) -> tuple[Tensor, Tensor]:
    """The matrix ``I + weight * L`` and its inverse, with ``L`` the graph Laplacian of the mesh.

    Both are dense, which is fine up to about ten thousand vertices (400 MB each).
    """
    i, j = tri.long().flatten(), tri.long().roll(-1, dims=1).flatten()
    system = torch.zeros(vertices, vertices, device=tri.device)
    system[i, j] = system[j, i] = -weight
    system.diagonal().copy_(1 - system.sum(1))
    return system, torch.cholesky_inverse(torch.linalg.cholesky(system))


class UniformAdam:
    """Adam with a single second moment shared by all coordinates.

    Adam divides every coordinate by its own gradient magnitude. That throws away the relative
    sizes the smoothing has just given the gradient, so vertices that hardly matter move as far
    as those on a silhouette and the surface folds. Dividing by one number, the largest squared
    gradient, keeps the step parallel to the smoothed gradient, and ``lr`` becomes the furthest
    any coordinate moves in a step.
    """

    def __init__(self, parameter: Tensor, betas: tuple[float, float] = (0.9, 0.99)) -> None:
        self.parameter, self.betas = parameter, betas
        self.mean = torch.zeros_like(parameter)
        self.peak = 0.0
        self.steps = 0

    @torch.no_grad()
    def step(self, gradient: Tensor, lr: float) -> None:
        beta1, beta2 = self.betas
        self.steps += 1
        self.mean = beta1 * self.mean + (1 - beta1) * gradient
        self.peak = beta2 * self.peak + (1 - beta2) * gradient.square().max().item()
        mean = self.mean / (1 - beta1**self.steps)
        peak = self.peak / (1 - beta2**self.steps)
        self.parameter -= lr * mean / max(math.sqrt(peak), 1e-20)


def learning_rate(step: int, iterations: int, peak: float, warmup: float) -> float:
    """Linear warm-up over the first ``warmup`` share of the iterations, then a slow decay.

    Full-size steps while the sphere is still far from the target overshoot and fold the surface
    before it has found the rough shape. The decay lets the last steps settle the fine detail.
    """
    ramp = min(1.0, (step + 1) / max(1.0, warmup * iterations))
    return peak * ramp * LR_DECAY ** (step / iterations)


def cameras(yaw: Tensor, pitch: Tensor) -> Camera:
    """Clip-space matrices ``[views, 4, 4]`` and world-to-camera rotations ``[views, 3, 3]``."""
    view = look_at(orbit(yaw, pitch, CAMERA_DISTANCE))
    return perspective(FOV, 1.0, device=yaw.device) @ view, view[:, :3, :3]


def random_views(count: int, generator: torch.Generator, device: torch.device) -> Camera:
    """Cameras spread evenly over all directions, including from above and below."""
    yaw = torch.rand(count, generator=generator) * 2 * math.pi
    pitch = torch.asin(torch.rand(count, generator=generator) * 2 - 1)
    return cameras(yaw.to(device), pitch.to(device))


def render(ctx: dr.RasterizeContext, mesh: Mesh, camera: Camera, resolution: int) -> Tensor:
    """Antialiased RGBA renders of a white surface, ``[views, height, width, 4]``.

    RGB is the shading and alpha the coverage, both zero outside the shape. Everything from the
    vertex positions to the output is differentiable.
    """
    matrix, rotation = camera
    pos = to_clip(mesh.positions, matrix)
    rast, _ = dr.rasterize(ctx, pos, mesh.tri, (resolution, resolution))
    normal, _ = dr.interpolate(vertex_normals(mesh.positions, mesh.tri), rast, mesh.tri)
    normal = F.normalize(normal, dim=-1)

    shade = torch.full_like(normal, AMBIENT)
    for direction, colour in LIGHTS:
        # A row vector times the world-to-camera rotation is that vector in world space.
        towards = F.normalize(normal.new_tensor(direction), dim=0) @ rotation
        lambert = (normal * towards[:, None, None]).sum(-1, keepdim=True).clamp(min=0)
        shade = shade + lambert * normal.new_tensor(colour)

    covered = (rast[..., 3:] > 0).float()
    image = torch.cat([shade * covered, covered], -1)
    # Without this call the coverage is a step function of the vertex positions and the
    # silhouette would never move.
    return dr.antialias(image, rast, pos, mesh.tri)


def image_loss(image: Tensor, reference: Tensor) -> Tensor:
    """Squared error of the coverage, plus that of the shading where the target is.

    Pixels that the mesh covers and the target does not are left to the coverage term alone.
    Comparing their shading with the black around the target would reward turning the surface
    away from the lights, and that creases it.
    """
    covered = reference[..., 3:]
    shading = (covered * (image[..., :3] - reference[..., :3]).square()).mean()
    return F.mse_loss(image[..., 3:], covered) + shading


@torch.no_grad()
def silhouette_iou(
    ctx: dr.RasterizeContext, mesh: Mesh, target: Mesh, camera: Camera, resolution: int
) -> float:
    """Intersection over union of the pixels that two meshes cover, over all the views."""
    ours = render(ctx, mesh, camera, resolution)[..., 3] > 0.5
    theirs = render(ctx, target, camera, resolution)[..., 3] > 0.5
    return ((ours & theirs).sum() / (ours | theirs).sum()).item()


def flipped_fraction(mesh: Mesh) -> float:
    """Fraction of triangles that face against the vertex normals around them.

    A smooth closed surface has none. Folds and creases show up here.
    """
    corners = mesh.positions[mesh.tri.long()]
    face = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    around = vertex_normals(mesh.positions, mesh.tri)[mesh.tri.long()].sum(1)
    return ((face * around).sum(-1) < 0).float().mean().item()


def composite(layers: Sequence[tuple[Tensor, tuple[float, ...]]]) -> Image.Image:
    """Paint RGBA renders, each tinted with an albedo, back to front on the panel colour.

    The renders are twice the size of the picture and are averaged down.
    """
    colour = from_srgb(PANEL, layers[0][0].device)
    for image, albedo in layers:
        colour = image[..., :3] * image.new_tensor(albedo) + (1 - image[..., 3:]) * colour
    return to_image(downsample(to_srgb(colour), 2))


@torch.no_grad()
def portrait(
    ctx: dr.RasterizeContext, mesh: Mesh, albedo: tuple[float, ...], yaw: float, size: int
) -> Image.Image:
    """Picture of a mesh from slightly above, with the camera ``yaw`` radians around it."""
    device = mesh.positions.device
    camera = cameras(
        torch.tensor([yaw], device=device), torch.tensor([PICTURE_PITCH], device=device)
    )
    return composite([(render(ctx, mesh, camera, 2 * size)[0], albedo)])


@torch.no_grad()
def overlays(
    ctx: dr.RasterizeContext, mesh: Mesh, target: Mesh, camera: Camera, size: int
) -> Image.Image:
    """The first four views in a grid, with the mesh in clay painted over the target in orange.

    Orange is visible wherever the target is not covered yet.
    """
    gap = max(1, size // 75)
    cell = (size - gap) // 2
    camera = (camera[0][:4], camera[1][:4])
    behind = render(ctx, target, camera, 2 * cell)
    front = render(ctx, mesh, camera, 2 * cell)
    grid = Image.new("RGB", (size, size), rgb(BACKGROUND))
    for k in range(behind.shape[0]):
        picture = composite([(behind[k], ORANGE), (front[k], CLAY)])
        grid.paste(picture, ((k % 2) * (size - cell), (k // 2) * (size - cell)))
    return grid


def frame_steps(iterations: int, count: int) -> list[int]:
    """Steps at which to draw the frames of the animation.

    Most of the change happens early, so the distance between frames grows geometrically: the
    last two are about fifty times further apart than the first two.
    """
    last = max(1, count - 1)
    steps = {round(iterations * math.expm1(4 * i / last) / math.expm1(4)) for i in range(count)}
    return sorted(steps)


def progress_frame(
    ctx: dr.RasterizeContext,
    mesh: Mesh,
    target: Mesh,
    held_out: Camera,
    step: int,
    iou: float,
    turn: float,
    width: int,
) -> Image.Image:
    """The target, the mesh on a turntable and four of the held-out views side by side.

    ``turn`` runs from 0 to 1 over the animation.
    """
    gap, panel, label_size = layout(width, 3)
    # Half a turn that ends on the view of the target picture.
    yaw = PICTURE_YAW + math.pi * (turn - 1)
    pictures = [
        portrait(ctx, target, ORANGE, PICTURE_YAW, panel),
        portrait(ctx, mesh, CLAY, yaw, panel),
        overlays(ctx, mesh, target, held_out, panel),
    ]
    labels = ["target", f"fitted mesh, step {step}", f"held-out views, IoU {iou:.3f}"]
    return panels(pictures, labels, gap, label_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mesh", help="OBJ file to use as the target instead of the pawn")
    parser.add_argument("--iterations", type=int, default=600, help="optimisation steps")
    parser.add_argument("--views", type=int, default=8, help="random views per iteration")
    parser.add_argument("--resolution", type=int, default=256, help="size of the views")
    parser.add_argument("--subdivisions", type=int, default=5, help="icosphere subdivisions")
    parser.add_argument("--segments", type=int, default=96, help="segments around the pawn")
    parser.add_argument("--smoothing", type=float, default=20.0, help="weight of the Laplacian")
    parser.add_argument("--lr", type=float, default=0.1, help="peak learning rate")
    parser.add_argument("--warmup", type=float, default=0.1, help="share of steps spent warming up")
    parser.add_argument("--seed", type=int, default=0, help="picks the random views")
    parser.add_argument(
        "--gif",
        help="write the optimisation progress to this animation; "
        "the extension picks the format: .gif, .webp, .mp4 or .webm",
    )
    parser.add_argument("--frames", type=int, default=64, help="frames of the animation")
    parser.add_argument("--gif-width", type=int, default=960, help="width of the animation")
    parser.add_argument("--out", help="write a filmstrip of the optimisation to this PNG")
    parser.add_argument(
        "--snapshots", type=int, nargs="*", default=[0, 40, 70, 120], help="steps in the filmstrip"
    )
    parser.add_argument("--out-width", type=int, default=1660, help="width of the PNG")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    ctx = dr.RasterizeContext(device)
    target = load_obj(args.mesh, device) if args.mesh else pawn(args.segments, device)
    sphere = icosphere(args.subdivisions, device)
    held_out = random_views(HELD_OUT_VIEWS, generator, device)

    # Moving the vertices directly gives a crumpled, self-intersecting surface, because each
    # view pulls hardest on the few vertices at its silhouettes. Following Nicolet et al. 2021,
    # "Large Steps in Inverse Rendering of Geometry", the optimised variable is u = (I + w L) x
    # instead of the positions x. The gradient with respect to u is the gradient with respect to
    # x spread over each vertex's neighbourhood, so every step moves the surface as a whole.
    to_latent, to_positions = smoothing_matrices(len(sphere.positions), sphere.tri, args.smoothing)
    latent = (to_latent @ (START_RADIUS * sphere.positions)).requires_grad_(True)
    optimiser = UniformAdam(latent)

    gif_steps = frame_steps(args.iterations, args.frames)
    strip_steps = sorted({s for s in args.snapshots if s < args.iterations} | {args.iterations})
    strip_gap, strip_panel, strip_label = layout(args.out_width, len(strip_steps) + 1)
    frames, strip = [], []

    for step in range(args.iterations + 1):
        mesh = Mesh(to_positions @ latent, sphere.tri)

        report = step % 100 == 0 or step == args.iterations
        draw = bool(args.gif) and step in gif_steps
        if report or draw:
            iou = silhouette_iou(ctx, mesh, target, held_out, args.resolution)
        if report:
            print(f"step {step:5d}  silhouette IoU on held-out views {iou:.4f}")
        if draw:
            turn = len(frames) / max(1, len(gif_steps) - 1)
            frames.append(
                progress_frame(ctx, mesh, target, held_out, step, iou, turn, args.gif_width)
            )
        if args.out and step in strip_steps:
            strip.append(portrait(ctx, mesh, CLAY, PICTURE_YAW, strip_panel))
        if step == args.iterations:
            break

        camera = random_views(args.views, generator, device)
        with torch.no_grad():
            reference = render(ctx, target, camera, args.resolution)
        image = render(ctx, mesh, camera, args.resolution)
        loss = image_loss(image, reference)
        (gradient,) = torch.autograd.grad(loss, latent)
        optimiser.step(gradient, learning_rate(step, args.iterations, args.lr, args.warmup))

    flipped = flipped_fraction(Mesh(mesh.positions.detach(), mesh.tri))
    print(f"final: silhouette IoU {iou:.4f}, flipped triangles {100 * flipped:.2f} %")

    if args.gif:
        save_animation(frames, args.gif, frame_ms=70)
        print(f"wrote {args.gif}")
    if args.out:
        strip.append(portrait(ctx, target, ORANGE, PICTURE_YAW, strip_panel))
        labels = [f"step {step}" for step in strip_steps] + ["target"]
        panels(strip, labels, strip_gap, strip_label).save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
