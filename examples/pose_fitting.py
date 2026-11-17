"""Recover the rotation of a die from a single image by fitting many pose hypotheses at once.

A die is rendered in a hidden pose. A batch of hypotheses, each with its own quaternion, is rendered
by one call of each op and compared with that image. The loss of a hypothesis depends on its own
image only, so one backward pass gives every quaternion its own gradient.

Gradient descent from a random rotation rarely ends on the true pose: about 1 start in 40 does.
Every second start settles on a pose that is 90, 120 or 180 degrees off, where the outline matches
and only the pips differ, because 24 rotations map a cube onto itself. The search therefore has
two stages, and the batch dimension carries both:

1. Random starts are fitted together. The one with the lowest loss has found the outline.
2. That pose is turned by each of the 24 rotations and the copies are fitted together. Now the
   pips decide: the copy with the lowest loss is the answer.

    python examples/pose_fitting.py --gif pose_fitting.gif --out pose_fitting.png

The animation shows the target beside all hypotheses. A tile is brighter the lower its loss, the
lowest is framed, and when a stage has finished every tile is captioned with its distance from
the true pose. Between the stages the best start is copied to every tile and the copies turn into
place. The still shows the end of both stages.

With the default settings and seeds 0 to 99, the answer was within 0.5 degrees of the true pose
97 times in one sweep and 98 times in another; stage 1 on its own was right 42 and 43 times.
Five of those hidden poses show a single face, an image that fits two or four poses equally well.
The script reports such a tie. Which of the tied poses comes out is left to rounding, and every
miss was one of these five. Stage 2 relies on knowing that the object is a cube, and the target
is a render without noise from this same renderer.

``--gif`` writes the format that its file name ends in: ``.gif``, ``.webp`` or, with ffmpeg on the
PATH, ``.mp4`` and ``.webm``.

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from functools import partial
from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

import diffrast as dr
from common import (
    ACCENT,
    BACKGROUND,
    CLAY,
    INK,
    MUTED,
    PANEL,
    TEXT,
    downsample,
    font,
    from_srgb,
    lighting,
    look_at,
    perspective,
    rgb,
    save_animation,
    stack,
    to_clip,
    to_image,
    to_srgb,
)

# Pips and outward direction (axis, sign) of each face; opposite faces sum to seven. Face k is
# drawn in cell (k % 3, k // 3) of the atlas.
FACES = [(1, 1, 1.0), (6, 1, -1.0), (2, 2, 1.0), (5, 2, -1.0), (3, 0, 1.0), (4, 0, -1.0)]
PIP_LAYOUT = {
    1: [(0, 0)],
    2: [(-1, -1), (1, 1)],
    3: [(-1, -1), (0, 0), (1, 1)],
    4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
    6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
}

# Directions towards the lights. The key light is far enough to the side that the three visible
# faces of the die get three different shades.
KEY_LIGHT = (0.55, 0.75, 0.36)
FILL_LIGHT = (-0.7, -0.2, 0.4)

FOV = 30.0
# From this far away the die leaves a margin in every pose, with room for a caption in a corner.
CAMERA_DISTANCE = 4.2

# A hypothesis this close to the true pose, in degrees, counts as having found it.
FOUND = 2.0
# Two losses closer than this cannot be told apart: the image fits both poses.
TIE = 5e-6
# Milliseconds per frame of the animation, and for which it stops on the result of stage 1.
FRAME_MS = 80
PAUSE_MS = 1120


class Die(NamedTuple):
    """Cube with one index buffer per attribute.

    The eight corners are shared by all faces, so ``tri`` describes a closed surface, which is
    what ``antialias`` needs to tell silhouette edges from the edges between two visible faces.
    Texture coordinates and normals differ per face and are indexed by their own buffers.
    """

    positions: Tensor  # [8, 3]
    tri: Tensor  # [12, 3]
    uvs: Tensor  # [24, 2]
    tri_uv: Tensor  # [12, 3]
    normals: Tensor  # [6, 3]
    tri_normal: Tensor  # [12, 3]


class Scene(NamedTuple):
    """Everything a render needs besides the poses."""

    ctx: dr.RasterizeContext
    die: Die
    atlas: Tensor
    camera: Tensor


def make_die(device: torch.device) -> Die:
    """Unit cube whose face ``k`` maps onto cell ``k`` of a 3x2 texture atlas."""
    signs = (-1.0, 1.0)
    corners = [[x, y, z] for x in signs for y in signs for z in signs]
    tri, uvs, tri_uv, normals, tri_normal = [], [], [], [], []
    for face, (_, axis, sign) in enumerate(FACES):
        u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
        quad = []
        for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
            corner = [0.0, 0.0, 0.0]
            corner[axis] = sign
            corner[u_axis] = (2 * u - 1) * sign  # keeps the face from being mirrored
            corner[v_axis] = 2 * v - 1
            quad.append(corners.index(corner))
            uvs.append([(face % 3 + u) / 3, (face // 3 + v) / 2])
        first = 4 * face
        tri += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
        tri_uv += [[first, first + 1, first + 2], [first, first + 2, first + 3]]
        normals.append([sign if k == axis else 0.0 for k in range(3)])
        tri_normal += [[face] * 3] * 2

    def floats(values):
        return torch.tensor(values, dtype=torch.float32, device=device)

    def indices(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    return Die(
        0.5 * floats(corners),
        indices(tri),
        floats(uvs),
        indices(tri_uv),
        floats(normals),
        indices(tri_normal),
    )


def pip_atlas(cell: int, device: torch.device) -> Tensor:
    """Linear albedo of the six faces as a ``[1, 2 * cell, 3 * cell, 3]`` texture."""
    y, x = torch.meshgrid(
        (torch.arange(2 * cell, device=device) + 0.5) / cell,
        (torch.arange(3 * cell, device=device) + 0.5) / cell,
        indexing="ij",
    )
    ink = torch.zeros_like(x)
    for face, (pips, _, _) in enumerate(FACES):
        radius = 0.15 if pips == 1 else 0.095
        for pip_x, pip_y in PIP_LAYOUT[pips]:
            centre_x = face % 3 + 0.5 + 0.25 * pip_x
            centre_y = face // 3 + 0.5 + 0.25 * pip_y
            distance = torch.hypot(x - centre_x, y - centre_y)
            # A ramp one texel wide keeps the edge of the pip smooth.
            ink = torch.maximum(ink, ((radius - distance) * cell + 0.5).clamp(0, 1))
    # A slightly darker rim reads as a bevel and makes the edges between faces visible.
    to_edge = torch.minimum(0.5 - (x % 1 - 0.5).abs(), 0.5 - (y % 1 - 0.5).abs())
    rim = 1 - 0.22 * (1 - to_edge / 0.045).clamp(0, 1) ** 2
    clay, dark = x.new_tensor(CLAY), x.new_tensor(INK)
    albedo = torch.lerp(clay * rim[..., None], dark, ink[..., None])
    return albedo[None].contiguous()


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Rotation matrices ``[n, 3, 3]`` of quaternions ``[n, 4]`` in ``(w, x, y, z)`` order."""
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).view(-1, 3, 3)  # fmt: skip


def compose(a: Tensor, b: Tensor) -> Tensor:
    """Quaternion of the rotation ``b`` followed by the rotation ``a``."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def cube_rotations(device: torch.device) -> Tensor:
    """Quaternions ``[24, 4]`` of the rotations that map a cube onto itself.

    They are the quaternions with one, two or four entries of equal size and zeros elsewhere:
    turns about a face by a multiple of 90 degrees, about an edge by 180 and about a corner by
    120.
    """
    q = torch.cartesian_prod(*[torch.tensor([0.0, 1.0, -1.0])] * 4)
    entries = (q != 0).sum(-1)
    q = q[(entries == 1) | (entries == 2) | (entries == 4)]
    # q and -q are the same rotation. The weights let the first non-zero entry decide the sign.
    q = q[q @ torch.tensor([8.0, 4.0, 2.0, 1.0]) > 0]
    return F.normalize(q, dim=-1).to(device)


def turned(pose: Tensor, rotations: Tensor, amount: float = 1.0) -> Tensor:
    """The die in ``pose`` after turning it by each of ``rotations``.

    An ``amount`` below one stops part of the way, for the animation.
    """
    identity = rotations.new_tensor([1.0, 0.0, 0.0, 0.0])
    return compose(pose, F.normalize(torch.lerp(identity, rotations, amount), dim=-1))


def random_quaternions(count: int, generator: torch.Generator, device: torch.device) -> Tensor:
    """Uniformly distributed rotations, drawn on the CPU so a seed gives the same on any device."""
    return F.normalize(torch.randn(count, 4, generator=generator), dim=-1).to(device)


def angle_between(q: Tensor, reference: Tensor) -> Tensor:
    """Angle in degrees of the rotation that takes each ``q`` to ``reference``."""
    dot = (F.normalize(q, dim=-1) * reference).sum(-1).abs().clamp(max=1)
    return torch.rad2deg(2 * torch.acos(dot))


def matches_outline(q: Tensor, reference: Tensor, tolerance: float = 0.1) -> Tensor:
    """Whether each ``q`` differs from ``reference`` by a rotation that maps a cube onto itself.

    Such a rotation permutes the axes of the die, so its matrix holds only 0, 1 and -1.
    """
    relative = quaternion_to_matrix(reference).transpose(1, 2) @ quaternion_to_matrix(q)
    return (relative - relative.round()).abs().amax(dim=(1, 2)) < tolerance


def render(scene: Scene, poses: Tensor, resolution: int) -> Tensor:
    """Linear images ``[n, resolution, resolution, 3]`` of the die, one per quaternion."""
    die = scene.die
    rotation = quaternion_to_matrix(poses).transpose(1, 2)
    pos = to_clip(die.positions @ rotation, scene.camera)

    rast, rast_db = dr.rasterize(scene.ctx, pos, die.tri, (resolution, resolution))
    uv, uv_da = dr.interpolate(die.uvs, rast, die.tri_uv, rast_db=rast_db, diff_attrs="all")
    normal, _ = dr.interpolate(die.normals @ rotation, rast, die.tri_normal)
    albedo = dr.texture(
        scene.atlas, uv, uv_da, filter_mode="linear-mipmap-linear", boundary_mode="clamp"
    )

    shaded = albedo * lighting(normal, KEY_LIGHT, FILL_LIGHT)
    covered = rast[..., 3:] > 0
    colour = torch.where(covered, shaded, from_srgb(PANEL, pos.device))
    return dr.antialias(colour, rast, pos, die.tri)


def image_loss(scene: Scene, poses: Tensor, target: Tensor) -> Tensor:
    """Mean squared difference between the image of each pose and the target, ``[n]``."""
    image = render(scene, poses, target.shape[1])
    # Every image is compared with the target on its own: one loss per hypothesis.
    return (image - target).square().mean(dim=(1, 2, 3))


def fit(
    scene: Scene, target: Tensor, poses: Tensor, steps: int, lr: float, momentum: float
) -> Iterator[tuple[int, Tensor, Tensor]]:
    """Fit all poses to the target at once by gradient descent with momentum.

    Yields ``(step, poses, losses)`` before every update and once more after the last one.
    """
    poses = poses.clone().requires_grad_(True)
    optimiser = torch.optim.SGD([poses], lr=lr, momentum=momentum)
    for step in range(steps + 1):
        loss = image_loss(scene, poses, target)
        yield step, poses.detach().clone(), loss.detach()
        if step == steps:
            return
        optimiser.zero_grad()
        # No loss depends on another hypothesis, so the gradient of their sum holds the gradient
        # of each hypothesis with respect to its own quaternion.
        loss.sum().backward()
        optimiser.step()
        with torch.no_grad():
            # Only the direction of a quaternion matters; unit length keeps the steps comparable.
            poses.copy_(F.normalize(poses, dim=-1))


def report(stage: int, step: int, loss: Tensor, error: Tensor) -> None:
    """Print how the hypothesis with the lowest loss is doing, and how many found the pose."""
    best = int(loss.argmin())
    print(
        f"stage {stage}  step {step:3d}  lowest loss {loss[best].item():.6f}, "
        f"{error[best].item():6.2f} deg off  within {FOUND:g} deg: "
        f"{int((error < FOUND).sum())}/{len(loss)}"
    )


def early_steps(steps: int, count: int) -> set[int]:
    """``count`` steps to draw a frame at. Most turning happens early, so they crowd there."""
    return {round(steps * (i / count) ** 2) for i in range(count)}


class Layout(NamedTuple):
    """Pixel geometry of a figure: one large image on the left, a grid of tiles on the right."""

    width: int
    height: int
    pad: int
    gap: int
    text: int
    top: int
    large: int
    cell: int
    columns: int
    grid_x: int


def plan_layout(width: int, tiles: int) -> Layout:
    """Square tiles sized so that the grid is as tall as the large image and fills ``width``."""
    unit = width / 960
    pad, gap, text = round(16 * unit), round(6 * unit), round(18 * unit)
    rows = math.ceil(math.sqrt(tiles / 2))
    columns = math.ceil(tiles / rows)
    cell = (width - 3 * pad - (rows + columns - 2) * gap) // (rows + columns)
    large = rows * cell + (rows - 1) * gap
    top = 2 * pad + text
    grid_x = width - pad - columns * cell - (columns - 1) * gap
    return Layout(width, top + large + pad, pad, gap, text, top, large, cell, columns, grid_x)


def figure(
    scene: Scene,
    true_pose: Tensor,
    layout: Layout,
    label: str,
    large_pose: Tensor,
    heading: str,
    poses: Tensor,
    loss: Tensor,
    captions: bool = False,
) -> Image.Image:
    """The die in ``large_pose`` beside all hypotheses, each dimmed by its loss.

    The hypothesis with the lowest loss is framed. With ``captions``, every tile shows its
    distance from the true pose, muted where not even the outline fits.
    """
    # Rendered at twice the size and averaged down.
    with torch.no_grad():
        large = downsample(to_srgb(render(scene, large_pose, 2 * layout.large)), 2)[0]
        tiles = downsample(to_srgb(render(scene, poses, 2 * layout.cell)), 2)
    error = angle_between(poses, true_pose)
    settled = matches_outline(poses, true_pose)
    best = int(loss.argmin())

    canvas = Image.new("RGB", (layout.width, layout.height), rgb(BACKGROUND))
    draw = ImageDraw.Draw(canvas)
    big, small = font(layout.text), font(round(layout.text * 8 / 9))
    draw.text((layout.pad, layout.pad), label, rgb(MUTED), big)
    canvas.paste(to_image(large), (layout.pad, layout.top))
    draw.text((layout.grid_x, layout.pad), heading, rgb(MUTED), big)
    # The number ends at the right edge and its label before a slot of fixed width, so that
    # nothing shifts between frames.
    right = layout.width - layout.pad
    slot = round(draw.textlength(" 000.0°", font=big))
    draw.text((right - slot, layout.pad), "best is off by", rgb(MUTED), big, anchor="ra")
    draw.text((right, layout.pad), f"{error[best].item():.1f}°", rgb(TEXT), big, anchor="ra")

    # Tiles fade towards the panel colour with their loss, measured between the best and the
    # worst of the moment, so the ranking stays readable while every loss shrinks.
    excess = (loss - loss.min()) / (loss.max() - loss.min()).clamp(min=TIE)
    opacity = 1 - 0.68 * excess.sqrt()
    tiles = torch.lerp(tiles.new_tensor(PANEL), tiles, opacity[:, None, None, None])
    for index in range(len(loss)):
        x = layout.grid_x + (index % layout.columns) * (layout.cell + layout.gap)
        y = layout.top + (index // layout.columns) * (layout.cell + layout.gap)
        canvas.paste(to_image(tiles[index]), (x, y))
        if captions:
            caption = f"{error[index].item():.0f}°"
            corner = (x + layout.gap, y + layout.cell - layout.gap // 2)
            draw.text(corner, caption, rgb(TEXT if settled[index] else MUTED), small, anchor="ls")
        if index == best:
            line = max(2, round(layout.text / 7))
            box = [x - line, y - line, x + layout.cell + line - 1, y + layout.cell + line - 1]
            draw.rectangle(box, outline=rgb(ACCENT), width=line)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hypotheses", type=int, default=24, help="random starts of stage 1")
    parser.add_argument("--iterations", type=int, default=60, help="steps of stage 1")
    parser.add_argument("--refine", type=int, default=60, help="steps of stage 2")
    parser.add_argument("--resolution", type=int, default=64, help="size of the fitted images")
    parser.add_argument("--texture-size", type=int, default=256, help="size of one atlas cell")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of gradient descent")
    parser.add_argument(
        "--seed",
        type=int,
        default=14,
        help="picks the hidden pose and the starts; with the default, stage 1 alone ends a "
        "quarter turn off",
    )
    parser.add_argument(
        "--gif",
        help="write the optimisation progress to this animation; "
        "the extension picks the format: .gif, .webp, .mp4 or .webm",
    )
    parser.add_argument(
        "--frames", type=int, default=32, help="frames of stage 1; the later parts scale with it"
    )
    parser.add_argument("--gif-width", type=int, default=960, help="width of the animation")
    parser.add_argument("--out", help="write a still of both stages to this PNG")
    parser.add_argument("--out-width", type=int, default=1660, help="width of the PNG")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    eye = torch.tensor([[0.0, 0.0, CAMERA_DISTANCE]], device=device)
    scene = Scene(
        dr.RasterizeContext(device),
        make_die(device),
        pip_atlas(args.texture_size, device),
        perspective(FOV, 1.0, device=device) @ look_at(eye),
    )
    true_pose = random_quaternions(1, generator, device)
    starts = random_quaternions(args.hypotheses, generator, device)
    rotations = cube_rotations(device)
    with torch.no_grad():
        target = render(scene, true_pose, args.resolution)

    draw = partial(figure, scene, true_pose)
    tiles = max(args.hypotheses, len(rotations))
    layout = plan_layout(args.gif_width, tiles)
    # The frames of the animation, and how long the ones that pause stay on screen.
    frames, holds = [], {}

    # Stage 1: from random rotations, most hypotheses turn until their outline fits.
    heading = f"{args.hypotheses} random starts in one batch"
    capture = early_steps(args.iterations, args.frames)
    for step, poses, loss in fit(scene, target, starts, args.iterations, args.lr, args.momentum):
        error = angle_between(poses, true_pose)
        if step % 20 == 0 or step == args.iterations:
            report(1, step, loss, error)
        if args.gif and step in capture:
            frames.append(draw(layout, "target", true_pose, f"{heading}, step {step}", poses, loss))
    best_start = poses[loss.argmin()]
    first = (f"{heading}, step {args.iterations}", poses, loss)  # kept for the figures
    found = int((error < FOUND).sum())
    # The smallest turn that maps a cube onto itself is a quarter turn.
    symmetric = int((matches_outline(poses, true_pose) & (error > 45)).sum())
    print(
        f"stage 1: of {args.hypotheses} random starts, {found} reached the true pose, "
        f"{symmetric} a symmetric pose and {args.hypotheses - found - symmetric} neither; "
        f"the one with the lowest loss is {error[loss.argmin()].item():.2f} deg off"
    )

    # Stage 2: the true pose has the same outline as the best start, so it is one of the 24
    # turned copies of it, and the pips single it out.
    heading = f"the best start, turned {len(rotations)} ways"
    if args.gif:
        # Pause on the settled poses with their errors, then show where the copies come from:
        # every tile starts as the best start, rests for a moment and turns into its place.
        holds[len(frames)] = PAUSE_MS
        frames.append(draw(layout, "target", true_pose, *first, captions=True))
        holds[len(frames)] = PAUSE_MS // 2
        turn_frames = max(1, args.frames // 2)
        for frame in range(turn_frames):
            amount = frame / turn_frames
            poses = turned(best_start, rotations, amount * amount * (3 - 2 * amount))
            with torch.no_grad():
                loss = image_loss(scene, poses, target)
            frames.append(draw(layout, "target", true_pose, heading, poses, loss))
    capture = early_steps(args.refine, max(1, args.frames // 8))
    copies = turned(best_start, rotations)
    for step, poses, loss in fit(scene, target, copies, args.refine, args.lr, args.momentum):
        error = angle_between(poses, true_pose)
        if step % 20 == 0 or step == args.refine:
            report(2, step, loss, error)
        if args.gif and step in capture:
            frames.append(draw(layout, "target", true_pose, f"{heading}, step {step}", poses, loss))
    best, runner_up = loss.argsort()[:2].tolist()
    second = (f"{heading}, step {args.refine}", poses, loss)
    print(
        f"stage 2: hypothesis {best} has the lowest loss ({loss[best].item():.6f}) and is "
        f"{error[best].item():.2f} deg off; the runner-up ({loss[runner_up].item():.6f}) is "
        f"{error[runner_up].item():.0f} deg off"
    )
    if loss[runner_up] - loss[best] < TIE:
        print("the two fit the image equally well: this view of the die does not tell them apart")

    if args.gif:
        frames.append(draw(layout, "target", true_pose, *second, captions=True))
        save_animation(frames, args.gif, frame_ms=FRAME_MS, holds=holds)
        print(f"wrote {args.gif}")
    if args.out:
        layout = plan_layout(args.out_width, tiles)
        rows = [
            draw(layout, "target", true_pose, *first, captions=True),
            draw(layout, "the best random start", best_start[None], *second, captions=True),
        ]
        stack(rows, layout.pad).save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
