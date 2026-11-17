"""Show why a silhouette needs ``antialias`` before it can be optimised.

A coarse sphere is rendered as a flat silhouette and compared with the silhouette of the same
sphere moved a little to the side. Whether a pixel is covered is a yes-or-no decision, so without
``antialias`` the image does not change when a vertex moves by a small amount, and the gradient of
the loss with respect to the vertex positions is exactly zero. ``antialias`` blends the pixels
along the outline by how far the edge reaches into them. That makes the image a continuous function
of the vertices: the gradient appears, and only on the outline.

The mesh is coarse on purpose. ``antialias`` looks for the silhouette edge among the edges of the
triangle that a pixel shows, so it needs triangles that are larger than a pixel where they meet
the outline. Try ``--subdivisions 3`` to see how much of the outline a finer mesh loses at the same
resolution.

    python examples/antialias_gradients.py --out antialias_gradients.png

Requires Pillow (``pip install -e . --group examples``).
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

import diffrast as dr
from common import (
    ACCENT,
    BACKGROUND,
    CLAY,
    MUTED,
    PANEL,
    TEXT,
    Mesh,
    downsample,
    icosphere,
    look_at,
    orbit,
    perspective,
    rgb,
    titled_panels,
    to_clip,
    to_image,
    to_srgb,
)

TARGET_OFFSET = (0.16, 0.10, 0.0)


def render_silhouette(
    ctx: dr.RasterizeContext, mesh: Mesh, mvp: Tensor, resolution: int, antialias: bool
) -> Tensor:
    """Coverage image ``[1, height, width, 1]``: one inside the silhouette, zero outside."""
    pos = to_clip(mesh.positions, mvp)
    rast, _ = dr.rasterize(ctx, pos, mesh.tri, (resolution, resolution))
    white = torch.ones_like(mesh.positions[:, :1])
    coverage, _ = dr.interpolate(white, rast, mesh.tri)
    if antialias:
        coverage = dr.antialias(coverage, rast, pos, mesh.tri)
    return coverage


def vertex_gradient(
    ctx: dr.RasterizeContext, mesh: Mesh, mvp: Tensor, target: Tensor, antialias: bool
) -> tuple[Tensor, Tensor]:
    """The silhouette and the gradient of its squared error with respect to the vertices."""
    positions = mesh.positions.detach().requires_grad_()
    image = render_silhouette(
        ctx, mesh._replace(positions=positions), mvp, target.shape[1], antialias
    )
    loss = F.mse_loss(image, target)
    loss.backward()
    return image.detach(), positions.grad


def blended_share(aliased: Tensor, smooth: Tensor) -> float:
    """Share of the pixel pairs lying across the outline in which ``antialias`` changed a pixel."""
    inside = aliased > 0.5
    changed = smooth != aliased
    last = inside.shape[0] - 1
    pairs = blended = 0
    for axis in (0, 1):
        across = inside.narrow(axis, 0, last) != inside.narrow(axis, 1, last)
        touched = changed.narrow(axis, 0, last) | changed.narrow(axis, 1, last)
        pairs += int(across.sum())
        blended += int((across & touched).sum())
    return blended / max(pairs, 1)


def draw_flat(
    ctx: dr.RasterizeContext, pos: Tensor, tri: Tensor, colour: tuple[float, ...], under: Tensor
) -> Tensor:
    """Triangles in one colour over the image ``under``, with antialiased edges."""
    rast, _ = dr.rasterize(ctx, pos, tri, under.shape[1:3])
    image = torch.where(rast[..., 3:] > 0, under.new_tensor(colour), under)
    return dr.antialias(image, rast, pos, tri)


def draw_mesh(ctx: dr.RasterizeContext, mesh: Mesh, mvp: Tensor, under: Tensor) -> Tensor:
    """Flat-shaded clay mesh with its triangle edges drawn in, over the image ``under``."""
    pos = to_clip(mesh.positions, mvp)
    rast, rast_db = dr.rasterize(ctx, pos, mesh.tri, under.shape[1:3])

    # The triangle id in rast picks the normal of the face behind each pixel.
    a, b, c = mesh.positions[mesh.tri.long()].unbind(1)
    face_normals = F.normalize(torch.linalg.cross(b - a, c - a), dim=-1)
    normal = face_normals[(rast[..., 3].long() - 1).clamp(min=0)]
    light = F.normalize(normal.new_tensor([0.45, 0.85, 0.55]), dim=0)
    diffuse = (normal @ light).clamp(min=0)[..., None]
    colour = to_srgb(normal.new_tensor(CLAY) * (0.2 + 0.3 * diffuse))

    # A barycentric is zero on one edge of its triangle and rast_db says how fast it grows per
    # pixel, so their ratio is the distance to that edge in pixels.
    u, v = rast[..., 0], rast[..., 1]
    du_dx, du_dy, dv_dx, dv_dy = rast_db.unbind(-1)
    weights = torch.stack([u, v, 1 - u - v], -1)
    slopes = torch.stack(
        [
            torch.hypot(du_dx, du_dy),
            torch.hypot(dv_dx, dv_dy),
            torch.hypot(du_dx + dv_dx, du_dy + dv_dy),
        ],
        -1,
    )
    distance = (weights / slopes.clamp(min=1e-8)).amin(-1, keepdim=True)
    colour = colour * (0.6 + 0.4 * (distance - 0.5).clamp(0, 1))

    image = torch.where(rast[..., 3:] > 0, colour, under)
    return dr.antialias(image, rast, pos, mesh.tri)


def arrow_mesh(
    start: Tensor, end: Tensor, width: float, margin: float = 0.0
) -> tuple[Tensor, Tensor]:
    """Flat arrows from ``start`` to ``end`` (``[count, 2]``, in clip space) as triangles.

    ``margin`` moves every edge of the arrows outwards by that distance.
    """
    along = end - start
    length = along.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    along = along / length
    side = torch.stack([-along[:, 1], along[:, 0]], -1)
    head = torch.minimum(0.6 * length, torch.full_like(length, 3.5 * width))
    slope = 0.45
    reach = margin * math.hypot(1, slope) / slope

    tail = start - margin * along
    neck = end - (head + margin) * along
    tip = end + reach * along
    half = (0.5 * width + margin) * side
    barb = slope * (head + margin + reach) * side
    corners = torch.stack(
        [tail - half, tail + half, neck + half, neck - half, neck - barb, neck + barb, tip], 1
    )
    pos = F.pad(F.pad(corners.reshape(1, -1, 2), (0, 1)), (0, 1), value=1.0)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6]], device=start.device)
    first = 7 * torch.arange(start.shape[0], device=start.device)
    tri = (faces + first[:, None, None]).reshape(-1, 3).int()
    return pos.contiguous(), tri


def gradient_picture(
    ctx: dr.RasterizeContext, mesh: Mesh, target: Mesh, mvp: Tensor, grad: Tensor, size: int
) -> Tensor:
    """The mesh in front of the target silhouette, with an arrow down the gradient per vertex."""
    image = grad.new_tensor(PANEL).expand(1, size, size, 3)
    image = draw_flat(ctx, to_clip(target.positions, mvp), target.tri, ACCENT, image)
    image = draw_mesh(ctx, mesh, mvp, image)

    moved = grad.norm(dim=-1) > 0
    if moved.any():
        # The longest arrow stands for a step of 0.45 units; the others are in proportion.
        step = 0.45 / grad.norm(dim=-1).max()
        start = to_clip(mesh.positions[moved], mvp)[0]
        end = to_clip(mesh.positions[moved] - step * grad[moved], mvp)[0]
        start, end = start[:, :2] / start[:, 3:], end[:, :2] / end[:, 3:]
        # A dark rim under each arrow keeps it readable on the clay and on the target.
        for colour, margin in ((BACKGROUND, 0.01), (TEXT, 0.0)):
            pos, tri = arrow_mesh(start, end, 0.026, margin)
            image = draw_flat(ctx, pos, tri, colour, image)
    return image


def edge_window(changed: Tensor, size: int) -> tuple[int, int]:
    """Lower-left corner of a ``size`` pixel window centred on the upper left of the outline."""
    rows, cols = changed.nonzero(as_tuple=True)
    pick = (rows - cols).argmax()
    limit = changed.shape[0] - size
    row = int((rows[pick] - size // 2).clamp(0, limit))
    col = int((cols[pick] - size // 2).clamp(0, limit))
    return row, col


def coverage_colours(coverage: Tensor) -> Tensor:
    """Clay on the panel colour, mixed by coverage so that in-between values show as they are."""
    clay = to_srgb(coverage.new_tensor(CLAY) * 0.6)
    return torch.lerp(coverage.new_tensor(PANEL), clay, coverage)


def enlarge(image: Tensor, size: int) -> Image.Image:
    """Nearest-neighbour magnification, so every pixel stays a flat square."""
    return to_image(image).resize((size, size), Image.Resampling.NEAREST)


def figure(
    ctx: dr.RasterizeContext,
    mesh: Mesh,
    target: Mesh,
    mvp: Tensor,
    images: tuple[Tensor, Tensor],
    grads: tuple[Tensor, Tensor],
    scale: int,
    zoom: int,
) -> Image.Image:
    """Close-ups of one edge, the pixels ``antialias`` changed, and both gradients on the mesh."""
    aliased, smooth = (image[0] for image in images)
    resolution = aliased.shape[0]
    size = resolution * scale
    difference = (smooth - aliased).abs()
    changed = difference[..., 0] > 0

    window = max(2, size // zoom)
    row, col = edge_window(changed, window)
    close_ups = [
        enlarge(coverage_colours(image[row : row + window, col : col + window]), size)
        for image in (aliased, smooth)
    ]

    # Every changed pixel is drawn at least half bright, so that faint changes stay visible.
    strength = torch.where(changed[..., None], 0.5 + difference, difference)
    change_map = enlarge(
        torch.lerp(strength.new_tensor(PANEL), strength.new_tensor(TEXT), strength), size
    )
    left, top = col * scale, (resolution - row - window) * scale
    box = (left - 2, top - 2, left + window * scale + 1, top + window * scale + 1)
    ImageDraw.Draw(change_map).rectangle(box, outline=rgb(MUTED), width=2)

    # Rendered at twice the size and averaged down.
    with torch.no_grad():
        pictures = [
            to_image(downsample(gradient_picture(ctx, mesh, target, mvp, grad, 2 * size), 2)[0])
            for grad in grads
        ]
    vertices = mesh.positions.shape[0]
    reached = [int((grad.norm(dim=-1) > 0).sum()) for grad in grads]
    cells = [
        ("without antialias", f"edge pixels, {zoom}x", close_ups[0]),
        ("with antialias", f"edge pixels, {zoom}x", close_ups[1]),
        ("changed pixels", f"{int(changed.sum())} of {changed.numel()} pixels", change_map),
        ("gradient, no antialias", f"{reached[0]} of {vertices} vertices", pictures[0]),
        ("gradient, with antialias", f"{reached[1]} of {vertices} vertices", pictures[1]),
    ]
    titles, details, images = zip(*cells, strict=True)
    return titled_panels(images, titles, details, columns=len(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resolution", type=int, default=154, help="size of the rendered image")
    parser.add_argument("--subdivisions", type=int, default=1, help="icosphere subdivisions")
    parser.add_argument("--scale", type=int, default=2, help="figure pixels per image pixel")
    parser.add_argument("--zoom", type=int, default=14, help="magnification of the close-ups")
    parser.add_argument("--out", help="write the figure to this PNG")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    ctx = dr.RasterizeContext(device)
    mesh = icosphere(args.subdivisions, device)
    target = mesh._replace(positions=mesh.positions + mesh.positions.new_tensor(TARGET_OFFSET))
    eye = orbit(torch.tensor([0.9], device=device), torch.tensor([0.6], device=device), 6.0)
    mvp = perspective(30, 1.0, device=device) @ look_at(eye)

    with torch.no_grad():
        goal = render_silhouette(ctx, target, mvp, args.resolution, antialias=True)
    aliased, grad_aliased = vertex_gradient(ctx, mesh, mvp, goal, antialias=False)
    smooth, grad_smooth = vertex_gradient(ctx, mesh, mvp, goal, antialias=True)

    vertices = mesh.positions.shape[0]
    for name, grad in (("without antialias", grad_aliased), ("with antialias", grad_smooth)):
        reached = int((grad.norm(dim=-1) > 0).sum())
        print(
            f"{name:18s} gradient norm {grad.norm().item():.3e}, "
            f"{reached} of {vertices} vertices receive gradient"
        )
    changed = int((smooth != aliased).sum())
    share = blended_share(aliased[0, ..., 0], smooth[0, ..., 0])
    print(f"antialias changed {changed} of {smooth.numel()} pixels")
    print(f"it blended {share:.0%} of the pixel pairs that lie across the outline")

    if args.out:
        picture = figure(
            ctx,
            mesh,
            target,
            mvp,
            (aliased, smooth),
            (grad_aliased, grad_smooth),
            args.scale,
            args.zoom,
        )
        picture.save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
