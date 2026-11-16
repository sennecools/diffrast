"""Meshes, cameras, colours and figure layout shared by the examples.

Only boilerplate lives here. Each example spells out its own chain of diffrast ops.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

BACKGROUND = (0.071, 0.067, 0.063)
PANEL = (0.102, 0.094, 0.090)
TEXT = (0.941, 0.918, 0.875)
MUTED = (0.62, 0.60, 0.57)
ACCENT = (1.0, 0.31, 0.12)

# The colours above are display values. These two are linear albedos: the warm clay of whatever
# an example optimises, and the near black of what is printed on it.
CLAY = (0.85, 0.82, 0.78)
INK = (0.03, 0.028, 0.026)


class Mesh(NamedTuple):
    """Triangle mesh with separate index buffers for positions and texture coordinates.

    Positions are shared across texture seams, so ``tri`` describes the surface itself and is the
    index buffer to rasterize and antialias with. ``tri_uv`` indexes ``uvs``, which are cut along
    the seams. Meshes without texture coordinates leave both as ``None``.
    """

    positions: Tensor
    tri: Tensor
    uvs: Tensor | None = None
    tri_uv: Tensor | None = None


def uv_sphere(rings: int = 48, segments: int = 96, device: torch.device | str = "cpu") -> Mesh:
    """Unit sphere with an equirectangular texture layout; the poles are single vertices."""
    theta = torch.linspace(0, math.pi, rings + 1)
    return lathe(torch.sin(theta), torch.cos(theta), segments, device)


def lathe(
    radius: Tensor, height: Tensor, segments: int = 96, device: torch.device | str = "cpu"
) -> Mesh:
    """Surface of revolution around the y axis.

    ``radius`` and ``height`` give the profile from top to bottom. It must start and end on the
    axis, where the surface closes in a single vertex. ``u`` runs around the axis and ``v`` along
    the profile, from 1 at the top to 0 at the bottom.
    """
    radius = radius.to(device, torch.float32)
    height = height.to(device, torch.float32)
    rings = radius.shape[0] - 1
    phi = torch.linspace(0, 2 * math.pi, segments + 1, device=device)

    inner_r, inner_h = radius[1:-1, None], height[1:-1, None]
    ring_xyz = torch.stack(
        [
            inner_r * torch.cos(phi[None, :-1]),
            inner_h.expand(-1, segments),
            -inner_r * torch.sin(phi[None, :-1]),
        ],
        -1,
    ).reshape(-1, 3)
    poles = torch.zeros(2, 3, device=device)
    poles[:, 1] = height[[0, -1]]
    positions = torch.cat([poles[:1], ring_xyz, poles[1:]])

    steps = torch.hypot(radius.diff(), height.diff())
    along = F.pad(steps.cumsum(0), (1, 0)) / steps.sum()
    u, v = torch.meshgrid(phi / (2 * math.pi), 1 - along, indexing="xy")
    uvs = torch.stack([u, v], -1).reshape(-1, 2)

    def pos_index(r, s):
        if r == 0:
            return 0
        if r == rings:
            return positions.shape[0] - 1
        return 1 + (r - 1) * segments + s % segments

    def uv_index(r, s):
        return r * (segments + 1) + s

    tri, tri_uv = [], []
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
                tri.append([pos_index(*corner) for corner in corners])
                tri_uv.append([uv_index(*corner) for corner in corners])
    tri = torch.tensor(tri, dtype=torch.int32, device=device)
    tri_uv = torch.tensor(tri_uv, dtype=torch.int32, device=device)
    return Mesh(positions, tri, uvs, tri_uv)


def torus_knot(
    p: int = 2,
    q: int = 3,
    segments: int = 384,
    sides: int = 48,
    tube: float = 0.2,
    device: torch.device | str = "cpu",
    flat: bool = False,
    flip_v: bool = False,
) -> Mesh:
    """Tube around a ``(p, q)`` torus knot that fits in the unit sphere.

    The knot winds around the z axis. ``flat`` lays it down instead, with the y axis through its
    middle, so that a camera above it sees the lobes apart and a turn about y is a turntable.

    ``u`` runs along the knot and ``v`` around the tube. Seen from outside the tube that frame is
    left-handed, so a texture appears as its mirror image. Stripes do not mind, lettering does:
    ``flip_v`` reverses ``v`` so that text reads the right way round.
    """

    def curve(t):
        r = 2 + torch.cos(q * t)
        return torch.stack([r * torch.cos(p * t), r * torch.sin(p * t), -torch.sin(q * t)], -1)

    # Frenet frame from finite differences, in float64 so the second difference is accurate.
    t = torch.arange(segments, dtype=torch.float64) / segments * 2 * math.pi
    step = 1e-4
    centre, ahead, behind = curve(t), curve(t + step), curve(t - step)
    tangent = F.normalize(ahead - behind, dim=-1)
    bend = ahead - 2 * centre + behind
    normal = F.normalize(bend - (bend * tangent).sum(-1, keepdim=True) * tangent, dim=-1)
    binormal = torch.linalg.cross(tangent, normal)

    scale = 1 / (3 + 3 * tube)
    angle = torch.arange(sides, dtype=torch.float64) / sides * 2 * math.pi
    around = (
        torch.cos(angle)[:, None] * normal[:, None] + torch.sin(angle)[:, None] * binormal[:, None]
    )
    positions = ((centre[:, None] + 3 * tube * around) * scale).reshape(-1, 3).float().to(device)
    if flat:
        # A quarter turn about the x axis.
        positions = positions[:, [0, 2, 1]] * positions.new_tensor([1.0, 1.0, -1.0])

    i, j = torch.meshgrid(torch.arange(segments), torch.arange(sides), indexing="ij")
    i, j = i.flatten(), j.flatten()

    def pos_index(a, b):
        return (a % segments) * sides + b % sides

    def uv_index(a, b):
        return a * (sides + 1) + b

    def quads(index):
        a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
        return torch.cat([torch.stack([a, c, b], -1), torch.stack([a, d, c], -1)])

    u, v = torch.meshgrid(
        torch.linspace(0, 1, segments + 1), torch.linspace(0, 1, sides + 1), indexing="ij"
    )
    uvs = torch.stack([u, 1 - v if flip_v else v], -1).reshape(-1, 2).to(device)
    tri = quads(pos_index).to(torch.int32).to(device)
    tri_uv = quads(uv_index).to(torch.int32).to(device)
    return Mesh(positions, tri, uvs, tri_uv)


def icosphere(subdivisions: int = 4, device: torch.device | str = "cpu") -> Mesh:
    """Unit sphere made of near-equal triangles, without texture coordinates."""
    g = (1 + math.sqrt(5)) / 2
    positions = torch.tensor(
        [
            [-1, g, 0], [1, g, 0], [-1, -g, 0], [1, -g, 0],
            [0, -1, g], [0, 1, g], [0, -1, -g], [0, 1, -g],
            [g, 0, -1], [g, 0, 1], [-g, 0, -1], [-g, 0, 1],
        ],
        dtype=torch.float64,
    )  # fmt: skip
    tri = torch.tensor(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
    )  # fmt: skip
    positions = F.normalize(positions, dim=-1)
    for _ in range(subdivisions):
        positions, tri = _subdivide(positions, tri)
        positions = F.normalize(positions, dim=-1)
    return Mesh(positions.float().to(device), tri.to(torch.int32).to(device))


def _subdivide(positions: Tensor, tri: Tensor) -> tuple[Tensor, Tensor]:
    """Split every triangle in four, sharing the new vertex on each edge."""
    edges = torch.cat([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).sort(-1).values
    unique, inverse = torch.unique(edges, dim=0, return_inverse=True)
    midpoints = positions[unique].mean(1)
    mid = inverse.view(3, -1) + positions.shape[0]
    a, b, c = tri.unbind(-1)
    ab, bc, ca = mid
    tri = torch.cat(
        [
            torch.stack([a, ab, ca], -1),
            torch.stack([b, bc, ab], -1),
            torch.stack([c, ca, bc], -1),
            torch.stack([ab, bc, ca], -1),
        ]
    )
    return torch.cat([positions, midpoints]), tri


def load_obj(path: str, device: torch.device | str = "cpu") -> Mesh:
    """Read positions, texture coordinates and faces from a Wavefront OBJ file.

    Polygons are split into triangle fans. The mesh is centred and scaled to the unit sphere.
    """
    positions, uvs, tri, tri_uv = [], [], [], []
    with open(path) as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                positions.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif parts[0] == "f":
                corners = [corner.split("/") for corner in parts[1:]]
                pos_ids = [_obj_index(c[0], len(positions)) for c in corners]
                has_uv = all(len(c) > 1 and c[1] for c in corners)
                uv_ids = [_obj_index(c[1], len(uvs)) for c in corners] if has_uv else None
                for k in range(1, len(corners) - 1):
                    tri.append([pos_ids[0], pos_ids[k], pos_ids[k + 1]])
                    if uv_ids:
                        tri_uv.append([uv_ids[0], uv_ids[k], uv_ids[k + 1]])

    positions = torch.tensor(positions, dtype=torch.float32, device=device)
    positions = positions - (positions.amax(0) + positions.amin(0)) / 2
    positions = positions / positions.norm(dim=-1).max()
    tri = torch.tensor(tri, dtype=torch.int32, device=device)
    if len(tri_uv) != len(tri):
        return Mesh(positions, tri)
    return Mesh(
        positions,
        tri,
        torch.tensor(uvs, dtype=torch.float32, device=device),
        torch.tensor(tri_uv, dtype=torch.int32, device=device),
    )


def _obj_index(token: str, count: int) -> int:
    index = int(token)
    return index - 1 if index > 0 else count + index


def vertex_normals(positions: Tensor, tri: Tensor) -> Tensor:
    """Area-weighted vertex normals for positions of shape ``[..., vertices, 3]``."""
    tri = tri.long()
    v0, v1, v2 = positions[..., tri, :].unbind(-2)
    face = torch.linalg.cross(v1 - v0, v2 - v0, dim=-1)
    normals = torch.zeros_like(positions).index_add(
        -2, tri.flatten(), face.repeat_interleave(3, dim=-2)
    )
    return F.normalize(normals, dim=-1)


def perspective(
    fov_y: float,
    aspect: float,
    near: float = 0.1,
    far: float = 20.0,
    device: torch.device | str = "cpu",
) -> Tensor:
    """OpenGL-style projection matrix for a vertical field of view in degrees."""
    f = 1 / math.tan(math.radians(fov_y) / 2)
    return torch.tensor(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )


def look_at(eye: Tensor, target: Tensor | None = None, up: Tensor | None = None) -> Tensor:
    """View matrices ``[batch, 4, 4]`` for cameras at ``eye`` (``[batch, 3]``) facing ``target``."""
    target = torch.zeros_like(eye) if target is None else target.expand_as(eye)
    up = eye.new_tensor([0.0, 1.0, 0.0]) if up is None else up
    forward = F.normalize(target - eye, dim=-1)
    right = F.normalize(torch.linalg.cross(forward, up.expand_as(eye), dim=-1), dim=-1)
    true_up = torch.linalg.cross(right, forward, dim=-1)
    rotation = torch.stack([right, true_up, -forward], -2)
    view = torch.eye(4, device=eye.device).repeat(eye.shape[0], 1, 1)
    view[:, :3, :3] = rotation
    view[:, :3, 3] = -(rotation @ eye[..., None]).squeeze(-1)
    return view


def orbit(yaw: Tensor, pitch: Tensor, distance: float) -> Tensor:
    """Camera positions ``[batch, 3]`` on a sphere around the origin; angles in radians."""
    return distance * torch.stack(
        [torch.sin(yaw) * torch.cos(pitch), torch.sin(pitch), torch.cos(yaw) * torch.cos(pitch)],
        -1,
    )


def to_clip(positions: Tensor, matrix: Tensor) -> Tensor:
    """Transform ``[vertices, 3]`` positions by ``[batch, 4, 4]`` matrices to clip space."""
    homogeneous = F.pad(positions, (0, 1), value=1.0)
    return (homogeneous @ matrix.transpose(-1, -2)).contiguous()


def to_srgb(linear: Tensor) -> Tensor:
    """Linear radiance to display values, with a filmic shoulder so highlights roll off."""
    x = linear.clamp(min=0)
    x = x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14)
    return x.clamp(0, 1) ** (1 / 2.2)


def from_srgb(
    display: Sequence[float] | Tensor, device: torch.device | str | None = None
) -> Tensor:
    """Inverse of ``to_srgb``: the linear colour that it maps to ``display``.

    A render that fills its backdrop with ``from_srgb(PANEL)`` can be antialiased and tone-mapped
    as a whole, and the backdrop still comes out as the panel colour exactly.
    """
    t = torch.as_tensor(display, dtype=torch.float32, device=device) ** 2.2
    a, b, c = 2.43 * t - 2.51, 0.59 * t - 0.03, 0.14 * t
    return (-b - torch.sqrt(b * b - 4 * a * c)) / (2 * a)


def lighting(
    normal: Tensor,
    key: Sequence[float],
    fill: Sequence[float],
    ambient: float = 0.22,
    fill_colour: Sequence[float] = (0.05, 0.065, 0.09),
) -> Tensor:
    """Linear light ``[..., 3]`` that falls on surfaces with the given normals.

    ``key`` and ``fill`` are the directions towards a white key light and a faint cool fill. A
    surface that faces the key light receives one; a surface turned away from both lights is left
    with ``ambient``.
    """
    normal = F.normalize(normal, dim=-1)

    def lambert(direction: Sequence[float]) -> Tensor:
        towards = F.normalize(normal.new_tensor(direction), dim=0)
        return (normal * towards).sum(-1, keepdim=True).clamp(min=0)

    return ambient + (1 - ambient) * lambert(key) + normal.new_tensor(fill_colour) * lambert(fill)


def downsample(image: Tensor, factor: int) -> Tensor:
    """Average ``factor`` by ``factor`` blocks of an image ``[..., height, width, channels]``.

    The examples render their pictures larger than needed and average them down after
    ``to_srgb``. Averaged in linear light, a pixel half on a bright surface and half on the dark
    backdrop would come out almost as bright as the surface, and the outline would look jagged.
    Only images that are still to be blended, like the layers of depth_peeling.py, are averaged
    before ``to_srgb``.
    """
    if factor == 1:
        return image
    return F.avg_pool2d(image.movedim(-1, -3), factor).movedim(-3, -1)


def psnr(image: Tensor, reference: Tensor) -> float:
    """Peak signal-to-noise ratio in dB between two tensors, for a peak value of one."""
    return -10 * math.log10(max(F.mse_loss(image, reference).item(), 1e-12))


def to_image(image: Tensor, size: int | None = None) -> Image.Image:
    """Tensor ``[height, width, 3]`` with row 0 at the bottom to a PIL image ``size`` wide."""
    array = (image.detach().flip(0).clamp(0, 1) * 255).round().byte().cpu().numpy()
    picture = Image.fromarray(array)
    if size is not None and size != picture.width:
        height = round(size * picture.height / picture.width)
        picture = picture.resize((size, height), Image.Resampling.LANCZOS)
    return picture


def rgb(color: Sequence[float]) -> tuple[int, int, int]:
    """A colour with components in ``[0, 1]`` as the bytes that Pillow draws with."""
    return tuple(round(255 * c) for c in color)


def font(size: int) -> ImageFont.FreeTypeFont:
    """Pillow's built-in font, ``size`` pixels high."""
    return ImageFont.load_default(size=size)


def text_mask(
    text: str, width: int, height: int, size: int, device: torch.device | str = "cpu"
) -> Tensor:
    """Text rendered into a ``[height, width]`` mask in ``[0, 1]``, with row 0 at the bottom."""
    image = Image.new("L", (width, height), 0)
    centre = (width / 2, height / 2)
    ImageDraw.Draw(image).text(centre, text, fill=255, font=font(size), anchor="mm")
    return torch.from_numpy(np.array(image)).to(device).float().flip(0) / 255


def layout(width: int, count: int) -> tuple[int, int, int]:
    """Gap, panel width and label size for ``count`` panels in a row ``width`` pixels wide.

    Passed on to ``panels``, they give a figure of exactly that width. An animation 960 pixels
    wide gets 18-pixel labels, which can still be read at half size.
    """
    gap = width // 64
    # A few pixels more between the panels let them fill the width exactly.
    gap += (width - (count + 1) * gap) % count
    return gap, (width - (count + 1) * gap) // count, width // 53


def panels(
    images: Sequence[Image.Image],
    labels: Sequence[str],
    gap: int = 16,
    label_size: int = 18,
    *,
    details: Sequence[str] | None = None,
    columns: int | None = None,
) -> Image.Image:
    """Lay images out in a row on the background colour, each with a label above it.

    ``columns`` wraps the row after that many images. ``details`` adds a second line under every
    label; the label is then drawn bright and the detail muted.
    """
    columns = columns or len(images)
    cells = list(zip(images, labels, details or labels, strict=True))
    rows = [cells[start : start + columns] for start in range(0, len(cells), columns)]
    heights = [max(image.height for image, _, _ in row) for row in rows]
    header = label_size + gap if details is None else round(2.9 * label_size)
    width = max(sum(image.width + gap for image, _, _ in row) for row in rows) + gap
    height = sum(header + row_height + gap for row_height in heights) + gap
    canvas = Image.new("RGB", (width, height), rgb(BACKGROUND))
    draw = ImageDraw.Draw(canvas)
    y = gap
    for row, row_height in zip(rows, heights, strict=True):
        x = gap
        for image, label, detail in row:
            if details is None:
                draw.text((x, y), label, fill=rgb(MUTED), font=font(label_size))
            else:
                draw.text((x, y), label, fill=rgb(TEXT), font=font(label_size))
                below = (x, y + round(1.3 * label_size))
                draw.text(below, detail, fill=rgb(MUTED), font=font(label_size))
            canvas.paste(image, (x, y + header))
            x += image.width + gap
        y += header + row_height + gap
    return canvas


def titled_panels(
    images: Sequence[Image.Image], titles: Sequence[str], details: Sequence[str], columns: int
) -> Image.Image:
    """Panels in rows of ``columns``, each under a bright title and a muted line of detail.

    Gaps and text follow the size of the panels. A figure 1660 pixels wide gets 28-pixel text,
    a little smaller than the labels of ``layout`` so that two lines fit above a narrow panel.
    """
    scale = images[0].width * columns / 1560
    gap, text = max(4, round(20 * scale)), max(8, round(28 * scale))
    return panels(images, titles, gap, text, details=details, columns=columns)


def stack(rows: Sequence[Image.Image], overlap: int = 0) -> Image.Image:
    """Figures below each other on the background colour, aligned on the left.

    Every figure brings its own margin. Letting neighbours share ``overlap`` pixels keeps the
    gap between two figures equal to the gaps inside them.
    """
    height = sum(row.height - overlap for row in rows) + overlap
    canvas = Image.new("RGB", (max(row.width for row in rows), height), rgb(BACKGROUND))
    # Pasted from the bottom up: a figure may draw into its lower margin, as with a frame around
    # a panel, while its upper margin is empty.
    bottom = height
    for row in reversed(rows):
        canvas.paste(row, (0, bottom - row.height))
        bottom -= row.height - overlap
    return canvas


def save_animation(
    frames: Sequence[Image.Image],
    path: str,
    frame_ms: int = 60,
    hold_ms: int = 2500,
    *,
    holds: Mapping[int, int] | None = None,
    seamless: bool = False,
    dither: bool = True,
) -> None:
    """Write frames as an animation, in the format that the extension of ``path`` names.

    ``.gif`` and ``.webp`` are written by Pillow. ``.mp4`` (H.264) and ``.webm`` (VP9) are
    encoded by ffmpeg, which has to be on the PATH.

    Every frame is shown for ``frame_ms`` milliseconds and the last one for ``hold_ms``.
    ``holds`` maps the index of a frame to a longer time for that frame. ``seamless`` is for a
    loop whose last frame leads back into the first: the last frame is then not held.

    A GIF has 256 colours, and here all frames share them. The flat colours of a figure (canvas,
    panels, labels and any other colour that fills a visible share of a frame) are stored as they
    are. Shading in between is dithered, unless ``dither`` is off: that leaves faint bands on
    smooth surfaces, and a smaller file when much of every frame changes.
    """
    frames = [frame.convert("RGB") for frame in frames]
    durations = [frame_ms] * len(frames)
    if not seamless:
        durations[-1] = hold_ms
    for index, duration in (holds or {}).items():
        durations[index] = duration

    suffix = Path(path).suffix.lower()
    if suffix in _VIDEO_CODECS:
        # A video has one frame rate, so a frame that stays longer is sent several times.
        repeats = [max(1, round(duration / frame_ms)) for duration in durations]
        _encode_video(frames, repeats, frame_ms, path)
    elif suffix == ".webp":
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=durations, loop=0, quality=88
        )
    else:
        frames = _to_shared_palette(frames, dither)
        frames[0].save(
            path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
        )


# A colour that fills at least this share of a frame counts as flat and is never dithered.
_FLAT_SHARE = 0.001
# Dithering moves a pixel by at most this many levels before it is matched to the palette.
_DITHER_LEVELS = 3.0
# Without dithering, other palette colours keep this distance in levels from the flat ones. Noise
# around a flat colour then falls onto it instead of flickering between two entries, which a GIF
# compresses badly. With dithering, such a gap in the palette would show as a band.
_FLAT_MARGIN = 5.0
_VIDEO_CODECS = {
    ".mp4": ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-movflags", "+faststart"],
    ".webm": ["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-row-mt", "1"],
}


def _count_colours(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distinct colours ``[count, 3]`` of RGB pixels, where each pixel is among them, and counts."""
    codes = pixels.reshape(-1, 3).astype(np.uint32) @ np.array([65536, 256, 1], dtype=np.uint32)
    codes, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    colours = np.stack([codes >> 16, codes >> 8 & 255, codes & 255], -1).astype(np.uint8)
    return colours, inverse, counts


def _closest(colours: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each colour ``[count, 3]`` the index of the closest palette entry, and its distance."""
    palette = torch.from_numpy(palette).float()
    found = [
        torch.cdist(part.float(), palette, compute_mode="donot_use_mm_for_euclid_dist").min(1)
        for part in torch.from_numpy(colours).split(1 << 16)
    ]
    index = torch.cat([part.indices for part in found]).numpy()
    return index, torch.cat([part.values for part in found]).numpy()


def _shared_palette(arrays: Sequence[np.ndarray], margin: float, size: int = 255) -> np.ndarray:
    """Colours ``[count, 3]`` for a set of frames: the flat ones exactly, the rest fitted.

    A GIF palette has 256 entries, and ``size`` leaves one of them free. Pillow uses it to mark
    the pixels that did not change since the previous frame, which then compress to almost
    nothing.
    """
    present = np.zeros((0, 3), dtype=np.uint8)
    for array in arrays:
        present = _count_colours(np.concatenate([present, _count_colours(array)[0]]))[0]
        if len(present) > 256:
            break
    else:
        # Frames that were reduced to a palette before are stored as they are.
        return present

    sample = np.concatenate([a.reshape(-1, 3) for a in arrays[:: max(1, len(arrays) // 8)]])
    colours, inverse, counts = _count_colours(sample)
    flat = np.array([rgb(c) for c in (BACKGROUND, PANEL, TEXT, MUTED, ACCENT)], dtype=np.uint8)
    for index in np.argsort(-counts)[:256]:
        if counts[index] < _FLAT_SHARE * len(sample) or len(flat) == 64:
            break
        if _closest(colours[index : index + 1], flat)[1] >= margin:
            flat = np.concatenate([flat, colours[index : index + 1]])

    shaded = sample[(_closest(colours, flat)[1] >= margin)[inverse]]
    if len(shaded) == 0:
        return flat
    shaded = shaded[:: max(1, len(shaded) // 2_000_000)]
    # This quantiser spreads its colours over everything that occurs. Median cut would spend
    # them on the large surfaces and leave a small one, like a highlight, with too few.
    fitted = Image.fromarray(shaded[None]).quantize(
        size - len(flat), Image.Quantize.MAXCOVERAGE, kmeans=3
    )
    fitted = np.array(fitted.getpalette(), dtype=np.uint8).reshape(-1, 3)[: size - len(flat)]
    return np.concatenate([flat, fitted[_closest(fitted, flat)[1] >= margin]])


def _dither_pattern(height: int, width: int) -> np.ndarray:
    """Ordered-dither offsets ``[height, width, 1]``, spread evenly between -1 and 1."""
    tile = np.zeros((1, 1))
    for _ in range(3):
        tile = np.block([[4 * tile, 4 * tile + 2], [4 * tile + 3, 4 * tile + 1]])
    tile = (2 * tile + 1) / tile.size - 1
    return np.tile(tile, (height // 8 + 1, width // 8 + 1))[:height, :width, None]


def _to_shared_palette(frames: Sequence[Image.Image], dither: bool) -> list[Image.Image]:
    """The frames as palette images that all use the colours of ``_shared_palette``."""
    arrays = [np.asarray(frame) for frame in frames]
    # Distinct colours are at least one level apart, so a margin of one only avoids duplicates.
    palette = _shared_palette(arrays, margin=1.0 if dither else _FLAT_MARGIN)
    # An ordered pattern, not error diffusion: it is the same in every frame, so still surfaces
    # do not shimmer and still compress, and no pixel can end up far from its colour.
    offsets = _DITHER_LEVELS * _dither_pattern(*arrays[0].shape[:2])

    reduced = []
    for array in arrays:
        colours, inverse, counts = _count_colours(array)
        index, distance = _closest(colours, palette)
        index = index[inverse]
        if dither:
            # Colours that the palette holds stay as they are too, however few pixels have them.
            flat = (counts >= _FLAT_SHARE * len(inverse)) | (distance == 0)
            shifted = np.rint(array + offsets).clip(0, 255).astype(np.uint8)
            shifted, place, _ = _count_colours(shifted)
            index = np.where(flat[inverse], index, _closest(shifted, palette)[0][place])
        picture = Image.fromarray(index.reshape(array.shape[:2]).astype(np.uint8))
        picture.putpalette(palette.tobytes())
        reduced.append(picture)
    return reduced


def _encode_video(
    frames: Sequence[Image.Image], repeats: Sequence[int], frame_ms: int, path: str
) -> None:
    """Pipe RGB frames to ffmpeg; ``repeats`` says how often each one is sent."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Writing {Path(path).suffix} needs ffmpeg on the PATH. "
            "A .gif or .webp is written without it."
        )
    # yuv420p stores colour at half resolution, which takes an even width and height. The canvas
    # ends in a margin, so repeating its last row or column does not show.
    width, height = frames[0].size
    pad = ((0, height % 2), (0, width % 2), (0, 0))
    # Converted with exact rounding, which keeps the flat colours within a level of the frames,
    # and tagged as BT.709, so that a player does not have to guess how to get back to RGB.
    # Recent versions of ffmpeg take the tags from the filter, older ones from the options.
    convert = (
        "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,"
        "setparams=color_primaries=bt709:color_trc=bt709"
    )
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width + width % 2}x{height + height % 2}", "-r", f"1000/{frame_ms}", "-i", "-",
        "-vf", convert, "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        *_VIDEO_CODECS[Path(path).suffix.lower()], str(path),
    ]  # fmt: skip
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame, count in zip(frames, repeats, strict=True):
            data = np.pad(np.asarray(frame), pad, mode="edge").tobytes()
            for _ in range(count):
                encoder.stdin.write(data)
    except OSError:
        pass  # ffmpeg stopped reading; the message it left says why
    _, errors = encoder.communicate()
    if encoder.returncode != 0:
        raise RuntimeError(f"ffmpeg could not write {path}:\n{errors.decode(errors='replace')}")
