"""Independent implementations used as test oracles."""

import torch


def pixel_centres(height, width, dtype=torch.float32):
    """NDC coordinates of pixel centres, shape ``[height, width, 2]``."""
    x = (torch.arange(width, dtype=dtype) * 2 + 1) / width - 1
    y = (torch.arange(height, dtype=dtype) * 2 + 1) / height - 1
    return torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)


def screen_space_rasterize(pos, tri, height, width, margin=1e-4):
    """Rasterize triangles that lie fully in front of the camera, in screen space.

    Projects the vertices, tests coverage with 2D barycentrics and resolves visibility with
    ``z / w``, which is affine in screen space. Returns ``(tri_id, depth, ambiguous)``, where
    ``tri_id`` is -1 for background and ``ambiguous`` marks pixels whose centre lies within
    ``margin`` of an edge or whose two nearest depths are within ``margin``.
    """
    pos = pos.double()
    ndc = pos[..., :3] / pos[..., 3:]
    v = ndc[:, tri.long()][:, :, None, None]
    p = pixel_centres(height, width, torch.float64)
    px, py = p[..., 0], p[..., 1]

    def edge(i, j):
        return (v[..., j, 0] - v[..., i, 0]) * (py - v[..., i, 1]) - (
            v[..., j, 1] - v[..., i, 1]
        ) * (px - v[..., i, 0])

    lam = torch.stack([edge(1, 2), edge(2, 0), edge(0, 1)], -1)
    lam = lam / lam.sum(-1, keepdim=True)
    inside = (lam >= 0).all(-1)
    depth = (lam * v[..., 2]).sum(-1)
    depth = torch.where(inside, depth, torch.inf)

    near = lam.abs().min(-1).values < margin
    sorted_depth = depth.sort(dim=1).values
    tie = (sorted_depth[:, 1] - sorted_depth[:, 0]) < margin if depth.shape[1] > 1 else False
    best_depth, tri_id = depth.min(dim=1)
    tri_id = torch.where(best_depth.isinf(), -1, tri_id)
    ambiguous = near.any(dim=1) | tie
    return tri_id, best_depth, ambiguous
