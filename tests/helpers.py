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


def screen_space_layers(pos, tri, height, width, margin=1e-4):
    """Covering triangles per pixel sorted by ``(depth, index)``, for triangles in front.

    Returns ``(layers, ambiguous)``: ``layers`` has shape ``[batch, triangles, height, width]``
    and holds triangle indices in layer order, then -1 once a pixel runs out of triangles.
    ``ambiguous`` marks pixels near an edge or with two depths within ``margin``.
    """
    pos = pos.double()
    ndc = pos[..., :3] / pos[..., 3:]
    v = ndc[:, tri.long()][:, :, None, None]
    p = pixel_centres(height, width, torch.float64)
    px, py = p[..., 0], p[..., 1]

    def edge(i, j):
        dx, dy = v[..., j, 0] - v[..., i, 0], v[..., j, 1] - v[..., i, 1]
        return dx * (py - v[..., i, 1]) - dy * (px - v[..., i, 0])

    lam = torch.stack([edge(1, 2), edge(2, 0), edge(0, 1)], -1)
    lam = lam / lam.sum(-1, keepdim=True)
    inside = (lam >= 0).all(-1)
    depth = torch.where(inside, (lam * v[..., 2]).sum(-1), torch.inf)

    # A stable sort keeps equal depths in index order.
    sorted_depth, order = depth.sort(dim=1, stable=True)
    layers = torch.where(sorted_depth.isinf(), -1, order)
    near_edge = (lam.abs().min(-1).values < margin).any(dim=1)
    gaps = sorted_depth[:, 1:] - sorted_depth[:, :-1]
    tie = ((gaps < margin) & (gaps > 0)).any(dim=1)
    return layers, near_edge | tie
