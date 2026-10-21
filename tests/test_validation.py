import pytest
import torch

import diffrast as dr


def test_rasterize_rejects_wrong_pos_shape(ctx, triangle):
    pos, tri = triangle
    with pytest.raises(ValueError, match="pos must have shape"):
        dr.rasterize(ctx, pos[..., :3], tri, (8, 8))


def test_rasterize_rejects_int64_triangles(ctx, triangle):
    pos, tri = triangle
    with pytest.raises(TypeError, match=r"tri must be torch\.int32"):
        dr.rasterize(ctx, pos, tri.long(), (8, 8))


@pytest.mark.parametrize("resolution", [(8,), (0, 8), (8, -1)])
def test_rasterize_rejects_bad_resolution(ctx, triangle, resolution):
    pos, tri = triangle
    with pytest.raises(ValueError, match="resolution"):
        dr.rasterize(ctx, pos, tri, resolution)


def test_rasterize_rejects_range_mode(ctx, triangle):
    pos, tri = triangle
    with pytest.raises(NotImplementedError, match="Range mode"):
        dr.rasterize(ctx, pos, tri, (8, 8), ranges=torch.zeros(1, 2, dtype=torch.int32))


def test_interpolate_requires_rast_db_for_derivatives(triangle):
    _, tri = triangle
    attr = torch.zeros(3, 2)
    rast = torch.zeros(1, 8, 8, 4)
    with pytest.raises(ValueError, match="rast_db is required"):
        dr.interpolate(attr, rast, tri, diff_attrs="all")


def test_interpolate_rejects_out_of_range_channels(triangle):
    _, tri = triangle
    attr = torch.zeros(3, 2)
    rast = torch.zeros(1, 8, 8, 4)
    with pytest.raises(ValueError, match="out of range"):
        dr.interpolate(attr, rast, tri, rast_db=rast, diff_attrs=[2])


def test_interpolate_rejects_batch_mismatch(triangle):
    _, tri = triangle
    attr = torch.zeros(3, 3, 2)
    rast = torch.zeros(2, 8, 8, 4)
    with pytest.raises(ValueError, match="batch size"):
        dr.interpolate(attr, rast, tri)


def test_texture_mipmap_requires_uv_da():
    tex = torch.zeros(1, 16, 16, 3)
    uv = torch.zeros(1, 8, 8, 2)
    with pytest.raises(ValueError, match="requires uv_da"):
        dr.texture(tex, uv, filter_mode="linear-mipmap-linear")


def test_texture_rejects_unknown_modes():
    tex = torch.zeros(1, 16, 16, 3)
    uv = torch.zeros(1, 8, 8, 2)
    with pytest.raises(ValueError, match="filter_mode"):
        dr.texture(tex, uv, filter_mode="cubic")
    with pytest.raises(ValueError, match="boundary_mode"):
        dr.texture(tex, uv, boundary_mode="mirror")


def test_texture_rejects_unsupported_mip_options():
    tex = torch.zeros(1, 16, 16, 3)
    uv = torch.zeros(1, 8, 8, 2)
    with pytest.raises(NotImplementedError, match="mip_level_bias"):
        dr.texture(tex, uv, mip_level_bias=torch.zeros(1, 8, 8))


def test_antialias_rejects_mismatched_color(triangle):
    pos, tri = triangle
    color = torch.zeros(1, 4, 4, 3)
    rast = torch.zeros(1, 8, 8, 4)
    with pytest.raises(ValueError, match="color must match rast"):
        dr.antialias(color, rast, pos, tri)
