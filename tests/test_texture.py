import pytest
import torch

import diffrast as dr
from diffrast.backends import reference

BOUNDARY_MODES = ["wrap", "clamp", "zero"]


def texel_centres(height, width, dtype=torch.float32):
    """``uv`` of every texel centre, shape ``[1, height, width, 2]``."""
    u = (torch.arange(width, dtype=dtype) + 0.5) / width
    v = (torch.arange(height, dtype=dtype) + 0.5) / height
    return torch.stack(torch.meshgrid(u, v, indexing="xy"), -1)[None]


def footprint(uv, texels, size):
    """``uv_da`` for a footprint of ``texels`` texels along both screen axes."""
    da = uv.new_zeros(*uv.shape[:3], 4)
    da[..., 0] = texels / size
    da[..., 3] = texels / size
    return da


@pytest.fixture
def tex():
    return torch.randn(1, 6, 8, 3, generator=torch.Generator().manual_seed(0))


@pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
@pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
def test_texel_centres_return_texels(tex, filter_mode, boundary_mode):
    out = dr.texture(tex, texel_centres(6, 8), filter_mode=filter_mode, boundary_mode=boundary_mode)
    torch.testing.assert_close(out, tex)


def test_nearest_picks_containing_texel(tex):
    uv = torch.tensor([[[[0.01, 0.01], [0.124, 0.99], [0.126, 0.5]]]])
    out = dr.texture(tex, uv, filter_mode="nearest")
    expected = torch.stack([tex[0, 0, 0], tex[0, 5, 0], tex[0, 3, 1]])[None, None]
    torch.testing.assert_close(out, expected)


def test_linear_reproduces_affine_functions():
    height, width = 5, 7
    centres = texel_centres(height, width)
    tex = (2 * centres[..., :1] - 3 * centres[..., 1:] + 0.5).expand(-1, -1, -1, 2).clone()
    uv = torch.rand(1, 10, 10, 2, generator=torch.Generator().manual_seed(1))
    # Stay between the outermost texel centres, where bilinear filtering is exact for affine data.
    uv[..., 0] = 0.5 / width + uv[..., 0] * (1 - 1 / width)
    uv[..., 1] = 0.5 / height + uv[..., 1] * (1 - 1 / height)
    out = dr.texture(tex, uv, filter_mode="linear", boundary_mode="clamp")
    torch.testing.assert_close(out[..., 0], 2 * uv[..., 0] - 3 * uv[..., 1] + 0.5)


@pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
def test_wrap_is_periodic(tex, filter_mode):
    uv = torch.rand(1, 6, 6, 2, generator=torch.Generator().manual_seed(2))
    shifted = uv + torch.tensor([2.0, -1.0])
    a = dr.texture(tex, uv, filter_mode=filter_mode, boundary_mode="wrap")
    b = dr.texture(tex, shifted, filter_mode=filter_mode, boundary_mode="wrap")
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_wrap_blends_across_the_seam(tex):
    uv = torch.tensor([[[[0.0, 0.5 / 6]]]])
    out = dr.texture(tex, uv, filter_mode="linear", boundary_mode="wrap")
    torch.testing.assert_close(out[0, 0, 0], (tex[0, 0, 0] + tex[0, 0, -1]) / 2)


def test_clamp_repeats_edge_texels(tex):
    uv = torch.tensor([[[[-3.0, 0.5 / 6], [0.5 / 8, 7.0]]]])
    out = dr.texture(tex, uv, filter_mode="linear", boundary_mode="clamp")
    torch.testing.assert_close(out[0, 0], torch.stack([tex[0, 0, 0], tex[0, -1, 0]]))


def test_zero_fades_to_zero_outside(tex):
    uv = torch.tensor([[[[-3.0, 0.5], [0.0, 0.5 / 6]]]])
    out = dr.texture(tex, uv, filter_mode="linear", boundary_mode="zero")
    torch.testing.assert_close(out[0, 0, 0], torch.zeros(3))
    torch.testing.assert_close(out[0, 0, 1], tex[0, 0, 0] / 2)


def test_shared_texture_matches_per_view_textures(tex):
    uv = torch.rand(3, 4, 4, 2, generator=torch.Generator().manual_seed(3))
    shared = dr.texture(tex, uv)
    per_view = dr.texture(tex.expand(3, -1, -1, -1).contiguous(), uv)
    torch.testing.assert_close(shared, per_view)


def test_per_view_textures_are_independent():
    tex = torch.stack([torch.zeros(4, 4, 1), torch.ones(4, 4, 1)])
    out = dr.texture(tex, torch.rand(2, 3, 3, 2), boundary_mode="clamp")
    torch.testing.assert_close(out[0], torch.zeros(3, 3, 1))
    torch.testing.assert_close(out[1], torch.ones(3, 3, 1))


class TestMipmaps:
    def test_coarsest_level_is_the_mean(self):
        tex = torch.randn(1, 16, 16, 2, generator=torch.Generator().manual_seed(4))
        uv = torch.rand(1, 4, 4, 2, generator=torch.Generator().manual_seed(5))
        out = dr.texture(tex, uv, footprint(uv, 64, 16))
        torch.testing.assert_close(out, tex.mean((1, 2), keepdim=True).expand_as(out))

    def test_level_one_texel_centres(self):
        # At a level-1 texel centre, bilinear level 0 is the mean of the 2x2 block, which is also
        # the level-1 texel, so every level of detail between 0 and 1 gives the same value.
        tex = torch.randn(1, 8, 8, 1, generator=torch.Generator().manual_seed(6))
        uv = texel_centres(4, 4)
        block_mean = tex.view(1, 4, 2, 4, 2, 1).mean((2, 4))
        for texels in (1.0, 1.3, 2.0):
            out = dr.texture(tex, uv, footprint(uv, texels, 8), filter_mode="linear-mipmap-linear")
            torch.testing.assert_close(out, block_mean)

    def test_trilinear_blends_neighbouring_levels(self, tex):
        uv = torch.rand(1, 5, 5, 2, generator=torch.Generator().manual_seed(7))

        def at_lod(lod):
            return dr.texture(tex, uv, footprint(uv, 2.0**lod, 8))

        torch.testing.assert_close(at_lod(1.25), 0.75 * at_lod(1) + 0.25 * at_lod(2))

    def test_mipmap_nearest_rounds_level_of_detail(self, tex):
        uv = torch.rand(1, 5, 5, 2, generator=torch.Generator().manual_seed(8))

        def sample(lod, mode):
            return dr.texture(tex, uv, footprint(uv, 2.0**lod, 8), filter_mode=mode)

        level_0 = sample(0, "linear-mipmap-linear")
        level_1 = sample(1, "linear-mipmap-linear")
        torch.testing.assert_close(sample(0.4, "linear-mipmap-nearest"), level_0)
        torch.testing.assert_close(sample(0.6, "linear-mipmap-nearest"), level_1)

    def test_magnification_uses_base_level(self, tex):
        uv = torch.rand(1, 5, 5, 2, generator=torch.Generator().manual_seed(9))
        linear = dr.texture(tex, uv, filter_mode="linear")
        mip = dr.texture(tex, uv, footprint(uv, 0.25, 8), filter_mode="linear-mipmap-linear")
        torch.testing.assert_close(mip, linear)

    def test_max_mip_level_zero_disables_mipmaps(self, tex):
        uv = torch.rand(1, 5, 5, 2, generator=torch.Generator().manual_seed(10))
        linear = dr.texture(tex, uv, filter_mode="linear")
        mip = dr.texture(tex, uv, footprint(uv, 16, 8), max_mip_level=0)
        torch.testing.assert_close(mip, linear)

    def test_footprint_uses_longer_axis(self, tex):
        uv = torch.rand(1, 5, 5, 2, generator=torch.Generator().manual_seed(11))
        along_x = torch.zeros(1, 5, 5, 4)
        along_x[..., 0] = 3 / 8
        along_both = footprint(uv, 3, 8)
        along_both[..., 3] = 0.5 / 6
        torch.testing.assert_close(dr.texture(tex, uv, along_x), dr.texture(tex, uv, along_both))

    def test_odd_sizes_keep_constants(self):
        tex = torch.full((1, 5, 7, 2), 0.25)
        uv = torch.rand(1, 4, 4, 2, generator=torch.Generator().manual_seed(12))
        for texels in (1.5, 3.0, 9.0):
            out = dr.texture(tex, uv, footprint(uv, texels, 7))
            torch.testing.assert_close(out, torch.full_like(out, 0.25))


class TestGradients:
    def inputs(self, seed):
        gen = torch.Generator().manual_seed(seed)
        tex = torch.randn(1, 4, 5, 2, generator=gen, dtype=torch.float64, requires_grad=True)
        uv = (
            torch.rand(1, 3, 3, 2, generator=gen, dtype=torch.float64) * 1.4 - 0.2
        ).requires_grad_()
        return tex, uv

    @pytest.mark.parametrize("boundary_mode", BOUNDARY_MODES)
    def test_linear(self, boundary_mode):
        tex, uv = self.inputs(13)
        assert torch.autograd.gradcheck(
            lambda t, c: reference.texture(t, c, None, "linear", boundary_mode, None), (tex, uv)
        )

    def test_nearest_texture_gradient(self):
        tex, uv = self.inputs(14)
        assert torch.autograd.gradcheck(
            lambda t: reference.texture(t, uv.detach(), None, "nearest", "wrap", None), (tex,)
        )

    @pytest.mark.parametrize("filter_mode", ["linear-mipmap-linear", "linear-mipmap-nearest"])
    def test_mipmapped(self, filter_mode):
        tex, uv = self.inputs(15)
        gen = torch.Generator().manual_seed(16)
        # Footprints of 1.2 to 3.2 texels keep the level of detail away from integers and clamps.
        scale = (torch.rand(1, 3, 3, 1, generator=gen, dtype=torch.float64) * 2 + 1.2) / 5
        uv_da = (scale * torch.tensor([1.0, 0.1, -0.2, 0.9], dtype=torch.float64)).requires_grad_()
        assert torch.autograd.gradcheck(
            lambda t, c, d: reference.texture(t, c, d, filter_mode, "wrap", None), (tex, uv, uv_da)
        )


def test_texture_fitting_recovers_target(ctx):
    """Render a textured quad and fit the texture to a target render by gradient descent."""
    pos = torch.tensor(
        [
            [
                [-0.9, -0.9, 0.0, 1.0],
                [0.9, -0.9, 0.0, 1.0],
                [0.9, 0.9, 0.0, 1.0],
                [-0.9, 0.9, 0.0, 1.0],
            ]
        ]
    )
    tri = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    uv_attr = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rast, _ = dr.rasterize(ctx, pos, tri, (32, 32))
    uv, _ = dr.interpolate(uv_attr, rast, tri)
    mask = (rast[..., 3:] > 0).float()

    target_tex = torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(17))
    target = dr.texture(target_tex, uv, boundary_mode="clamp") * mask

    tex = torch.full((1, 8, 8, 3), 0.5, requires_grad=True)
    optimizer = torch.optim.Adam([tex], lr=0.05)
    for _ in range(300):
        optimizer.zero_grad()
        loss = ((dr.texture(tex, uv, boundary_mode="clamp") * mask - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    assert loss.item() < 1e-5
