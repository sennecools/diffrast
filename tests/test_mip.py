"""Mip chains, custom mip stacks, and the corner cases of mipmapped lookups."""

import itertools
from fractions import Fraction

import pytest
import torch
import torch.nn.functional as F

import diffrast as dr
from diffrast import backends
from diffrast.backends import reference
from diffrast.ops import MipStack, texture_construct_mip

FILTERS = ("nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear")
MIP_FILTERS = FILTERS[2:]
BOUNDARIES = ("wrap", "clamp", "zero")
INF, NAN = float("inf"), float("nan")


def area_weights(n):
    """Exact weights, ``[ceil(n / 2), n]``, that resample ``n`` texels into equal bins by area."""
    m = (n + 1) // 2
    weights = torch.zeros(m, n, dtype=torch.float64)
    for j, i in itertools.product(range(m), range(n)):
        overlap = min(Fraction((j + 1) * n, m), i + 1) - max(Fraction(j * n, m), i)
        weights[j, i] = float(max(overlap, 0) / Fraction(n, m))
    return weights


def footprint(shape, texels, size):
    """``uv_da`` for a footprint of ``texels`` texels along both screen axes."""
    da = torch.zeros(*shape, 4)
    da[..., 0] = texels / size
    da[..., 3] = texels / size
    return da


def scene(size=(8, 12), tex_batch=1, batch=2, channels=3, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    tex = torch.randn(tex_batch, *size, channels, generator=gen, dtype=dtype)
    uv = torch.rand(batch, 5, 6, 2, generator=gen, dtype=dtype) * 1.6 - 0.3
    uv_da = torch.rand(batch, 5, 6, 4, generator=gen, dtype=dtype) * 0.6 - 0.3
    return tex, uv, uv_da


def gradients(out, inputs, weights=None):
    """Gradients of a weighted sum of ``out``, with zeros for inputs it does not depend on."""
    loss = out.nansum() if weights is None else (out * weights).nansum()
    grads = torch.autograd.grad(loss, inputs, allow_unused=True)
    return [torch.zeros_like(t) if g is None else g for g, t in zip(grads, inputs, strict=True)]


class TestMipChain:
    @pytest.mark.parametrize("size", [(1, 1025), (3, 3), (513, 513), (5, 7)])
    def test_every_texel_has_the_same_weight_in_every_level(self, size):
        tex = torch.rand(1, *size, 1, dtype=torch.float64, requires_grad=True)
        levels = reference._mip_chain(tex, None)
        for level in levels[1:]:
            (weight,) = torch.autograd.grad(level.sum(), tex, retain_graph=True)
            share = level.shape[1] * level.shape[2] / (size[0] * size[1])
            torch.testing.assert_close(weight, torch.full_like(weight, share), rtol=1e-9, atol=0)
        assert levels[-1].shape == (1, 1, 1, 1)
        torch.testing.assert_close(levels[-1].flatten(), tex.detach().mean().view(1))

    def test_sizes_halve_rounding_up(self):
        levels = reference._mip_chain(torch.zeros(2, 5, 12, 3), None)
        sizes = [tuple(level.shape) for level in levels]
        assert sizes == [(2, 5, 12, 3), (2, 3, 6, 3), (2, 2, 3, 3), (2, 1, 2, 3), (2, 1, 1, 3)]
        assert len(reference._mip_chain(torch.zeros(2, 5, 12, 3), 2)) == 3

    @pytest.mark.parametrize("size", [(16, 8), (2, 6), (1, 8), (6, 1), (64, 64)])
    def test_even_sizes_are_the_two_by_two_box_filter(self, size):
        tex = torch.randn(2, *size, 3, generator=torch.Generator().manual_seed(0))
        kernel = (min(size[0], 2), min(size[1], 2))
        box = F.avg_pool2d(tex.permute(0, 3, 1, 2), kernel).permute(0, 2, 3, 1)
        assert torch.equal(reference._mip_chain(tex, 1)[1], box)

    @pytest.mark.parametrize("size", [(5, 7), (6, 9), (1, 13), (10, 3), (3, 3), (4, 4), (15, 2)])
    def test_matches_independent_area_weighting(self, size):
        gen = torch.Generator().manual_seed(1)
        tex = torch.randn(2, *size, 2, generator=gen, dtype=torch.float64)
        expected = tex
        for level in reference._mip_chain(tex, None)[1:]:
            rows, cols = area_weights(expected.shape[1]), area_weights(expected.shape[2])
            expected = torch.einsum("ri,bijc,sj->brsc", rows, expected, cols)
            torch.testing.assert_close(level, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("width", [5, 9, 13, 513, 1025, 1500])
    def test_levels_stay_aligned(self, width):
        # A texture holding u at each texel centre holds nearly u at each coarser texel centre.
        # Averaging a window that hangs over an odd edge would shift levels by up to a texel.
        tex = ((torch.arange(width, dtype=torch.float64) + 0.5) / width).view(1, 1, width, 1)
        for level in reference._mip_chain(tex, None)[1:]:
            size = level.shape[2]
            centres = (torch.arange(size, dtype=torch.float64) + 0.5) / size
            assert (level.flatten() - centres).abs().max() * size < 0.1

    def test_gradcheck(self):
        gen = torch.Generator().manual_seed(2)
        tex = torch.randn(1, 5, 7, 2, generator=gen, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda t: torch.cat([lv.flatten() for lv in reference._mip_chain(t, None)[1:]]), (tex,)
        )

    @pytest.mark.parametrize("size", [(5, 7), (1, 1025), (33, 17)])
    def test_far_minified_lookup_returns_the_mean(self, size):
        gen = torch.Generator().manual_seed(3)
        tex = torch.randn(1, *size, 2, generator=gen)
        uv = torch.rand(1, 3, 3, 2, generator=gen)
        out = dr.texture(tex, uv, footprint((1, 3, 3), 4096.0, 1))
        expected = tex.mean((1, 2), keepdim=True).expand_as(out)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


class TestTopLevelGradient:
    """At the coarsest level the level of detail is clamped, so it has no gradient."""

    def lookup(self, bias, max_mip_level=None, size=(8, 8), uv_da=None):
        gen = torch.Generator().manual_seed(4)
        tex = torch.randn(1, *size, 2, generator=gen, dtype=torch.float64)
        uv = torch.rand(1, 4, 4, 2, generator=gen, dtype=torch.float64)
        return reference.texture(
            tex, uv, uv_da, "linear-mipmap-linear", "wrap", max_mip_level, mip_level_bias=bias
        )

    @pytest.mark.parametrize(
        "max_mip_level, size, top",
        [(None, (8, 8), 3), (0, (8, 8), 0), (1, (8, 8), 1), (None, (1, 1), 0)],
    )
    def test_bias_at_the_top_level(self, max_mip_level, size, top):
        bias = torch.full((1, 4, 4), float(top), dtype=torch.float64, requires_grad=True)
        out = self.lookup(bias, max_mip_level, size)
        (grad,) = torch.autograd.grad(out.sum(), bias)
        assert (grad == 0).all()
        # Zero is the derivative towards coarser levels: the output does not change that way.
        torch.testing.assert_close(self.lookup(bias + 1e-3, max_mip_level, size), out)

    @pytest.mark.parametrize("max_mip_level, texels", [(None, 8.0), (0, 1.0), (None, 20.0)])
    def test_footprint_at_or_past_the_top_level(self, max_mip_level, texels):
        uv_da = footprint((1, 4, 4), texels, 8).double().requires_grad_()
        out = self.lookup(None, max_mip_level, uv_da=uv_da)
        (grad,) = torch.autograd.grad(out.sum(), uv_da)
        assert (grad == 0).all()

    def test_interior_level_uses_the_derivative_towards_coarser(self):
        bias = torch.full((1, 4, 4), 1.0, dtype=torch.float64, requires_grad=True)
        out = self.lookup(bias)
        (grad,) = torch.autograd.grad(out.sum(), bias)
        step = 1e-6
        slope = (self.lookup(bias + step) - out).sum((0, 3)) / step
        assert grad.abs().min() > 1e-3
        torch.testing.assert_close(grad[0], slope.detach(), rtol=1e-5, atol=1e-7)


class TestOutOfRange:
    """Coordinates far outside the texture and values that are not finite."""

    def ramp(self):
        return torch.arange(2048.0).view(1, 1, 2048, 1)

    def sample(self, u, filter_mode, boundary_mode, tex=None):
        uv = torch.tensor([[[[u, 0.5]]]])
        tex = self.ramp() if tex is None else tex
        return dr.texture(tex, uv, filter_mode=filter_mode, boundary_mode=boundary_mode).item()

    @pytest.mark.parametrize(
        "u, filter_mode, expected",
        [
            (1048575.25, "nearest", 512.0),
            (1048576.25, "nearest", 512.0),
            (-2000000.25, "nearest", 1536.0),
            (2000000.25, "linear", 511.5),
            (3e8, "linear", 1023.5),
            (-1e30, "nearest", 0.0),
        ],
    )
    def test_wrap_keeps_the_fractional_part(self, u, filter_mode, expected):
        assert self.sample(u, filter_mode, "wrap") == expected

    @pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
    @pytest.mark.parametrize(
        "u, expected", [(1e30, 2047.0), (-1e30, 0.0), (INF, 2047.0), (-INF, 0.0)]
    )
    def test_clamp_returns_the_nearest_edge(self, filter_mode, u, expected):
        assert self.sample(u, filter_mode, "clamp") == expected

    @pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
    @pytest.mark.parametrize("u", [1e30, -1e30, INF, -INF])
    def test_zero_returns_zero_far_outside(self, filter_mode, u):
        assert self.sample(u, filter_mode, "zero", self.ramp() + 1) == 0.0

    @pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
    @pytest.mark.parametrize(
        "u, boundary_mode", [(NAN, "wrap"), (NAN, "clamp"), (NAN, "zero"), (INF, "wrap")]
    )
    def test_undefined_coordinates_give_nan(self, filter_mode, u, boundary_mode):
        assert self.sample(u, filter_mode, boundary_mode) != self.sample(
            u, filter_mode, boundary_mode
        )

    @pytest.mark.parametrize("filter_mode", FILTERS)
    @pytest.mark.parametrize("boundary_mode", BOUNDARIES)
    def test_bad_pixels_pass_no_gradient(self, filter_mode, boundary_mode):
        tex, uv, uv_da = scene(batch=1)
        uv[0, 0, 0, 0], uv[0, 1, 1, 1], uv[0, 2, 2, 0] = NAN, INF, -1e30
        uv_da[0, 3, 3, 2] = NAN
        mip = "mipmap" in filter_mode
        inputs = [t.requires_grad_() for t in (tex, uv, uv_da)]
        out = dr.texture(
            tex, uv, uv_da if mip else None, filter_mode=filter_mode, boundary_mode=boundary_mode
        )
        expected = torch.zeros(5, 6, dtype=torch.bool)
        expected[0, 0] = True
        expected[1, 1] = boundary_mode == "wrap"
        expected[3, 3] = mip
        assert torch.equal(out.isnan().any(-1)[0], expected)
        assert torch.equal(out.isnan().all(-1)[0], expected)
        for grad in gradients(out, inputs):
            assert torch.isfinite(grad).all()

    @pytest.mark.parametrize("value", [3e38, -3e38, INF, -INF])
    def test_huge_footprint_selects_the_coarsest_level(self, value):
        tex, uv, uv_da = scene(size=(8, 8), batch=1)
        uv_da[..., 0] = value
        uv_da.requires_grad_()
        out = dr.texture(tex, uv, uv_da)
        torch.testing.assert_close(out, tex.mean((1, 2), keepdim=True).expand_as(out))
        (grad,) = torch.autograd.grad(out.sum(), uv_da)
        assert (grad == 0).all()

    @pytest.mark.parametrize("value, max_mip_level", [(INF, None), (-INF, 0), (50.0, None)])
    def test_infinite_bias_is_clamped(self, value, max_mip_level):
        tex, uv, _ = scene(size=(8, 8), batch=1)
        bias = torch.full((1, 5, 6), value, requires_grad=True)
        out = dr.texture(tex, uv, mip_level_bias=bias)
        level = 0 if value < 0 else 3
        expected = dr.texture(tex, uv, mip_level_bias=torch.full((1, 5, 6), float(level)))
        torch.testing.assert_close(out, expected)
        (grad,) = torch.autograd.grad(out.sum(), bias)
        assert (grad == 0).all()

    @pytest.mark.parametrize("filter_mode", FILTERS)
    @pytest.mark.parametrize("shape", [(0, 4, 4, 2), (2, 0, 4, 2), (2, 4, 0, 2)])
    def test_no_pixels(self, filter_mode, shape):
        tex = torch.rand(2 if shape[0] else 0, 4, 4, 3, requires_grad=True)
        uv = torch.zeros(shape)
        uv_da = torch.zeros(*shape[:3], 4) if "mipmap" in filter_mode else None
        out = dr.texture(tex, uv, uv_da, filter_mode=filter_mode)
        assert out.shape == (*shape[:3], 3)
        out.sum().backward()
        assert (tex.grad == 0).all()


class TestValidation:
    @pytest.mark.parametrize("shape", [(1, 0, 4, 3), (1, 4, 0, 3), (1, 4, 4, 0)])
    def test_zero_sized_texture(self, shape):
        with pytest.raises(ValueError, match="at least one texel and one channel"):
            dr.texture(torch.zeros(shape), torch.zeros(1, 4, 4, 2))
        with pytest.raises(ValueError, match="at least one texel and one channel"):
            texture_construct_mip(torch.zeros(shape))

    @pytest.mark.parametrize(
        "level, error, match",
        [
            (torch.zeros(4, 4, 3), ValueError, "mip level 1 must have shape"),
            (torch.zeros(1, 4, 4, 2), ValueError, "mip level 1 has 2 channels but tex has 3"),
            (torch.zeros(3, 4, 4, 3), ValueError, "mip level 1 has batch size 3"),
            (torch.zeros(1, 4, 0, 3), ValueError, "mip level 1 must have at least one texel"),
            (torch.zeros(1, 4, 4, 3, dtype=torch.float64), TypeError, "mip level 1 must be"),
            (torch.zeros(1, 6, 4, 4, 3), ValueError, "mip level 1 has the shape of a cube map"),
            ([[0.0]], TypeError, "mip level 1 must be a torch.Tensor"),
        ],
    )
    def test_bad_mip_level(self, level, error, match):
        tex = torch.zeros(2, 8, 8, 3)
        with pytest.raises(error, match=match):
            dr.texture(tex, torch.zeros(2, 4, 4, 2), mip=[level])

    def test_later_levels_are_checked_too(self):
        tex = torch.zeros(1, 8, 8, 3)
        with pytest.raises(ValueError, match="mip level 2 has 1 channels"):
            dr.texture(
                tex, torch.zeros(1, 4, 4, 2), mip=[tex[:, ::2, ::2], torch.zeros(1, 2, 2, 1)]
            )

    def test_stack_of_another_kind(self):
        stack = texture_construct_mip(torch.zeros(1, 6, 4, 4, 3), cube_mode=True)
        with pytest.raises(ValueError, match="cube_mode=True"):
            dr.texture(torch.zeros(1, 4, 4, 3), torch.zeros(1, 4, 4, 2), mip=stack)

    @pytest.mark.parametrize("shape", [(1, 16, 8, 3), (1, 8, 16, 3), (2, 8, 8, 3)])
    def test_stack_of_another_texture(self, shape):
        stack = texture_construct_mip(torch.zeros(1, 8, 8, 3))
        uv = torch.zeros(2, 4, 4, 2)
        with pytest.raises(ValueError, match="constructed for a texture of shape"):
            dr.texture(torch.zeros(shape), uv, mip_level_bias=torch.zeros(2, 4, 4), mip=stack)
        with pytest.raises(ValueError, match="constructed for a texture of shape"):
            dr.texture(torch.zeros(shape), uv, mip=stack, filter_mode="linear")

    def test_construct_mip_arguments(self):
        with pytest.raises(ValueError, match="max_mip_level must be non-negative"):
            texture_construct_mip(torch.zeros(1, 4, 4, 3), max_mip_level=-1)
        with pytest.raises(TypeError, match=r"tex must be torch\.float32"):
            texture_construct_mip(torch.zeros(1, 4, 4, 3, dtype=torch.float64))
        with pytest.raises(ValueError, match="tex must have shape"):
            texture_construct_mip(torch.zeros(4, 4, 3))


class TestCustomMips:
    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    @pytest.mark.parametrize("max_mip_level", [None, 1, 0])
    def test_list_of_the_same_levels_reproduces_the_automatic_chain(
        self, filter_mode, max_mip_level
    ):
        tex, uv, uv_da = scene(size=(8, 16))
        levels = [tex]
        while max(levels[-1].shape[1:3]) > 1:
            height, width = levels[-1].shape[1:3]
            pairs = levels[-1].view(1, max(height // 2, 1), -1, max(width // 2, 1), 2, 3)
            levels.append(pairs.mean((2, 4)))
        kwargs = {"filter_mode": filter_mode, "max_mip_level": max_mip_level}
        auto = dr.texture(tex, uv, uv_da, **kwargs)
        custom = dr.texture(tex, uv, uv_da, mip=levels[1:], **kwargs)
        torch.testing.assert_close(custom, auto, rtol=1e-6, atol=1e-6)

    def test_list_may_stop_at_any_level(self):
        tex, uv, uv_da = scene()
        levels = reference._mip_chain(tex, None)
        short = dr.texture(tex, uv, uv_da * 8, mip=levels[1:3])
        assert torch.equal(short, dr.texture(tex, uv, uv_da * 8, max_mip_level=2))
        assert torch.equal(dr.texture(tex, uv, uv_da, mip=[]), dr.texture(tex, uv))

    def test_levels_keep_their_own_size(self):
        tex, uv, _ = scene(size=(8, 8))
        level = torch.randn(1, 3, 5, 3, generator=torch.Generator().manual_seed(5))
        bias = torch.full(uv.shape[:3], 1.0)
        out = dr.texture(tex, uv, mip_level_bias=bias, mip=[level])
        assert torch.equal(out, dr.texture(level, uv, filter_mode="linear"))

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    def test_level_shared_across_a_batch_of_textures(self, filter_mode):
        tex, uv, uv_da = scene(tex_batch=2)
        level = torch.randn(1, 4, 6, 3, generator=torch.Generator().manual_seed(6))
        shared = dr.texture(tex, uv, uv_da, mip=[level], filter_mode=filter_mode)
        repeated = dr.texture(
            tex, uv, uv_da, mip=[level.expand(2, -1, -1, -1)], filter_mode=filter_mode
        )
        assert torch.equal(shared, repeated)

    def test_gradients_stop_at_the_listed_tensors(self):
        tex, uv, _ = scene(size=(8, 8))
        tex.requires_grad_()
        levels = [lv.detach().requires_grad_() for lv in reference._mip_chain(tex, None)[1:]]
        # A bias of exactly one reads level one only.
        out = dr.texture(tex, uv, mip_level_bias=torch.full(uv.shape[:3], 1.0), mip=levels)
        out.sum().backward()
        assert (tex.grad == 0).all()
        assert levels[0].grad.abs().sum() > 0

    def test_list_gradients_are_those_of_the_automatic_chain(self):
        # The gradient the base texture gets from the automatic chain is what it gets directly
        # plus what each level of a custom list gets, carried through the box filters.
        tex, uv, _ = scene(size=(8, 12), dtype=torch.float64)
        tex.requires_grad_()
        chain = reference._mip_chain(tex, None)
        levels = [level.detach().requires_grad_() for level in chain[1:]]
        gen = torch.Generator().manual_seed(7)
        weights = torch.randn(2, 5, 6, 3, generator=gen, dtype=torch.float64)
        bias = torch.rand(2, 5, 6, generator=gen, dtype=torch.float64) * 4

        def lookup(mip):
            return reference.texture(
                tex, uv, None, "linear-mipmap-linear", "wrap", None, mip_level_bias=bias, mip=mip
            )

        (total,) = gradients(lookup(None), [tex], weights)
        direct, *level_grads = gradients(lookup(levels), [tex, *levels], weights)
        carried = torch.autograd.grad(chain[1:], tex, level_grads)[0]
        assert all(grad.abs().sum() > 0 for grad in level_grads)
        torch.testing.assert_close(direct + carried, total)

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    def test_gradcheck(self, filter_mode):
        tex, uv, _ = scene(size=(4, 6), batch=1, channels=2, dtype=torch.float64)
        gen = torch.Generator().manual_seed(8)
        levels = [
            torch.randn(1, *size, 2, generator=gen, dtype=torch.float64, requires_grad=True)
            for size in ((2, 3), (3, 1))
        ]
        # Footprints of 1.2 to 3.2 texels keep the level of detail away from integers and clamps.
        scale = (torch.rand(1, 5, 6, 1, generator=gen, dtype=torch.float64) * 2 + 1.2) / 6
        uv_da = (scale * torch.tensor([1.0, 0.1, -0.2, 0.9], dtype=torch.float64)).requires_grad_()
        tex.requires_grad_()
        uv.requires_grad_()
        assert torch.autograd.gradcheck(
            lambda t, a, b, c, d: reference.texture(t, c, d, filter_mode, "wrap", None, mip=[a, b]),
            (tex, *levels, uv, uv_da),
        )

    def test_trainable_mip_list_as_in_pipelines(self):
        # Pipelines keep a texture as a list of trainable levels and pass all but the first.
        _, uv, uv_da = scene()
        gen = torch.Generator().manual_seed(9)
        levels = [
            torch.rand(1, 16 >> i, 16 >> i, 3, generator=gen, requires_grad=True) for i in range(5)
        ]
        optimizer = torch.optim.SGD(levels, lr=0.1)
        before = [level.detach().clone() for level in levels]
        out = dr.texture(levels[0], uv, uv_da, mip=levels[1:], filter_mode="linear-mipmap-linear")
        out.square().mean().backward()
        optimizer.step()
        assert out.shape == (2, 5, 6, 3)
        assert sum(not torch.equal(a, b) for a, b in zip(levels, before, strict=True)) >= 3


class TestConstructMip:
    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    @pytest.mark.parametrize("size", [(8, 12), (5, 7)])
    def test_same_result_as_building_internally(self, filter_mode, size):
        tex, uv, uv_da = scene(size=size, tex_batch=2)
        mip = texture_construct_mip(tex)
        out = dr.texture(tex, uv, uv_da, mip=mip, filter_mode=filter_mode)
        assert torch.equal(out, dr.texture(tex, uv, uv_da, filter_mode=filter_mode))

    def test_holds_levels_one_and_up(self):
        tex = torch.rand(1, 8, 12, 3)
        chain = reference._mip_chain(tex, None)
        mip = texture_construct_mip(tex)
        assert isinstance(mip, MipStack)
        assert len(mip.levels) == len(chain) - 1 == 4
        assert all(torch.equal(a, b) for a, b in zip(mip.levels, chain[1:], strict=True))
        assert len(texture_construct_mip(tex, max_mip_level=2).levels) == 2
        assert len(texture_construct_mip(tex, max_mip_level=0).levels) == 0

    def test_max_mip_level_limits_the_levels_used(self):
        tex, uv, uv_da = scene()
        mip = texture_construct_mip(tex)
        limited = dr.texture(tex, uv, uv_da * 8, mip=mip, max_mip_level=1)
        assert torch.equal(limited, dr.texture(tex, uv, uv_da * 8, max_mip_level=1))
        short = texture_construct_mip(tex, max_mip_level=1)
        assert torch.equal(dr.texture(tex, uv, uv_da * 8, mip=short), limited)

    def test_gradients_flow_through_the_stack_to_the_texture(self):
        tex, uv, uv_da = scene()
        tex.requires_grad_()
        (expected,) = gradients(dr.texture(tex, uv, uv_da * 4), [tex])
        mip = texture_construct_mip(tex)
        (grad,) = gradients(dr.texture(tex, uv, uv_da * 4, mip=mip), [tex])
        torch.testing.assert_close(grad, expected)

    def test_reused_across_calls(self):
        tex, uv, uv_da = scene()
        mip = texture_construct_mip(tex)
        for scale in (1.0, 3.0, 9.0):
            out = dr.texture(tex, uv, uv_da * scale, mip=mip)
            assert torch.equal(out, dr.texture(tex, uv, uv_da * scale))

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    def test_packed_levels_are_kept_between_calls(self, filter_mode):
        tex, uv, uv_da = scene(tex_batch=2)
        mip = texture_construct_mip(tex)
        packed = mip.pack(tex)
        assert all(a is b for a, b in zip(mip.pack(tex), packed, strict=True))
        for max_mip_level in (None, 2, 1, 0, 7):
            kwargs = {"filter_mode": filter_mode, "max_mip_level": max_mip_level}
            out = dr.texture(tex, uv, uv_da * 4, mip=mip, **kwargs)
            assert torch.equal(out, dr.texture(tex, uv, uv_da * 4, **kwargs))
        assert mip.pack(tex)[0] is packed[0]

    def test_kept_levels_follow_the_texture_passed(self):
        # The stack was built for one texture. Whatever is passed as tex is still the base level
        # that is sampled, also after the levels have been packed for another one.
        tex, uv, uv_da = scene()
        mip = texture_construct_mip(tex)
        levels = list(mip.levels)
        dr.texture(tex, uv, uv_da, mip=mip)

        other = torch.rand_like(tex)
        assert torch.equal(
            dr.texture(other, uv, uv_da, mip=mip), dr.texture(other, uv, uv_da, mip=levels)
        )
        dr.texture(tex, uv, uv_da, mip=mip)
        tex.mul_(2.0)
        assert torch.equal(
            dr.texture(tex, uv, uv_da, mip=mip), dr.texture(tex, uv, uv_da, mip=levels)
        )

    def test_kept_levels_do_not_hide_a_texture_that_starts_to_require_gradients(self):
        tex, uv, uv_da = scene()
        mip = texture_construct_mip(tex)
        dr.texture(tex, uv, uv_da, mip=mip)
        tex.requires_grad_()
        (grad,) = gradients(dr.texture(tex, uv, uv_da, mip=mip), [tex])
        (expected,) = gradients(dr.texture(tex, uv, uv_da, mip=list(mip.levels)), [tex])
        assert expected.abs().sum() > 0
        torch.testing.assert_close(grad, expected)

    def test_inference_mode(self):
        tex, uv, uv_da = scene()
        mip = texture_construct_mip(tex)
        expected = dr.texture(tex, uv, uv_da * 4)
        with torch.inference_mode():
            assert torch.equal(dr.texture(tex, uv, uv_da * 4, mip=mip), expected)
        uv.requires_grad_()
        out = dr.texture(tex, uv, uv_da * 4, mip=mip)
        assert torch.equal(out, expected)
        (grad,) = gradients(out, [uv])
        assert grad.abs().sum() > 0


@pytest.fixture
def triton_backend():
    pytest.importorskip("triton")
    from diffrast.backends import triton

    return triton


@pytest.mark.gpu
class TestTriton:
    """The Triton backend against the reference backend on the same device."""

    def compare(self, triton_backend, inputs, call):
        """Run ``call(backend, *inputs)`` on both backends and compare outputs and gradients.

        Returns the reference output followed by its gradient for each input.
        """
        results = []
        for backend in (reference, triton_backend):
            leaves = [t.detach().clone().requires_grad_() for t in inputs]
            out = call(backend, *leaves)
            weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(10)).cuda()
            results.append([out.detach(), *gradients(out, leaves, weights)])
        ref, ours = results
        # The kernels round like the PyTorch code, so the same pixels are NaN and the rest agree
        # to the last bits.
        torch.testing.assert_close(ours[0], ref[0], atol=1e-6, rtol=1e-6, equal_nan=True)
        for grad, ref_grad in zip(ours[1:], ref[1:], strict=True):
            assert torch.isfinite(ref_grad).all()
            torch.testing.assert_close(grad, ref_grad, atol=1e-5, rtol=1e-4)
        return ref

    @pytest.mark.parametrize("max_mip_level, bias", [(None, 3.0), (0, 0.0), (1, 1.0), (None, 7.5)])
    def test_bias_at_or_past_the_top_level(self, triton_backend, max_mip_level, bias):
        tex, uv, _ = (t.cuda() for t in scene(size=(8, 8)))
        bias = torch.full(uv.shape[:3], bias).cuda()
        ref = self.compare(
            triton_backend,
            (tex, uv, bias),
            lambda backend, t, c, b: backend.texture(
                t, c, None, "linear-mipmap-linear", "wrap", max_mip_level, mip_level_bias=b
            ),
        )
        assert (ref[3] == 0).all()

    @pytest.mark.parametrize(
        "max_mip_level, texels", [(None, 8.0), (0, 1.0), (2, 4.0), (None, 64.0)]
    )
    def test_footprint_at_or_past_the_top_level(self, triton_backend, max_mip_level, texels):
        tex, uv, _ = (t.cuda() for t in scene(size=(8, 8)))
        uv_da = footprint(uv.shape[:3], texels, 8).cuda()
        ref = self.compare(
            triton_backend,
            (tex, uv, uv_da),
            lambda backend, t, c, d: backend.texture(
                t, c, d, "linear-mipmap-linear", "wrap", max_mip_level
            ),
        )
        assert (ref[3] == 0).all()

    @pytest.mark.parametrize("size", [(5, 1), (1, 7), (6, 10), (37, 50)])
    def test_isotropic_footprint_on_a_non_square_texture(self, triton_backend, size):
        # Both footprint axes are equally long here, and the two backends have to split the
        # gradient between them in the same way.
        tex, uv, _ = (t.cuda() for t in scene(size=size))
        uv_da = torch.zeros(*uv.shape[:3], 4)
        uv_da[..., 0], uv_da[..., 3] = 2.0 / size[1], 2.0 / size[0]
        ref = self.compare(
            triton_backend,
            (tex, uv, uv_da.cuda()),
            lambda backend, t, c, d: backend.texture(t, c, d, "linear-mipmap-linear", "wrap", None),
        )
        assert ref[3].abs().sum() > 0

    @pytest.mark.parametrize("size", [(1, 46342), (1, 65536), (65536, 1)])
    def test_level_of_detail_of_very_long_textures(self, triton_backend, size):
        # Stripes of zeros and ones average to one half from the first mip level on.
        length = max(size)
        tex = (torch.arange(length) % 2).float().view(1, *size, 1).cuda()
        centre = (torch.arange(16.0) * 1000 + 0.5) / length
        along_u = size[1] > 1
        uv = torch.full((1, 1, 16, 2), 0.5)
        uv[..., 0 if along_u else 1] = centre
        uv_da = torch.zeros(1, 1, 16, 4)
        uv_da[..., 0 if along_u else 3] = 8.0 / length
        out = triton_backend.texture(
            tex, uv.cuda(), uv_da.cuda(), "linear-mipmap-linear", "wrap", None
        )
        torch.testing.assert_close(out, torch.full_like(out, 0.5))

    @pytest.mark.parametrize("filter_mode", FILTERS)
    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_far_out_and_undefined_coordinates(self, triton_backend, filter_mode, boundary):
        values = [0.3, -0.2, 1.4, 3e8, -2000000.25, 1048576.25, 1e30, -1e30, 3e38, INF, -INF, NAN]
        pairs = torch.tensor(list(itertools.product(values, values))).view(1, 12, 12, 2)
        tex, _, _ = scene(size=(6, 9), tex_batch=2)
        uv = pairs.expand(2, -1, -1, -1).contiguous()
        gen = torch.Generator().manual_seed(11)
        uv_da = torch.rand(2, 12, 12, 4, generator=gen) * 0.4 - 0.2
        mip = "mipmap" in filter_mode
        ref = self.compare(
            triton_backend,
            (tex.cuda(), uv.cuda(), uv_da.cuda()),
            lambda backend, t, c, d: backend.texture(
                t, c, d if mip else None, filter_mode, boundary, None
            ),
        )
        finite = torch.isfinite(pairs).all(-1)[0]
        assert not ref[0][0][finite].isnan().any()

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_extreme_levels_of_detail(self, triton_backend, filter_mode, with_bias):
        values = [0.05, 0.0, 3e38, -3e38, INF, -INF, NAN, 1e-30, 1e10]
        pairs = torch.tensor(list(itertools.product(values, values))).view(1, 9, 9, 2)
        uv_da = torch.cat([pairs[..., :1], torch.zeros(1, 9, 9, 2), pairs[..., 1:]], -1)
        bias = torch.tensor([0.0, 1.5, -3.0, INF, -INF, NAN, 50.0, -55.0, 2.0])
        bias = bias.view(1, 9, 1).expand(1, 9, 9).contiguous()
        tex = scene(size=(6, 9))[0]
        uv = torch.rand(1, 9, 9, 2, generator=torch.Generator().manual_seed(12))
        self.compare(
            triton_backend,
            (tex.cuda(), uv.cuda(), uv_da.cuda(), bias.cuda()),
            lambda backend, t, c, d, b: backend.texture(
                t, c, d, filter_mode, "wrap", None, mip_level_bias=b if with_bias else None
            ),
        )

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    @pytest.mark.parametrize("level_batch", [1, 2])
    @pytest.mark.parametrize("max_mip_level", [None, 1])
    def test_custom_mip_list(self, triton_backend, filter_mode, level_batch, max_mip_level):
        tex, uv, _ = (t.cuda() for t in scene(size=(8, 12), tex_batch=2))
        gen = torch.Generator().manual_seed(13)
        levels = [
            torch.randn(level_batch, *size, 3, generator=gen).cuda()
            for size in ((4, 6), (3, 2), (1, 1))
        ]
        bias = (torch.rand(uv.shape[:3], generator=gen) * 3.5 - 0.25).cuda()
        ref = self.compare(
            triton_backend,
            (tex, uv, bias, *levels),
            lambda backend, t, c, b, *lv: backend.texture(
                t, c, None, filter_mode, "clamp", max_mip_level, mip_level_bias=b, mip=lv
            ),
        )
        used = 1 if max_mip_level == 1 else 3
        assert all(grad.abs().sum() > 0 for grad in ref[4 : 4 + used])

    @pytest.mark.parametrize("filter_mode, lone", [("nearest", 1), ("linear-mipmap-nearest", 2)])
    def test_output_requires_grad_as_on_the_reference_backend(
        self, triton_backend, filter_mode, lone
    ):
        # Inputs a filter mode does not depend on smoothly do not make the output differentiable.
        inputs = [t.cuda() for t in scene()]
        inputs[lone].requires_grad_()
        for backend in (reference, triton_backend):
            out = backend.texture(*inputs, filter_mode, "wrap", None)
            assert not out.requires_grad

    @pytest.mark.parametrize("filter_mode", ["linear", "linear-mipmap-linear"])
    @pytest.mark.parametrize(
        "loss", [lambda out: out.sum(), lambda out: out.mean(), lambda out: out.square().sum()]
    )
    def test_double_backward_raises(self, triton_backend, filter_mode, loss):
        # With a loss that is linear in the lookup, the incoming gradient is a constant and the
        # second-order terms come from the texture and the coordinates alone.
        tex, uv, uv_da = (t.cuda().requires_grad_() for t in scene())
        mip = "mipmap" in filter_mode
        out = triton_backend.texture(tex, uv, uv_da if mip else None, filter_mode, "wrap", None)
        (grad,) = torch.autograd.grad(loss(out), uv, create_graph=True)
        with pytest.raises(RuntimeError, match="first-order gradients only"):
            grad.square().sum().backward()

    def test_double_backward_of_a_nearest_lookup_raises(self, triton_backend):
        tex, uv, _ = (t.cuda() for t in scene())
        tex.requires_grad_()
        out = triton_backend.texture(tex, uv, None, "nearest", "wrap", None)
        (grad,) = torch.autograd.grad(out.square().sum(), tex, create_graph=True)
        with pytest.raises(RuntimeError, match="first-order gradients only"):
            grad.square().sum().backward()

    def test_first_order_gradients_with_create_graph(self, triton_backend):
        inputs = [t.cuda().requires_grad_() for t in scene()]
        out = triton_backend.texture(*inputs, "linear-mipmap-linear", "wrap", None)
        expected = torch.autograd.grad(out.square().sum(), inputs, retain_graph=True)
        grads = torch.autograd.grad(out.square().sum(), inputs, create_graph=True)
        for grad, expected_grad in zip(grads, expected, strict=True):
            assert expected_grad.abs().sum() > 0
            torch.testing.assert_close(grad, expected_grad, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("filter_mode", FILTERS)
    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_texture_gradient_alone(self, triton_backend, filter_mode, boundary):
        # Without gradients for the coordinates the backward pass does not read the texture.
        tex, uv, uv_da = (t.cuda() for t in scene(size=(6, 9), tex_batch=2))
        bias = torch.rand(uv.shape[:3], generator=torch.Generator().manual_seed(14)).cuda() * 2
        mip = "mipmap" in filter_mode
        ref = self.compare(
            triton_backend,
            (tex,),
            lambda backend, t: backend.texture(
                t, uv, uv_da if mip else None, filter_mode, boundary, None, bias if mip else None
            ),
        )
        assert ref[1].abs().sum() > 0

    @pytest.mark.parametrize("filter_mode", ["nearest", "linear"])
    def test_texture_modified_in_place_before_backward(self, triton_backend, filter_mode):
        results = []
        for backend in (reference, triton_backend):
            base, uv, _ = (t.cuda() for t in scene())
            base.requires_grad_()
            tex = base * 1.0
            out = backend.texture(tex, uv, None, filter_mode, "wrap", None)
            with torch.no_grad():
                tex.mul_(2.0)
            results.append(gradients(out, [base])[0])
        assert results[0].abs().sum() > 0
        torch.testing.assert_close(results[1], results[0], atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("filter_mode", MIP_FILTERS)
    def test_constructed_stack_is_packed_once(self, triton_backend, filter_mode, monkeypatch):
        monkeypatch.setattr(backends, "_selected", "triton")
        tex, uv, uv_da = (t.cuda() for t in scene(size=(5, 7), tex_batch=2))
        mip = texture_construct_mip(tex)
        uv.requires_grad_()
        for max_mip_level in (None, 1, 0):
            kwargs = {"filter_mode": filter_mode, "max_mip_level": max_mip_level}
            expected = dr.texture(tex, uv, uv_da * 4, **kwargs)
            with torch.inference_mode():
                assert torch.equal(dr.texture(tex, uv, uv_da * 4, mip=mip, **kwargs), expected)
            out = dr.texture(tex, uv, uv_da * 4, mip=mip, **kwargs)
            assert torch.equal(out, expected)
            assert torch.equal(gradients(out, [uv])[0], gradients(expected, [uv])[0])
        assert mip.pack(tex)[0] is mip.pack(tex)[0]

    @pytest.mark.parametrize("filter_mode", FILTERS)
    @pytest.mark.parametrize("shape", [(0, 4, 4, 2), (2, 0, 4, 2)])
    def test_no_pixels(self, triton_backend, filter_mode, shape):
        tex = torch.rand(2 if shape[0] else 0, 4, 4, 3).cuda().requires_grad_()
        uv = torch.zeros(shape).cuda()
        uv_da = torch.zeros(*shape[:3], 4).cuda() if "mipmap" in filter_mode else None
        out = triton_backend.texture(tex, uv, uv_da, filter_mode, "wrap", None)
        assert out.shape == (*shape[:3], 3)
        out.sum().backward()
        assert (tex.grad == 0).all()

    def test_public_api_on_the_gpu(self, triton_backend, monkeypatch):
        monkeypatch.setattr(backends, "_selected", "triton")
        tex, uv, uv_da = (t.cuda() for t in scene(size=(5, 7), tex_batch=2))
        tex.requires_grad_()
        auto = dr.texture(tex, uv, uv_da * 4)
        (expected,) = gradients(auto, [tex])
        mip = texture_construct_mip(tex)
        out = dr.texture(tex, uv, uv_da * 4, mip=mip)
        assert torch.equal(out, auto)
        torch.testing.assert_close(gradients(out, [tex])[0], expected, atol=1e-5, rtol=1e-4)
        levels = [level.detach() for level in mip.levels]
        assert torch.equal(dr.texture(tex, uv, uv_da * 4, mip=levels), auto)

        with pytest.raises(ValueError, match="at least one texel and one channel"):
            dr.texture(torch.zeros(1, 0, 4, 3).cuda(), uv)
        with pytest.raises(ValueError, match="mip level 1 is on cpu but tex is on cuda"):
            dr.texture(tex, uv, uv_da, mip=[levels[0].cpu()])
