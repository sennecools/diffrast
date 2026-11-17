"""Public differentiable rasterization ops.

Each op validates its inputs and dispatches to the active backend (see
:func:`diffrast.set_backend`). All ops accept a leading batch dimension so several views can be
rendered in one call.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Literal, SupportsFloat, get_args

import torch
from torch import Tensor

from diffrast._cache import TensorCache
from diffrast.backends import get_backend
from diffrast.context import RasterizeContext
from diffrast.topology import TopologyHash, neighbours_for

FilterMode = Literal["auto", "nearest", "linear", "linear-mipmap-nearest", "linear-mipmap-linear"]
BoundaryMode = Literal["wrap", "clamp", "zero", "cube"]

FILTER_MODES = get_args(FilterMode)
BOUNDARY_MODES = get_args(BoundaryMode)


def rasterize(
    glctx: RasterizeContext,
    pos: Tensor,
    tri: Tensor,
    resolution: tuple[int, int],
    ranges: Tensor | None = None,
    grad_db: bool = True,
) -> tuple[Tensor, Tensor]:
    """Rasterize triangles into per-pixel barycentrics, depth and triangle ids.

    The rank of ``pos`` selects the mode. In instanced mode every batch item draws all triangles
    with its own vertex positions. In range mode all items share one vertex buffer, and item ``i``
    draws the triangles ``tri[start_i : start_i + count_i]`` given by ``ranges``.

    A pixel is covered by a triangle when its centre lies inside it. A centre exactly on an edge
    belongs to the triangle whose interior lies in the +x direction from that edge (+y for a
    horizontal edge), so triangles sharing an edge never both cover a pixel. Triangles with zero
    area or with a NaN or infinite vertex are not drawn.

    Args:
        glctx: Rasterization context.
        pos: Clip-space vertex positions, float32 of shape ``[batch, vertices, 4]`` (instanced
            mode) or ``[vertices, 4]`` (range mode).
        tri: Vertex indices, int32 of shape ``[triangles, 3]``, at most ``2**24`` triangles.
        resolution: Output size as ``(height, width)``.
        ranges: In range mode, ``(start, count)`` per batch item, int32 of shape ``[batch, 2]``,
            normally kept on the CPU. Ignored in instanced mode.
        grad_db: Propagate gradients from ``rast_db`` back to ``pos``.

    Returns:
        ``(rast, rast_db)``, both of shape ``[batch, height, width, 4]``. Each pixel of ``rast``
        holds ``(u, v, z / w, triangle_id + 1)``, and is all zeros where no triangle covers it;
        the id is the triangle's row in ``tri`` in both modes. ``rast_db`` holds the screen-space
        derivatives ``(du/dx, du/dy, dv/dx, dv/dy)``. For a context whose ``output_db`` is false,
        ``rast_db`` has shape ``[batch, height, width, 0]`` and ``grad_db`` is ignored.
    """
    resolution = check_rasterize_inputs(glctx, pos, tri, resolution, ranges)
    backend = get_backend(pos.device, "rasterize")
    output_db = getattr(glctx, "output_db", True)
    grad_db = grad_db and output_db
    if pos.ndim == 2:
        layers = [
            backend.rasterize(pos[None], triangles, resolution, grad_db)
            for triangles in split_ranges(tri, ranges)
        ]
        rast, rast_db = join_ranges(pos, resolution, ranges, layers)
    else:
        rast, rast_db = backend.rasterize(pos, tri, resolution, grad_db)
    return rast, (rast_db if output_db else rast.new_zeros(*rast.shape[:3], 0))


def interpolate(
    attr: Tensor,
    rast: Tensor,
    tri: Tensor,
    rast_db: Tensor | None = None,
    diff_attrs: Sequence[int] | Literal["all"] | None = None,
) -> tuple[Tensor, Tensor]:
    """Interpolate vertex attributes across the rasterized triangles.

    Args:
        attr: Vertex attributes, float32 of shape ``[batch, vertices, channels]``, or
            ``[vertices, channels]`` to share them across the batch.
        rast: First output of :func:`rasterize`, float32 of shape ``[batch, height, width, 4]``.
            Pixels whose triangle id lies past the end of ``tri`` are treated as background.
        tri: Vertex indices into ``attr``, int32 of shape ``[triangles, 3]``, with one row per
            triangle of the index buffer that produced ``rast``. The indices are checked against
            the number of vertices once per tensor object, so pass the same tensor to every call
            rather than a fresh copy or cast. Only edits PyTorch records are noticed: after
            writing through ``tri.data`` or through memory shared with NumPy, pass a new tensor.
        rast_db: Second output of :func:`rasterize`. Required when ``diff_attrs`` is set; an
            empty tensor of shape ``[batch, height, width, 0]`` counts as ``None``.
        diff_attrs: Attribute channels to compute screen-space derivatives for, or ``"all"``.

    Returns:
        ``(out, out_da)``. ``out`` has shape ``[batch, height, width, channels]``. ``out_da`` has
        shape ``[batch, height, width, 2 * len(diff_attrs)]`` with ``(d/dx, d/dy)`` per selected
        channel, and is empty, of shape ``[batch, height, width, 0]``, when ``diff_attrs`` is not
        set.
    """
    shared = isinstance(attr, Tensor) and attr.ndim == 2
    layout = "[batch, vertices, channels] or [vertices, channels]"
    _check(attr, "attr", ndim=2 if shared else 3, dtype=torch.float32, layout=layout)
    _check(rast, "rast", ndim=4, last=4, dtype=torch.float32)
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    if attr.ndim == 3 and attr.shape[0] not in (1, rast.shape[0]):
        raise ValueError(
            f"attr batch size {attr.shape[0]} does not match rast batch size {rast.shape[0]}."
        )
    _check_devices(attr=attr, rast=rast, tri=tri, rast_db=rast_db)

    channels = _diff_channels(diff_attrs, attr.shape[-1])
    empty = rast_db is not None and rast_db.ndim == 4 and rast_db.shape[-1] == 0
    if rast_db is not None and not empty:
        _check(rast_db, "rast_db", ndim=4, last=4, dtype=torch.float32)
        if rast_db.shape != rast.shape:
            raise ValueError(
                f"rast_db must have the same shape as rast, got {tuple(rast_db.shape)} and "
                f"{tuple(rast.shape)}."
            )
    if not channels:
        rast_db = None
    elif rast_db is None or empty:
        given = ", but the one given is empty" if empty else ""
        raise ValueError(f"rast_db is required when diff_attrs is set{given}.")

    _check_indices(tri, attr.shape[-2])
    return get_backend(rast.device, "interpolate").interpolate(attr, rast, tri, rast_db, channels)


def texture(
    tex: Tensor,
    uv: Tensor,
    uv_da: Tensor | None = None,
    mip_level_bias: Tensor | None = None,
    mip: MipStack | Sequence[Tensor] | None = None,
    filter_mode: FilterMode = "auto",
    boundary_mode: BoundaryMode = "wrap",
    max_mip_level: int | None = None,
) -> Tensor:
    """Sample a 2D texture or a cube map at per-pixel texture coordinates.

    Args:
        tex: Texture, float32 of shape ``[batch, tex_height, tex_width, channels]``, or a cube
            map of shape ``[batch, 6, size, size, channels]`` with its faces in the order
            ``+x, -x, +y, -y, +z, -z`` and oriented as in OpenGL. A batch of one is shared across
            all views.
        uv: Texture coordinates, float32 of shape ``[batch, height, width, 2]``. ``u`` runs along
            the texture width and ``v`` along its height; ``(0, 0)`` is the outer corner of texel
            ``[0, 0]`` and ``(1, 1)`` the outer corner of the last texel. For a cube map, direction
            vectors of shape ``[batch, height, width, 3]``, which need not be normalised.
        uv_da: Screen-space derivatives ``(du/dx, du/dy, dv/dx, dv/dy)`` of ``uv``, shape
            ``[batch, height, width, 4]``, as returned by :func:`interpolate`. For a cube map,
            ``(d/dx, d/dy)`` of each of the three direction components, shape
            ``[batch, height, width, 6]``. The mipmapped filter modes need it or
            ``mip_level_bias``.
        mip_level_bias: Per-pixel offset added to the mip level of detail, shape
            ``[batch, height, width]``. Without ``uv_da`` it sets the level directly.
        mip: Mip levels to use instead of box filtering ``tex`` on every call. Either the result
            of :func:`texture_construct_mip` for the same ``tex`` (a ``tex`` of another shape is
            an error), or a list of tensors holding levels 1, 2 and so on. Each has the format of
            ``tex`` with its own height and width, and a batch of one or that of ``tex``. The base
            level stays in ``tex`` and is not part of the list, which may stop at any level.
            Gradients reach the listed tensors and do not continue from them to ``tex``.
        filter_mode: ``"auto"`` picks ``"linear-mipmap-linear"`` when ``uv_da`` or
            ``mip_level_bias`` is given and ``"linear"`` otherwise.
        boundary_mode: Coordinates outside ``[0, 1]`` repeat the texture (``"wrap"``), repeat its
            edge texels (``"clamp"``), or read zeros (``"zero"``). A cube map requires ``"cube"``:
            the lookup in a mip level is filtered across face edges and corners without seams,
            and directions that select no point on the cube (a component that is not finite,
            or a vector shorter than the smallest normal float, such as the zero vector) read
            zeros and receive no gradient. The level of detail is the footprint of ``uv_da`` on
            the face the direction selects, so the level chosen from ``uv_da`` can still change
            from one face to the next.
        max_mip_level: Index of the coarsest mip level to use, or ``None`` for a full chain.
            ``0`` disables mipmapping. It also limits the levels used from ``mip``.

    Returns:
        Sampled values of shape ``[batch, height, width, channels]``. Infinite ``uv`` counts as
        far outside the texture, where wrapping has no texel to return. A pixel that is undefined
        in that way, or whose ``uv``, ``uv_da`` or ``mip_level_bias`` is NaN, is NaN and passes
        no gradient.
    """
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"Unknown filter_mode {filter_mode!r}; expected one of {FILTER_MODES}.")
    if boundary_mode not in BOUNDARY_MODES:
        raise ValueError(
            f"Unknown boundary_mode {boundary_mode!r}; expected one of {BOUNDARY_MODES}."
        )
    cube = boundary_mode == "cube"
    _check_texture(tex, "tex", cube, 'boundary_mode="cube"')
    _check(uv, "uv", ndim=4, last=3 if cube else 2, dtype=torch.float32)
    if tex.shape[0] not in (1, uv.shape[0]):
        raise ValueError(
            f"tex batch size {tex.shape[0]} does not match uv batch size {uv.shape[0]}."
        )
    if isinstance(uv_da, Tensor) and uv_da.ndim == 4 and uv_da.shape[-1] == 0:
        uv_da = None
    has_lod = uv_da is not None or mip_level_bias is not None
    if filter_mode == "auto":
        filter_mode = "linear-mipmap-linear" if has_lod else "linear"
    if "mipmap" in filter_mode and not has_lod:
        raise ValueError(f"filter_mode {filter_mode!r} requires uv_da or mip_level_bias.")
    if uv_da is not None:
        _check(uv_da, "uv_da", ndim=4, last=6 if cube else 4, dtype=torch.float32)
        if uv_da.shape[:3] != uv.shape[:3]:
            raise ValueError("uv_da must match uv in batch, height and width.")
    if mip_level_bias is not None:
        _check(mip_level_bias, "mip_level_bias", ndim=3, dtype=torch.float32)
        if mip_level_bias.shape != uv.shape[:3]:
            raise ValueError("mip_level_bias must match uv in batch, height and width.")
    if max_mip_level is not None and max_mip_level < 0:
        raise ValueError(f"max_mip_level must be non-negative, got {max_mip_level}.")
    levels = None if mip is None else _check_mip(mip, tex, cube)
    stacked = isinstance(mip, MipStack) and "mipmap" in filter_mode

    _check_devices(tex=tex, uv=uv, uv_da=uv_da, mip_level_bias=mip_level_bias)
    return get_backend(uv.device, "texture").texture(
        tex,
        uv,
        uv_da,
        filter_mode,
        boundary_mode,
        max_mip_level,
        mip_level_bias=mip_level_bias,
        mip=levels,
        packed=mip.pack(tex) if stacked else None,
    )


def texture_construct_mip(
    tex: Tensor, max_mip_level: int | None = None, cube_mode: bool = False
) -> MipStack:
    """Build the mip levels of a texture once, for use in several :func:`texture` calls.

    Passing the result as ``mip``, together with the same ``tex``, saves rebuilding the levels on
    every call for a texture that does not change. If ``tex`` requires gradients they flow
    through the levels back to it, so the stack has to be built again whenever ``tex`` changes.
    For a ``tex`` that does not require gradients, the stack also keeps the buffer the backends
    sample, a copy of ``tex`` and the levels, from one call to the next.

    Args:
        tex: Texture, float32 of shape ``[batch, tex_height, tex_width, channels]``, or a cube
            map of shape ``[batch, 6, size, size, channels]``.
        max_mip_level: Index of the coarsest mip level to build, or ``None`` for a full chain.
        cube_mode: Must be true for a cube map, whose faces are filtered one by one.

    Returns:
        An opaque object holding mip levels 1 and up.
    """
    from diffrast.backends import reference

    _check_texture(tex, "tex", cube_mode, "cube_mode=True")
    if max_mip_level is not None and max_mip_level < 0:
        raise ValueError(f"max_mip_level must be non-negative, got {max_mip_level}.")
    levels = reference.texture_construct_mip(tex, max_mip_level)
    return MipStack(levels, bool(cube_mode), tex.shape)


class MipStack:
    """Mip levels built by :func:`texture_construct_mip`, passed to :func:`texture` as ``mip``.

    The contents are an implementation detail.
    """

    __slots__ = ("_packed", "cube_mode", "levels", "shape")

    def __init__(self, levels: Sequence[Tensor], cube_mode: bool, shape: Sequence[int]) -> None:
        self.levels = tuple(levels)
        self.cube_mode = cube_mode
        self.shape = tuple(shape)
        self._packed: tuple[Tensor, int, tuple[Tensor, Tensor, Tensor, Tensor]] | None = None

    def pack(self, tex: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``tex`` and the levels joined into the one buffer that the backends sample.

        The result is kept for later calls with the same tensor, as long as nothing in it takes
        part in autograd. A ``tex`` modified in place since is noticed by its version counter.
        """
        from diffrast.backends import reference

        constant = not (
            tex.requires_grad
            or tex.is_inference()
            or torch.is_inference_mode_enabled()
            or any(level.requires_grad for level in self.levels)
        )
        if constant and self._packed is not None:
            base, version, packed = self._packed
            if base is tex and version == tex._version:
                return packed
        packed = reference._pack([tex, *self.levels])
        if constant:
            self._packed = (tex, tex._version, packed)
        return packed


def _check_texture(t: Tensor, name: str, cube: bool, cube_option: str) -> None:
    """Validate the format of a texture or mip level; ``cube_option`` is what selects cube maps."""
    if isinstance(t, Tensor) and t.ndim == 5 and not cube:
        raise ValueError(
            f"{name} has the shape of a cube map, {tuple(t.shape)}, which needs {cube_option}."
        )
    if isinstance(t, Tensor) and t.ndim == 4 and cube:
        raise ValueError(
            f"{cube_option} needs {name} to be a cube map of shape "
            f"[batch, 6, size, size, channels], got {tuple(t.shape)}."
        )
    _check(t, name, ndim=5 if cube else 4, dtype=torch.float32)
    if cube and (t.shape[1] != 6 or t.shape[2] != t.shape[3]):
        raise ValueError(
            f"{name} must have six square faces, [batch, 6, size, size, channels], "
            f"got {tuple(t.shape)}."
        )
    if 0 in t.shape[1:]:
        raise ValueError(
            f"{name} must have at least one texel and one channel, got shape {tuple(t.shape)}."
        )


def _check_mip(mip: MipStack | Sequence[Tensor], tex: Tensor, cube: bool) -> list[Tensor]:
    """Validate ``mip`` against ``tex`` and return the tensors of mip levels 1 and up."""
    if isinstance(mip, MipStack):
        if mip.cube_mode != cube:
            raise ValueError(
                f"mip was constructed with cube_mode={mip.cube_mode}, which does not match "
                f"{'a cube map' if cube else 'a 2D texture'}."
            )
        if mip.shape != tuple(tex.shape):
            raise ValueError(
                f"mip was constructed for a texture of shape {mip.shape}, but tex has shape "
                f"{tuple(tex.shape)}."
            )
        levels = list(mip.levels)
    elif isinstance(mip, Sequence):
        levels = list(mip)
    else:
        raise TypeError(
            "mip must be a list of tensors or the result of texture_construct_mip, "
            f"got {type(mip).__name__}."
        )
    for index, level in enumerate(levels, start=1):
        name = f"mip level {index}"
        _check_texture(level, name, cube, 'boundary_mode="cube"')
        if level.shape[-1] != tex.shape[-1]:
            raise ValueError(f"{name} has {level.shape[-1]} channels but tex has {tex.shape[-1]}.")
        if level.shape[0] not in (1, tex.shape[0]):
            raise ValueError(
                f"{name} has batch size {level.shape[0]}; expected 1 or that of tex, "
                f"{tex.shape[0]}."
            )
        if level.device != tex.device:
            raise ValueError(f"{name} is on {level.device} but tex is on {tex.device}.")
    return levels


def antialias(
    color: Tensor,
    rast: Tensor,
    pos: Tensor,
    tri: Tensor,
    topology_hash: TopologyHash | None = None,
    pos_gradient_boost: float = 1.0,
) -> Tensor:
    """Antialias silhouette edges so that visibility changes produce gradients in ``pos``.

    Args:
        color: Shaded image, float32 of shape ``[batch, height, width, channels]``.
        rast: First output of :func:`rasterize` for the same geometry.
        pos: Clip-space vertex positions used to produce ``rast``, float32 of shape
            ``[batch, vertices, 4]``, or ``[vertices, 4]`` to share them across the batch.
        tri: Vertex indices used to produce ``rast``, int32 of shape ``[triangles, 3]``.
            Silhouettes are found from edges shared by index, so vertices split only for UV
            seams or normals turn those seams into silhouettes; pass position-only indices here.
        topology_hash: Result of :func:`antialias_construct_topology_hash` for ``tri``. Without
            it, the edge adjacency of ``tri`` is found on the first call and reused for as long
            as the same tensor object is passed and not modified in place, so pass the same
            tensor to every call rather than a fresh copy or cast. Only edits PyTorch records
            are noticed: after writing through ``tri.data`` or through memory shared with NumPy,
            pass a new tensor.
        pos_gradient_boost: Multiplier applied to the gradients flowing into ``pos``.

    Returns:
        Antialiased image with the same shape as ``color``.
    """
    _check(color, "color", ndim=4, dtype=torch.float32)
    _check(rast, "rast", ndim=4, last=4, dtype=torch.float32)
    shared = isinstance(pos, Tensor) and pos.ndim == 2
    layout = "[batch, vertices, 4] or [vertices, 4]"
    _check(pos, "pos", ndim=2 if shared else 3, last=4, dtype=torch.float32, layout=layout)
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    if color.shape[:3] != rast.shape[:3]:
        raise ValueError(
            "color must match rast in batch, height and width, got "
            f"{tuple(color.shape)} and {tuple(rast.shape)}."
        )
    if not shared and pos.shape[0] != rast.shape[0]:
        raise ValueError(
            f"pos batch size {pos.shape[0]} does not match rast batch size {rast.shape[0]}."
        )
    if not isinstance(pos_gradient_boost, SupportsFloat):
        raise TypeError(f"pos_gradient_boost must be a number, got {pos_gradient_boost!r}.")
    _check_devices(color=color, rast=rast, pos=pos, tri=tri)
    _check_indices(tri, pos.shape[-2])
    # Without triangles there is no adjacency to find, but a topology_hash is still checked.
    neighbours = None
    if topology_hash is not None or tri.shape[0]:
        neighbours = neighbours_for(tri, topology_hash)
    return get_backend(color.device, "antialias").antialias(
        color, rast, pos, tri, float(pos_gradient_boost), neighbours
    )


def antialias_construct_topology_hash(tri: Tensor) -> TopologyHash:
    """Precompute the part of :func:`antialias` that depends only on the index buffer.

    :func:`antialias` already reuses this between calls that pass the same ``tri`` tensor
    object. Building it explicitly helps when the indices arrive in a new tensor every time,
    or to choose when the work is done.

    Args:
        tri: Vertex indices, int32 of shape ``[triangles, 3]``.

    Returns:
        An opaque object to pass as ``topology_hash`` to :func:`antialias`, together with
        ``tri`` or any tensor of the same shape and contents on the same device.
    """
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    return TopologyHash(tri)


def check_rasterize_inputs(
    glctx: RasterizeContext,
    pos: Tensor,
    tri: Tensor,
    resolution: Sequence[int],
    ranges: Tensor | None,
) -> tuple[int, int]:
    """Validate the arguments of :func:`rasterize` and return the resolution as two ints."""
    range_mode = isinstance(pos, Tensor) and pos.ndim == 2
    _check(pos, "pos", ndim=2 if range_mode else 3, last=4, dtype=torch.float32)
    _check(tri, "tri", ndim=2, last=3, dtype=torch.int32)
    # rast stores triangle_id + 1 as float32, which holds integers exactly up to 2**24.
    if tri.shape[0] > 1 << 24:
        raise ValueError(
            f"tri has {tri.shape[0]} triangles, but at most 2**24 = {1 << 24} are supported: "
            "rast stores triangle ids as float32, which is exact only up to there."
        )
    _check_devices(pos=pos, tri=tri)
    _check_indices(tri, pos.shape[-2])
    if pos.device != glctx.device:
        raise ValueError(f"pos is on {pos.device} but the context is on {glctx.device}.")
    if range_mode:
        _check_ranges(ranges, tri.shape[0])
    return _check_resolution(resolution)


def _check_ranges(ranges: Tensor | None, num_tri: int) -> None:
    """Validate the ``ranges`` argument of range mode against the number of triangles."""
    if ranges is None:
        raise ValueError(
            "ranges is required when pos has shape [vertices, 4] (range mode); give pos a "
            "leading batch dimension to draw all triangles (instanced mode)."
        )
    _check(ranges, "ranges", ndim=2, last=2, dtype=torch.int32)
    for item, (start, count) in enumerate(ranges.tolist()):
        if start < 0 or count < 0 or start + count > num_tri:
            raise ValueError(
                f"ranges[{item}] = (start {start}, count {count}) does not lie within the "
                f"{num_tri} triangles of tri."
            )


def split_ranges(tri: Tensor, ranges: Tensor) -> list[Tensor]:
    """The triangles each batch item draws in range mode, as slices of ``tri``.

    ``ranges`` holds ``(start, count)`` per batch item, int32 of shape ``[batch, 2]``.
    """
    return [tri[start : start + count] for start, count in ranges.tolist()]


def join_ranges(
    pos: Tensor,
    resolution: tuple[int, int],
    ranges: Tensor,
    layers: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Stack the ``(rast, rast_db)`` of each batch item of range mode into one batch.

    Every item was rasterized with a batch of one from its own slice of ``tri`` (see
    :func:`split_ranges`). Its triangle ids are shifted to count from the start of ``tri``, so
    that :func:`interpolate` and :func:`antialias` can be given the whole ``tri`` tensor.
    """
    empty = pos.new_zeros(0, *resolution, 4)
    rasts, rast_dbs = [empty], [empty]
    for (start, _), (rast, rast_db) in zip(ranges.tolist(), layers, strict=True):
        ids = rast[..., 3:]
        rasts.append(torch.cat([rast[..., :3], torch.where(ids > 0, ids + start, ids)], -1))
        rast_dbs.append(rast_db)
    return torch.cat(rasts), torch.cat(rast_dbs)


def _check(
    t: Tensor,
    name: str,
    ndim: int,
    last: int | None = None,
    dtype: torch.dtype | None = None,
    layout: str | None = None,
) -> None:
    """Raise unless ``t`` is a tensor with ``ndim`` dimensions, the last of size ``last``.

    ``layout`` names the dimensions in the error message, such as ``"[batch, vertices, 4]"``.
    Without it, the usual layout of the argument called ``name`` is shown when it fits.
    """
    if not isinstance(t, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}.")
    if t.ndim != ndim or (last is not None and t.shape[-1] != last):
        if layout is None:
            layout = _LAYOUTS.get(name)
            fits = layout is not None and layout.count(",") + 1 == ndim
            if not fits or (last and not layout.endswith(f" {last}]")):
                layout = f"[{', '.join(['...'] * (ndim - 1) + [str(last) if last else '...'])}]"
        hint = ""
        unbatched = t.ndim == ndim - 1 and (last is None or t.shape[-1:] == (last,))
        if unbatched and layout.startswith("[batch") and " or " not in layout:
            hint = f"; add a batch dimension with {name}[None]"
        raise ValueError(f"{name} must have shape {layout}, got {tuple(t.shape)}{hint}.")
    if dtype is not None and t.dtype != dtype:
        cast = {torch.float32: "float", torch.int32: "int"}.get(dtype)
        hint = f"; convert it with {name}.{cast}()" if cast and name.isidentifier() else ""
        raise TypeError(f"{name} must be {dtype}, got {t.dtype}{hint}.")


_LAYOUTS = {
    "pos": "[batch, vertices, 4]",
    "tri": "[triangles, 3]",
    "rast": "[batch, height, width, 4]",
    "rast_db": "[batch, height, width, 4]",
    "color": "[batch, height, width, channels]",
    "tex": "[batch, tex_height, tex_width, channels]",
    "uv": "[batch, height, width, 2]",
    "uv_da": "[batch, height, width, 4]",
    "mip_level_bias": "[batch, height, width]",
}


def _check_devices(**tensors: Tensor | None) -> None:
    devices = {}
    for name, t in tensors.items():
        if t is None:
            continue
        if not isinstance(t, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}.")
        devices[name] = t.device
    if len(set(devices.values())) > 1:
        listing = ", ".join(f"{name} on {device}" for name, device in devices.items())
        raise ValueError(f"All inputs must be on the same device; got {listing}.")


def _index_range(tri: Tensor) -> tuple[int, int]:
    low, high = torch.stack(tri.aminmax()).tolist()
    return low, high


_index_ranges: TensorCache[tuple[int, int]] = TensorCache(_index_range)


def _check_indices(tri: Tensor, num_vertices: int, name: str = "tri") -> None:
    """Raise ``ValueError`` if ``tri`` refers to a vertex outside ``[0, num_vertices)``.

    Finding the smallest and largest index takes a reduction and, on a GPU, a synchronisation.
    Both are remembered per tensor object until it is modified in place, so an index buffer
    that is reused between calls is only scanned once, whatever it is checked against, and a new
    tensor with the same contents is scanned again. Only edits PyTorch records are noticed:
    writes through ``tri.data`` or through memory shared with NumPy leave the remembered range
    in place.

    Args:
        tri: Vertex indices, an integer tensor of any shape.
        num_vertices: Number of vertices the indices refer to.
        name: Argument name to use in the error message.
    """
    if tri.numel() == 0:
        return
    low, high = _index_ranges.get(tri)
    if low < 0 or high >= num_vertices:
        index = low if low < 0 else high
        valid = "there are no vertices"
        if num_vertices > 0:
            valid = f"valid indices are 0 to {num_vertices - 1} for {num_vertices} vertices"
        raise ValueError(f"{name} refers to vertex {index}, but {valid}.")


def _diff_channels(diff_attrs: Sequence[int] | str | None, num_channels: int) -> list[int]:
    """Channel indices selected by the ``diff_attrs`` argument of :func:`interpolate`."""
    if diff_attrs is None:
        return []
    expected = (
        f"diff_attrs must be a sequence of channel indices, 'all' or None, got {diff_attrs!r}."
    )
    if isinstance(diff_attrs, str):
        if diff_attrs != "all":
            raise ValueError(expected)
        return list(range(num_channels))
    try:
        channels = [operator.index(c) for c in diff_attrs]
    except TypeError:
        raise TypeError(expected) from None
    if any(c < 0 or c >= num_channels for c in channels):
        raise ValueError(f"diff_attrs {channels} out of range for {num_channels} channels.")
    return channels


def _check_resolution(resolution: Sequence[int]) -> tuple[int, int]:
    """Return ``resolution`` as ``(height, width)``; it may be a list, tuple, tensor or array."""
    # Tensors and arrays become lists, which leaves sets, mappings and iterators to be refused:
    # their order is not the caller's, or reading them uses them up.
    values = resolution.tolist() if hasattr(resolution, "tolist") else resolution
    try:
        if not isinstance(values, (list, tuple)) or any(isinstance(r, bool) for r in values):
            raise TypeError
        height, width = (operator.index(r) for r in values)
    except (TypeError, ValueError):
        raise ValueError(
            f"resolution must be (height, width) as two ints, got {resolution!r}."
        ) from None
    if height <= 0 or width <= 0:
        raise ValueError(f"resolution must be positive, got {resolution!r}.")
    return height, width
