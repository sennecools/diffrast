"""Edge adjacency for antialias: the topology hash and the per-tensor cache behind it."""

import copy
import gc
import io
import pickle
import threading

import pytest
import torch

import diffrast as dr
from diffrast import backends, topology
from diffrast._cache import TensorCache
from diffrast.backends import reference
from diffrast.ops import antialias_construct_topology_hash
from diffrast.topology import TopologyHash

QUAD = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)


@pytest.fixture(
    params=[("cpu", "reference"), pytest.param(("cuda", "triton"), marks=pytest.mark.gpu)],
    ids=["reference", "triton"],
)
def device(request):
    """Run a test on the reference backend on the CPU and on the Triton backend on the GPU."""
    name, backend = request.param
    if backend == "triton":
        pytest.importorskip("triton")
    previous = backends._selected
    dr.set_backend(backend)
    yield torch.device(name)
    backends._selected = previous


@pytest.fixture
def counted(monkeypatch):
    """Count how often the adjacency is rebuilt, starting from an empty cache."""
    calls = []

    def compute(tri):
        calls.append(tri.shape[0])
        return reference.edge_neighbours(tri)

    cache = TensorCache(compute, max_entries=16)
    monkeypatch.setattr(topology, "_neighbour_cache", cache)
    return calls


def folded_strip(device="cpu", columns=6):
    """A strip of quads with shared vertices, folded back over itself along its middle."""
    xs = torch.linspace(-0.8, 0.8, columns + 1)
    xs = torch.where(xs > 0, 0.4 - xs, xs)
    bottom = torch.stack([xs, torch.full_like(xs, -0.6)], -1)
    top = torch.stack([xs + 0.05, torch.full_like(xs, 0.7)], -1)
    xy = torch.cat([bottom, top])
    depth = torch.linspace(-0.5, 0.5, columns + 1).repeat(2)[:, None]
    w = torch.linspace(1.0, 1.6, 2 * columns + 2)[:, None]
    pos = torch.cat([xy * w, depth * w, w], -1)[None]
    tri = []
    for column in range(columns):
        a, b = column, column + 1
        c, d = b + columns + 1, a + columns + 1
        tri += [[a, b, c], [a, c, d]]
    return pos.to(device), torch.tensor(tri, dtype=torch.int32, device=device)


def scene(device, resolution=(24, 32)):
    pos, tri = folded_strip(device)
    pos = torch.cat([pos, pos * torch.tensor([0.8, -0.9, 1.0, 1.0], device=device)])
    ctx = dr.RasterizeContext(device)
    rast, _ = dr.rasterize(ctx, pos, tri, resolution)
    gen = torch.Generator().manual_seed(0)
    color = torch.rand(2, *resolution, 3, generator=gen).to(device)
    return color, rast, pos, tri


def run(color, rast, pos, tri, **kwargs):
    pos = pos.clone().requires_grad_(True)
    out = dr.antialias(color, rast, pos, tri, **kwargs)
    out.square().sum().backward()
    return out.detach(), pos.grad


class TestEdgeNeighbours:
    def test_quad(self):
        # Edge k is opposite vertex k, so the diagonal (0, 2) is edge 1 of [0, 1, 2], numbered
        # 1, and edge 2 of [0, 2, 3], numbered 5.
        neighbours = reference.edge_neighbours(QUAD)
        assert neighbours.dtype == torch.int64
        assert neighbours.tolist() == [[-1, 5, -1], [-1, -1, 1]]

    def test_edge_shared_by_three_triangles_has_no_neighbour(self):
        tri = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 2, 4]], dtype=torch.int32)
        assert (reference.edge_neighbours(tri) == -1).all()

    def test_does_not_depend_on_winding_or_index_size(self):
        tri = torch.tensor([[70000, 5, 900000], [900000, 70000, 12]], dtype=torch.int32)
        assert reference.edge_neighbours(tri).tolist() == [[-1, 5, -1], [-1, -1, 1]]

    def test_silhouettes_are_the_same_with_precomputed_neighbours(self):
        pos, tri = folded_strip()
        tri = tri.long()
        direct = reference._silhouette_edges(pos, tri)
        reused = reference._silhouette_edges(pos, tri, reference.edge_neighbours(tri))
        assert torch.equal(direct, reused)
        # Both the outline and the fold are silhouettes; the other shared edges are not.
        assert direct.any() and not direct.all()


class TestTopologyHash:
    def test_results_are_identical_with_and_without(self, device):
        color, rast, pos, tri = scene(device)
        topology_hash = antialias_construct_topology_hash(tri)
        assert isinstance(topology_hash, TopologyHash)

        plain = run(color, rast, pos, tri)
        cached = run(color, rast, pos, tri)
        hashed = run(color, rast, pos, tri, topology_hash=topology_hash)
        assert (plain[0] != color).any()
        for other in (cached, hashed):
            torch.testing.assert_close(other[0], plain[0], atol=1e-6, rtol=0)
            torch.testing.assert_close(other[1], plain[1], atol=1e-3, rtol=1e-4)

    def test_backend_gets_the_same_adjacency_either_way(self, device, monkeypatch):
        color, rast, pos, tri = scene(device)
        backend = backends.get_backend(device, "antialias")
        seen = []
        original = backend.antialias

        def spy(*args):
            seen.append(args[5])
            return original(*args)

        monkeypatch.setattr(backend, "antialias", spy)
        dr.antialias(color, rast, pos, tri)
        dr.antialias(color, rast, pos, tri, antialias_construct_topology_hash(tri))
        assert torch.equal(seen[0], seen[1])
        assert torch.equal(seen[0], reference.edge_neighbours(tri))

    def test_is_not_rebuilt_per_call(self, counted):
        color, rast, pos, tri = scene("cpu")
        topology_hash = antialias_construct_topology_hash(tri)
        for _ in range(3):
            dr.antialias(color, rast, pos, tri, topology_hash=topology_hash)
        assert not counted

    def test_works_with_an_equal_tensor(self):
        color, rast, pos, tri = scene("cpu")
        topology_hash = antialias_construct_topology_hash(tri)
        expected = dr.antialias(color, rast, pos, tri)
        assert torch.equal(dr.antialias(color, rast, pos, tri.clone(), topology_hash), expected)

    def test_rejects_another_shape(self):
        color, rast, pos, tri = scene("cpu")
        topology_hash = antialias_construct_topology_hash(tri)
        with pytest.raises(ValueError, match=r"built for a tri of shape \(12, 3\) on cpu"):
            dr.antialias(color, rast, pos, tri[:4], topology_hash=topology_hash)

    @pytest.mark.gpu
    def test_rejects_another_device(self):
        color, rast, pos, tri = scene("cuda")
        topology_hash = antialias_construct_topology_hash(tri.cpu())
        with pytest.raises(ValueError, match=r"on cpu, got shape .* on cuda:0"):
            dr.antialias(color, rast, pos, tri, topology_hash=topology_hash)

    def test_rejects_a_tensor_modified_since(self):
        color, rast, pos, tri = scene("cpu")
        topology_hash = antialias_construct_topology_hash(tri)
        tri[0] = tri[0].flip(0)
        with pytest.raises(ValueError, match="modified in place"):
            dr.antialias(color, rast, pos, tri, topology_hash=topology_hash)

    def test_rejects_other_objects(self):
        color, rast, pos, tri = scene("cpu")
        with pytest.raises(TypeError, match="topology_hash must come from"):
            dr.antialias(color, rast, pos, tri, topology_hash=torch.zeros(3))

    def test_constructor_checks_tri(self):
        with pytest.raises(TypeError, match=r"tri must be torch\.int32"):
            antialias_construct_topology_hash(QUAD.long())
        with pytest.raises(ValueError, match=r"tri must have shape \[triangles, 3\]"):
            antialias_construct_topology_hash(QUAD.flatten())

    def test_repr(self):
        assert repr(antialias_construct_topology_hash(QUAD)) == (
            "TopologyHash(triangles=2, device='cpu')"
        )

    def test_is_checked_without_triangles(self):
        color, rast, pos, tri = scene("cpu")
        with pytest.raises(TypeError, match="topology_hash must come from"):
            dr.antialias(color, rast, pos, tri[:0], topology_hash="junk")
        with pytest.raises(ValueError, match=r"built for a tri of shape \(12, 3\)"):
            dr.antialias(color, rast, pos, tri[:0], antialias_construct_topology_hash(tri))
        empty = antialias_construct_topology_hash(tri[:0])
        assert torch.equal(dr.antialias(color, rast, pos, tri[:0], empty), color)

    @pytest.mark.parametrize(
        "duplicate",
        [lambda h: pickle.loads(pickle.dumps(h)), copy.deepcopy, copy.copy],
        ids=["pickle", "deepcopy", "copy"],
    )
    def test_can_be_pickled_and_copied(self, device, duplicate):
        color, rast, pos, tri = scene(device)
        original = antialias_construct_topology_hash(tri)
        topology_hash = duplicate(original)
        assert isinstance(topology_hash, TopologyHash)
        assert repr(topology_hash) == repr(original)
        assert torch.equal(topology_hash.neighbours(tri), original.neighbours(tri))
        # The Triton kernels add the blends in no fixed order, so two runs differ by rounding.
        out = dr.antialias(color, rast, pos, tri, topology_hash)
        torch.testing.assert_close(out, dr.antialias(color, rast, pos, tri), atol=1e-6, rtol=0)
        with pytest.raises(ValueError, match="built for a tri of shape"):
            dr.antialias(color, rast, pos, tri[:4], topology_hash)

    def test_can_be_saved_with_torch(self):
        color, rast, pos, tri = scene("cpu")
        buffer = io.BytesIO()
        torch.save({"topology": antialias_construct_topology_hash(tri)}, buffer)
        buffer.seek(0)
        topology_hash = torch.load(buffer, weights_only=False)["topology"]
        expected = dr.antialias(color, rast, pos, tri)
        assert torch.equal(dr.antialias(color, rast, pos, tri, topology_hash), expected)


class TestAutomaticReuse:
    def test_same_tensor_is_analysed_once(self, counted):
        color, rast, pos, tri = scene("cpu")
        first = dr.antialias(color, rast, pos, tri)
        second = dr.antialias(color, rast, pos, tri)
        assert len(counted) == 1
        assert torch.equal(first, second)

    def test_in_place_edit_rebuilds(self, counted):
        color, rast, pos, tri = scene("cpu")
        before = dr.antialias(color, rast, pos, tri)
        # Pointing a corner of the first triangle at another vertex in the same place turns the
        # edges it shared into silhouettes, so a stale adjacency would give a different image.
        pos = torch.cat([pos, pos[:, 8:9]], 1)
        tri[0, 2] = 14
        edited = dr.antialias(color, rast, pos, tri)
        assert len(counted) == 2
        fresh = dr.antialias(color, rast, pos, tri.clone())
        assert torch.equal(edited, fresh)
        assert not torch.equal(edited, before)

    def test_edit_through_a_view_rebuilds(self, counted):
        color, rast, pos, tri = scene("cpu")
        dr.antialias(color, rast, pos, tri)
        tri[:1][0, 2] = 13
        dr.antialias(color, rast, pos, tri)
        assert len(counted) == 2

    def test_equal_tensor_gets_its_own_entry(self, counted):
        color, rast, pos, tri = scene("cpu")
        expected = dr.antialias(color, rast, pos, tri)
        copy = tri.clone()
        assert torch.equal(dr.antialias(color, rast, pos, copy), expected)
        assert len(counted) == 2
        assert len(topology._neighbour_cache) == 2

    def test_no_growth_after_tensors_are_freed(self, counted):
        color, rast, pos, tri = scene("cpu")
        for _ in range(40):
            dr.antialias(color, rast, pos, tri.clone())
            gc.collect()
            assert len(topology._neighbour_cache) == 0
        assert len(counted) == 40

    def test_live_tensors_are_bounded(self, counted):
        color, rast, pos, tri = scene("cpu")
        alive = [tri.clone() for _ in range(40)]
        for t in alive:
            dr.antialias(color, rast, pos, t)
        assert len(topology._neighbour_cache) == 16

    def test_zero_triangles_are_not_cached(self, counted):
        color, rast, pos, tri = scene("cpu")
        out = dr.antialias(color, rast, pos, tri[:0])
        assert torch.equal(out, color)
        assert out is not color
        assert not counted

    def test_concurrent_calls_give_the_same_image(self, counted):
        color, rast, pos, tri = scene("cpu")
        expected = dr.antialias(color, rast, pos, tri.clone())
        own = [tri.clone() for _ in range(4)]
        failures = []

        def work(index):
            try:
                for step in range(25):
                    # Alternate between a tensor all threads share, one that only this thread
                    # uses, and a new one, so that entries are added and dropped concurrently.
                    t = (tri, own[index], tri.clone())[step % 3]
                    if not torch.equal(dr.antialias(color, rast, pos, t), expected):
                        failures.append((index, step))
            except Exception as error:
                failures.append(error)

        threads = [threading.Thread(target=work, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures


class TestTensorCache:
    def make(self, max_entries=4):
        calls = []

        def compute(tensor):
            calls.append(1)
            return tensor.sum().item()

        return TensorCache(compute, max_entries), calls

    def test_hit_and_miss(self):
        cache, calls = self.make()
        a, b = torch.arange(3), torch.arange(4)
        assert [cache.get(a), cache.get(a), cache.get(b), cache.get(a)] == [3, 3, 6, 3]
        assert len(calls) == 2
        assert len(cache) == 2

    def test_in_place_edit_is_seen(self):
        cache, calls = self.make()
        a = torch.arange(3)
        assert cache.get(a) == 3
        a[0] = 10
        assert cache.get(a) == 13
        a.add_(1)
        assert cache.get(a) == 16
        a[1:].mul_(2)
        assert cache.get(a) == 21
        assert len(calls) == 4
        assert len(cache) == 1

    def test_new_storage_is_seen(self):
        # Assigning to .data leaves the version counter alone.
        cache, _ = self.make()
        a = torch.arange(3)
        assert cache.get(a) == 3
        a.data = torch.arange(5)
        assert cache.get(a) == 10
        a.resize_(2)
        assert cache.get(a) == 1

    def test_entry_dies_with_its_tensor(self):
        cache, _ = self.make()
        a = torch.arange(3)
        cache.get(a)
        del a
        gc.collect()
        assert len(cache) == 0

    def test_new_tensor_at_a_recycled_address_is_not_confused(self):
        cache, _ = self.make()
        for size in range(1, 200):
            # Each tensor is freed before the next is made, so addresses repeat.
            assert cache.get(torch.ones(size)) == size

    def test_oldest_entry_is_dropped(self):
        cache, calls = self.make(max_entries=2)
        a, b, c = torch.arange(2), torch.arange(3), torch.arange(4)
        for tensor in (a, b, c):
            cache.get(tensor)
        assert len(cache) == 2
        cache.get(b)
        cache.get(c)
        assert len(calls) == 3
        cache.get(a)
        assert len(calls) == 4

    def test_inference_tensors_are_not_cached(self):
        cache, calls = self.make()
        with torch.inference_mode():
            a = torch.arange(3)
        assert cache.get(a) == 3
        assert cache.get(a) == 3
        assert len(calls) == 2
        assert len(cache) == 0

    def test_clear(self):
        cache, calls = self.make()
        a = torch.arange(3)
        cache.get(a)
        cache.clear()
        cache.get(a)
        assert len(calls) == 2

    def test_concurrent_use(self):
        cache, _ = self.make(max_entries=3)
        shared = [torch.full((5,), value) for value in range(6)]
        failures = []

        def work():
            try:
                for step in range(300):
                    tensor = shared[step % 6] if step % 2 else torch.full((5,), step)
                    if cache.get(tensor) != 5 * int(tensor[0]):
                        failures.append(step)
            except Exception as error:
                failures.append(error)

        threads = [threading.Thread(target=work) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not failures
        assert len(cache) <= 3
