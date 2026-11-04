"""LSA numerical and cache lifecycle checks; run on either ROCm GPU family."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import pytest

from exllamav3.modules.lsa import LSAIndexer, LSASettings
from exllamav3.cache.lsa import CacheLayer_lsa, CacheLayer_lsa_quant, CacheLayer_lsa_i8f8, CacheLayer_lsa_r4d


def make_indexer(device="cpu", **settings):
    torch.manual_seed(31)
    s = LSASettings("unused.pt", window_size=64, pool_size=4, bucket_size=8,
                    bucket_cap=16, top_k=settings.pop("top_k", 32), top_p=8, **settings)
    qmap, kmap = torch.randn(64, 16), torch.randn(64, 16)
    indexer = LSAIndexer(s, qmap, kmap, head_dim=64, kv_heads=2, rotary_dim=32)
    indexer.load(torch.device(device))
    return indexer


def fake_attention(indexer):
    return SimpleNamespace(lsa_indexer=indexer, num_kv_heads=2, head_dim=64, sm_scale=0.125)


def test_projection_matches_gqa_logit_subterm():
    idx = make_indexer()
    q = torch.randn(2, 3, 4, 64)
    k = torch.randn(2, 3, 2, 64)
    query, key = idx.project(q, k, {})
    expected_q = torch.cat((q[:, :, :2, 32:].sum(2), q[:, :, 2:, 32:].sum(2)), -1)
    expected_k = torch.cat((k[:, :, 0, 32:], k[:, :, 1, 32:]), -1)
    torch.testing.assert_close(query, F.normalize(expected_q @ idx.query_map, dim=-1))
    torch.testing.assert_close(key, F.normalize(expected_k @ idx.key_map, dim=-1))


def test_full_budget_has_every_causal_position():
    idx = make_indexer(top_k=512)
    keys = F.normalize(torch.randn(3, 64, 16), dim=-1)
    centroids, members = idx.cluster(keys)
    members = torch.where(members >= 0, members + torch.arange(3)[:, None, None] * 64, -1)
    positions = torch.tensor([0, 63, 64, 91, 127])
    selected = idx.select(keys.flatten(0, 1)[positions], positions, centroids.flatten(0, 1), members.flatten(0, 1))
    for p, row in zip(positions.tolist(), selected):
        ids = row[row >= 0].sort().values
        # top_p is eight: at 127 only the first closed block is visible.
        torch.testing.assert_close(ids, torch.arange(p + 1, dtype=torch.int32))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_flat_query_batching_preserves_selection_and_ties(device, monkeypatch):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA GPU required")
    import exllamav3.modules.lsa as lsa
    idx = make_indexer(device)
    keys = F.normalize(torch.randn(4, 64, 16, device=device), dim=-1)
    centroids, members = idx.cluster(keys)
    offsets = torch.arange(4, device=device)[:, None, None] * 64
    members = torch.where(members >= 0, members + offsets, -1).flatten(0, 1)
    centroids = centroids.flatten(0, 1)
    positions = torch.tensor([64, 99, 127, 190, 255], device=device)
    queries = torch.zeros(5, 16, device=device)
    queries[1:] = F.normalize(torch.randn(4, 16, device=device), dim=-1)
    expected = idx.select(queries, positions, centroids, members)
    monkeypatch.setattr(lsa, "_FLAT_SORT_MAX_ELEMENTS", 32)
    actual = idx.select(queries, positions, centroids, members)
    torch.testing.assert_close(actual, expected)


def test_tp_reduces_before_normalizing_and_includes_headless_ranks():
    idx = make_indexer()
    q = torch.randn(1, 3, 4, 64)
    k = torch.randn(1, 3, 2, 64)
    expected = idx.project(q, k, {})
    other_query = q[:, :, 2:, 32:].sum(2) @ idx.query_map[32:]
    other_key = k[:, :, 1, 32:] @ idx.key_map[32:]
    class Backend:
        def all_reduce(self, joined):
            joined.add_(torch.stack((other_query, other_key)))
    idx.tp_mode = True
    query, key = idx.project(q[:, :, :2], k[:, :, :1], {"backend": Backend()})
    torch.testing.assert_close(query, expected[0])
    torch.testing.assert_close(key, expected[1])
    from dataclasses import replace
    idx.settings = replace(idx.settings, query_heads="rank")
    local_query, global_key = idx.project(q[:, :, :2], k[:, :, :1], {"backend": Backend()})
    torch.testing.assert_close(local_query, F.normalize(q[:, :, :2, 32:].sum(2) @ idx.query_map[:32], dim=-1))
    torch.testing.assert_close(global_key, expected[1])
    seen = []
    backend = SimpleNamespace(all_reduce=lambda joined: seen.append(joined.clone()))
    idx.empty_project(torch.zeros(1, 3, 512), {"backend": backend, "attn_mode": "flash_attn"})
    assert len(seen) == 1 and seen[0].shape == (2, 1, 3, 16) and not seen[0].count_nonzero()


def test_cache_versions_survive_copy_and_cpu_restore():
    idx = make_indexer()
    a = fake_attention(idx)
    cache = CacheLayer_lsa(None, a, 1, 1024)
    cache.alloc(torch.device("cpu"))
    bt = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    keys = F.normalize(torch.randn(1, 192, 16), dim=-1)
    cache.append_index(keys, bt, torch.tensor([0]))
    first = cache.buckets(idx, bt[0], 2)
    saved = [tensor[2].clone() for tensor in cache.get_tensors()]
    cache.append_index(F.normalize(torch.randn(1, 192, 16), dim=-1), bt, torch.tensor([0]))
    second = cache.buckets(idx, bt[0], 2)
    assert not torch.equal(first[0], second[0])
    for tensor, page in zip(cache.get_tensors(), saved):
        tensor[2].copy_(page)
    restored = cache.buckets(idx, bt[0], 2)
    torch.testing.assert_close(restored[0], first[0])
    torch.testing.assert_close(restored[1], first[1])
    cache.copy_page(cache, 2, 1, 128)
    copied_bt = torch.tensor([1, 0, 3, 2], dtype=torch.int32)
    copied = cache.buckets(idx, copied_bt, 2)
    torch.testing.assert_close(copied[0], first[0])
    assert cache.storage_size() == sum(t.numel() * t.element_size() for t in cache.get_tensors()) + cache.centroids.numel() * 4 + cache.members.numel() * 4
    cache.free()
    assert not cache.bucket_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
@pytest.mark.parametrize("quant", [False, True, "compand", "i8f8", "r4d"])
def test_paged_sparse_attention_matches_independent_gather(quant):
    dev = torch.device("cuda:0")
    idx = make_indexer(dev)
    a = fake_attention(idx)
    cls = {False: CacheLayer_lsa, True: CacheLayer_lsa_quant, "compand": CacheLayer_lsa_quant,
           "i8f8": CacheLayer_lsa_i8f8, "r4d": CacheLayer_lsa_r4d}[quant]
    kwargs = {"k_bits": 6, "v_bits": 4} if quant is True or quant == "compand" else {}
    if quant == "compand":
        kwargs["compand_a"] = 0.5
    cache = cls(None, a, 1, 1024, **kwargs)
    cache.alloc(dev)
    torch.manual_seed(83)
    q = torch.randn(1, 195, 4, 64, device=dev, dtype=torch.float16)
    k = torch.randn(1, 195, 2, 64, device=dev, dtype=torch.float16)
    v = torch.randn_like(k)
    bt = torch.tensor([[2, 0, 3, 1]], device=dev, dtype=torch.int32)
    actual = []
    for start, end in [(0, 128), (128, 192), (192, 195)]:
        past = torch.tensor([start], dtype=torch.int32)
        actual.append(idx.attend(a, q[:, start:end], k[:, start:end], v[:, start:end], {},
                                 cache, bt, past, past.to(dev)))
    actual = torch.cat(actual, 1)
    query, key = idx.project(q, k, {})
    centroids, members = idx.cluster(key[:, :192].reshape(3, 64, 16))
    members = torch.where(members >= 0, members + torch.arange(3, device=dev)[:, None, None] * 64, -1)
    selected = idx.select(query[0], torch.arange(195, device=dev, dtype=torch.int32),
                          centroids.flatten(0, 1), members.flatten(0, 1))
    dk, dv = cache.get_kv(torch.tensor([195], device=dev, dtype=torch.int32), bt)
    reference = []
    for row in range(195):
        ids = selected[row][selected[row] >= 0].long()
        pages, offsets = bt[0, ids // 256].long(), ids % 256
        kr = dk[pages, offsets].float().repeat_interleave(2, 1)
        vr = dv[pages, offsets].float().repeat_interleave(2, 1)
        scores = torch.einsum("hd,khd->hk", q[0, row].float(), kr) * a.sm_scale
        reference.append(torch.einsum("hk,khd->hd", scores.softmax(-1), vr))
    torch.testing.assert_close(actual[0].float(), torch.stack(reference), rtol=3e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
def test_gpu_ward_matches_reference_and_expansion():
    from exllamav3.vendor.lsa.ward_reference import ward_merge
    idx = make_indexer("cuda:0")
    keys = F.normalize(torch.randn(3, 64, 16), dim=-1)
    labels = ward_merge(keys.reshape(3, 16, 4, 16).sum(2), torch.full((3, 16), 4.), 8, 16)
    centroids, members = idx.cluster(keys.cuda())
    for block in range(3):
        for bucket in range(8):
            expected = (labels[block].repeat_interleave(4) == bucket).nonzero().flatten()
            actual = members[block, bucket].cpu()
            torch.testing.assert_close(actual[actual >= 0].long(), expected)
    idx_cpu = make_indexer()
    positions = torch.tensor([64, 99, 127, 190])
    queries = F.normalize(torch.randn(4, 16), dim=-1)
    offsets = torch.arange(3)[:, None, None] * 64
    members = torch.where(members >= 0, members + offsets.cuda(), -1).flatten(0, 1)
    actual = idx.select(queries.cuda(), positions.cuda(), centroids.flatten(0, 1), members).cpu()
    expected = idx_cpu.select(queries, positions, centroids.cpu().flatten(0, 1), members.cpu())
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
@pytest.mark.parametrize("cls", [CacheLayer_lsa, CacheLayer_lsa_i8f8, CacheLayer_lsa_r4d])
def test_index_planes_follow_native_page_rotation(cls):
    from exllamav3.ext import exllamav3_ext as ext
    idx = make_indexer("cuda:0")
    cache = cls(None, fake_attention(idx), 1, 1024)
    cache.alloc(torch.device("cuda:0"))
    bt = torch.tensor([[2, 0, 3, 1]], device="cuda:0", dtype=torch.int32)
    keys = F.normalize(torch.randn(1, 192, 16, device="cuda:0"), dim=-1)
    cache.append_index(keys, bt, torch.tensor([0]))
    for tensor in cache.get_tensors()[:-2]:
        tensor[2].fill_(63)
        tensor[1].fill_(42)
    saved = [tensor[2].clone() for tensor in cache.get_tensors()]
    before = cache.buckets(idx, bt[0], 2)
    order = torch.tensor([-1, 2, 2, 1, 1, -1], device="cuda:0", dtype=torch.int32)
    for tensor in cache.get_tensors():
        ext.cache_rotate(tensor, order, torch.empty_like(tensor[0]))
    for tensor, expected in zip(cache.get_tensors(), saved):
        torch.testing.assert_close(tensor[1], expected)
    rotated = cache.buckets(idx, torch.tensor([1, 0, 3, 2], device="cuda:0", dtype=torch.int32), 2)
    torch.testing.assert_close(rotated[0], before[0])
    torch.testing.assert_close(rotated[1], before[1])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_tree_beam_matches_official_reference(device, precision):
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA GPU required")
    from exllamav3.modules.lsa_tree import TreeStore, NODE_SCALE
    from exllamav3.vendor.lsa.reference import Buckets
    from exllamav3.vendor.lsa.hierarchy_reference import build_hierarchy, read_hierarchy
    idx = make_indexer(device, top_level=3, node_bucket_size=2,
                       node_cluster_block_size=[32, 16, 8], beam_width=[2, 3, 2], start_nodes=8,
                       node_centroid_precision=precision)
    torch.manual_seed(92)
    keys = F.normalize(torch.randn(32, 64, 16), dim=-1)
    cpu = make_indexer()
    centroids, members, sums = cpu.cluster(keys, return_sums=True)
    members = torch.where(members >= 0, members + torch.arange(32)[:, None, None] * 64, -1)
    buckets = Buckets(centroids.flatten(0, 1), members.flatten(0, 1),
                      (members >= 0).sum(-1).int().flatten(), 8, 64)
    hierarchy = build_hierarchy(keys.flatten(0, 1), buckets, 3, 2, [32, 16, 8])
    # Reference search on the same published centroid storage precision.
    nodes = [(v * NODE_SCALE).to(torch.float8_e4m3fn).float() / NODE_SCALE if precision == "fp8"
             else v.bfloat16().float() for v in hierarchy.node_centroids]
    hierarchy = hierarchy._replace(node_centroids=nodes)
    positions = torch.tensor([63, 127, 255, 256, 511, 512, 777, 1023, 1536, 2047])
    queries = F.normalize(torch.randn(len(positions), 16), dim=-1)
    expected, _ = read_hierarchy(queries, positions, buckets, hierarchy, [2, 3, 2], 8, 8, 32, 4, 64,
                                 page_size=128, growth_buckets=32)
    store = TreeStore(idx.settings, len(sums.flatten(0, 1)), 16, device, page_size=128, growth_buckets=32)
    view = store.view(sums.flatten(0, 1).to(device), list(range(256)))
    actual = idx.select(queries.to(device), positions.to(device), buckets.bucket_centroids.to(device),
                        buckets.bucket_members.to(device), view).cpu()
    torch.testing.assert_close(actual, expected)
    # Reusing a hierarchy must not rebuild its already closed node groups.
    before = [list(group.cached) for group in store.groups]
    store.view(sums.flatten(0, 1).to(device), list(range(256)))
    assert before == [list(group.cached) for group in store.groups]


def test_tree_cache_rebuilds_after_overwrite_and_restored_prefix():
    idx = make_indexer(top_level=3, node_bucket_size=2, node_cluster_block_size=32, beam_width=2)
    cache = CacheLayer_lsa(None, fake_attention(idx), 1, 1024)
    cache.alloc(torch.device("cpu"))
    bt = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    keys = F.normalize(torch.randn(1, 1024, 16), dim=-1)
    cache.append_index(keys, bt, torch.tensor([0]))
    view = cache.buckets(idx, bt[0], 16, with_tree=True)[2]
    expected = [(centroids.clone(), children.clone()) for centroids, children in view.levels]
    persisted = [t.clone() for t in cache.get_tensors()]
    # The bounded node cache cannot retain all three incarnations of these pages.
    for _ in range(2):
        cache.append_index(F.normalize(torch.randn_like(keys), dim=-1), bt, torch.tensor([0]))
        fresh = cache.buckets(idx, bt[0], 16, with_tree=True)[2]
        assert not torch.equal(fresh.levels[0][0], expected[0][0])
    for tensor, saved in zip(cache.get_tensors(), persisted):
        tensor.copy_(saved)
    restored = cache.buckets(idx, bt[0], 16, with_tree=True)[2]
    for (centroids, children), (wanted, wanted_children) in zip(restored.levels, expected):
        torch.testing.assert_close(centroids, wanted)
        torch.testing.assert_close(children, wanted_children)
    arrays = cache.get_tensors() + [cache.centroids, cache.members, cache.bucket_sums]
    for group in cache.tree_store.groups:
        arrays += [group.centroids, group.sums, group.children]
    assert cache.storage_size() == sum(t.numel() * t.element_size() for t in arrays)


def test_hierarchy_child_identifiers_have_bounded_hash_cost():
    from exllamav3.modules.lsa_tree import NodeGroups
    torch.manual_seed(908)
    group = NodeGroups(1, 32, 2, 16, "bf16", "cpu")
    sums = torch.randn(32, 16)
    nested_leaf_ids = [(tuple(range(64)), i) for i in range(32)]
    _, _, _, identities = group.read(sums, nested_leaf_ids)
    assert all(len(i) == 2 and all(isinstance(part, int) for part in i) for i in identities)
    from collections import OrderedDict
    class CountedCache(OrderedDict):
        lookups = 0
        def __getitem__(self, key):
            self.lookups += 1
            return super().__getitem__(key)
    group.cached = CountedCache(group.cached)
    _, _, _, same = group.read(sums, nested_leaf_ids)
    assert group.cached.lookups <= 3  # Group signatures are hashed per group, never once per child.
    assert identities == same
    _, _, _, replaced = group.read(sums + 1, [(tuple(range(64)), i + 32) for i in range(32)])
    assert not set(identities) & set(replaced)  # Reused slots never alias a prior node incarnation.


def test_lsa_r4d_storage_does_not_require_libr4d(monkeypatch):
    from exllamav3 import model_init
    from exllamav3.modules.attention_fn import r4d_attn
    monkeypatch.setenv("EXL3_R4D_ATTN", "bf16")
    monkeypatch.setenv("EXL3_R4D_OFF", "0")
    monkeypatch.setattr(r4d_attn, "r4d_lib_path", lambda: None)
    assert model_init._r4d_native_cache_wanted(lsa=True)


def test_quantized_dispatch_sanity_is_exclusive_of_unquantized_cache():
    from exllamav3.modules.attention_fn.common import AttnArgs
    q = torch.zeros((1, 1, 4, 64), dtype=torch.float16)
    k = torch.zeros((1, 1, 2, 64), dtype=torch.float16)
    args = AttnArgs(1, 1, 4, 64, 256, 2, q, k, k, None, None, True, 0.125,
                    None, None, None, 0., torch.zeros((1, 1), dtype=torch.int32),
                    torch.zeros(1, dtype=torch.int32), q_cache=(k, k, k, k, 5, 4))
    args.sanity_check()
    with pytest.raises(AssertionError):
        args._replace(i8f8_cache=(k,)).sanity_check()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
def test_r4d_dense_dispatch_keeps_native_storage():
    from exllamav3.modules.attention_fn import attn_dispatch
    idx = make_indexer("cuda:0")
    cache = CacheLayer_lsa_r4d(None, fake_attention(idx), 1, 1024)
    cache.alloc(torch.device("cuda:0"))
    q = torch.randn((1, 5, 4, 64), device="cuda:0", dtype=torch.float16)
    k = torch.randn((1, 5, 2, 64), device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    bt = torch.tensor([[2, 0, 3, 1]], device="cuda:0", dtype=torch.int32)
    past = torch.zeros(1, device="cuda:0", dtype=torch.int32)
    actual = attn_dispatch(q, k, v, cache=cache, cache_idx=0, block_table=bt,
                           cache_seqlens=past, causal=False, sm_scale=0.125)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2).float(),
        k.repeat_interleave(2, 2).transpose(1, 2).float(),
        v.repeat_interleave(2, 2).transpose(1, 2).float(), scale=0.125).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=3e-3)
    saved_k, saved_v = cache.get_kv(past + 5, bt)
    torch.testing.assert_close(saved_k[2, :5], k[0].bfloat16().half(), atol=0, rtol=0)
    torch.testing.assert_close(saved_v[2, :5], v[0].bfloat16().half(), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm/CUDA GPU required")
def test_quantized_dense_prefix_uses_actual_history_bound_with_shuffled_pages():
    from exllamav3.modules.attention_fn import attn_dispatch
    idx = make_indexer("cuda:0")
    cache = CacheLayer_lsa_quant(None, fake_attention(idx), 1, 1024, k_bits=5, v_bits=4)
    cache.alloc(torch.device("cuda:0"))
    q = torch.randn((1, 5, 4, 64), device="cuda:0", dtype=torch.float16)
    k = torch.randn((1, 5, 2, 64), device="cuda:0", dtype=torch.float16)
    v = torch.randn_like(k)
    bt = torch.tensor([[2, 0, 3, 1]], device="cuda:0", dtype=torch.int32)
    past = torch.zeros(1, device="cuda:0", dtype=torch.int32)
    actual = attn_dispatch(q, k, v, cache=cache, cache_idx=0, block_table=bt,
                           cache_seqlens=past, causal=True, sm_scale=0.125, max_kv_len=0)
    dk, dv = cache.get_kv(past + 5, bt)
    expected = F.scaled_dot_product_attention(q.transpose(1, 2).float(),
        dk[2:3, :5].repeat_interleave(2, 2).transpose(1, 2).float(),
        dv[2:3, :5].repeat_interleave(2, 2).transpose(1, 2).float(), is_causal=True, scale=0.125).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=3e-3)
