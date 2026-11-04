"""LSA numerical and cache lifecycle checks; run on either ROCm GPU family."""
from types import SimpleNamespace
import torch
import torch.nn.functional as F
import pytest

from exllamav3.modules.lsa import LSAIndexer, LSASettings
from exllamav3.cache.lsa import CacheLayer_lsa, CacheLayer_lsa_quant


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
@pytest.mark.parametrize("quant", [False, True])
def test_paged_sparse_attention_matches_independent_gather(quant):
    dev = torch.device("cuda:0")
    idx = make_indexer(dev)
    a = fake_attention(idx)
    cls = CacheLayer_lsa_quant if quant else CacheLayer_lsa
    cache = cls(None, a, 1, 1024, **({"k_bits": 6, "v_bits": 4} if quant else {}))
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
def test_index_planes_follow_native_page_rotation():
    from exllamav3.ext import exllamav3_ext as ext
    idx = make_indexer("cuda:0")
    cache = CacheLayer_lsa(None, fake_attention(idx), 1, 1024)
    cache.alloc(torch.device("cuda:0"))
    bt = torch.tensor([[2, 0, 3, 1]], device="cuda:0", dtype=torch.int32)
    keys = F.normalize(torch.randn(1, 192, 16, device="cuda:0"), dim=-1)
    cache.append_index(keys, bt, torch.tensor([0]))
    before = cache.buckets(idx, bt[0], 2)
    order = torch.tensor([-1, 2, 2, 1, 1, -1], device="cuda:0", dtype=torch.int32)
    for tensor in cache.get_tensors():
        ext.cache_rotate(tensor, order, torch.empty_like(tensor[0]))
    rotated = cache.buckets(idx, torch.tensor([1, 0, 3, 2], device="cuda:0", dtype=torch.int32), 2)
    torch.testing.assert_close(rotated[0], before[0])
    torch.testing.assert_close(rotated[1], before[1])
