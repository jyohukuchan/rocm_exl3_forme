"""Distilled LSA for grouped-query attention, independent of the KV storage dtype.

The index uses the position-free, QK-normalized heads, as in the official Qwen
SGLang backend. Only the small projected keys persist alongside paged KV; the
original keys and values continue to use the caller's cache quantization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LSASettings:
    projections: str
    window_size: int = 512
    sink_size: int = 4
    top_k: int = 4096
    top_p: int = 1024
    pool_size: int | None = 4
    bucket_size: int = 8
    bucket_cap: int = 16
    index_dim: int | None = None

    def __post_init__(self):
        for name in ("window_size", "top_k", "top_p", "pool_size", "bucket_size", "bucket_cap"):
            value = getattr(self, name)
            if name == "pool_size" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"LSA {name} must be a positive integer")
        if not isinstance(self.sink_size, int) or isinstance(self.sink_size, bool) or self.sink_size < 0:
            raise ValueError("LSA sink_size must be a nonnegative integer")
        if self.bucket_size % (self.pool_size or 1) or self.window_size % self.bucket_size:
            raise ValueError("LSA pool_size must divide bucket_size, and bucket_size must divide window_size")
        if self.bucket_cap < 2 * self.bucket_size:
            raise ValueError("LSA bucket_cap must be at least twice bucket_size for Ward to reach its bucket count")


def settings_from_config(config):
    body = getattr(config, "lsa", None)
    if not body:
        return None
    body = dict(body)
    # These describe the published Qwen maps; they are accepted for config portability.
    for key, expected in (("input", "key_and_query"), ("query_heads", "all"), ("top_level", 0)):
        if key in body and body.pop(key) != expected:
            raise ValueError(f"EXL3 LSA currently implements {key}={expected!r}")
    path = Path(body["projections"]).expanduser()
    if not path.is_absolute():
        path = Path(config.directory) / path
    body["projections"] = str(path)
    return LSASettings(**body)


class LSAIndexer:
    def __init__(self, settings, query_map, key_map, head_dim, kv_heads, rotary_dim, first_head=0):
        self.settings = settings
        self.query_map_cpu = query_map.float().contiguous().cpu()
        self.key_map_cpu = key_map.float().contiguous().cpu()
        if self.query_map_cpu.ndim != 2 or self.query_map_cpu.shape != self.key_map_cpu.shape:
            raise ValueError("LSA maps must be matching [input_dim, index_dim] matrices")
        if self.key_map_cpu.shape[0] != kv_heads * (head_dim - rotary_dim):
            raise ValueError("LSA projection input does not match the position-free KV heads")
        self.index_dim = key_map.shape[1]
        if settings.index_dim is not None and settings.index_dim != self.index_dim:
            raise ValueError("LSA index_dim does not match the projection file")
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        self.rotary_dim = rotary_dim
        self.first_head = first_head
        self.tp_mode = False
        self.query_map = self.key_map = None

    @classmethod
    def from_attention(cls, attention):
        if attention.key.startswith("mtp."):
            # Released maps describe target decoder layers, not the independent MTP head.
            return None
        settings = settings_from_config(attention.config)
        if settings is None:
            return None
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", attention.key)
        if match is None:
            return None  # e.g. vision attention, which has no language-model layer id
        layer = int(match[1])
        config = attention.config
        if getattr(config, "_lsa_weights_path", None) != settings.projections:
            config._lsa_weights = torch.load(settings.projections, map_location="cpu", weights_only=True)
            config._lsa_weights_path = settings.projections
        weights = config._lsa_weights
        q_key, k_key = f"{layer}.W_Q_idx", f"{layer}.W_K_idx"
        if q_key not in weights or k_key not in weights:
            raise ValueError(f"LSA projections lack attention layer {layer}: {settings.projections}")
        rs = attention.rope_settings
        rotary_dim = (rs.rotary_dim or int(rs.head_dim * rs.partial_rotary_factor)) if rs else 0
        return cls(settings, weights[q_key], weights[k_key], attention.head_dim,
                   attention.num_kv_heads, rotary_dim)

    def load(self, device):
        self.query_map = self.query_map_cpu.to(device)
        self.key_map = self.key_map_cpu.to(device)

    def unload(self):
        self.query_map = self.key_map = None

    def storage_size(self):
        return self.query_map_cpu.numel() * 8

    def tp_export(self):
        return dict(settings=self.settings, query_map=self.query_map_cpu, key_map=self.key_map_cpu,
                    head_dim=self.head_dim, kv_heads=self.kv_heads, rotary_dim=self.rotary_dim)

    def project(self, q, k, params):
        """Read normalized, rotated Q/K; the unrotated suffix survives RoPE unchanged."""
        if self.query_map is None or self.query_map.device != q.device:
            self.load(q.device)
        b, t, qh, _ = q.shape
        kh = k.shape[2]
        width = self.head_dim - self.rotary_dim
        query = q[..., self.rotary_dim:].float().reshape(b, t, kh, qh // kh, width).sum(3)
        key = k[..., self.rotary_dim:].float()
        start, end = self.first_head * width, (self.first_head + kh) * width
        query = query.flatten(2) @ self.query_map[start:end]
        key = key.flatten(2) @ self.key_map[start:end]
        if self.tp_mode:
            # Linear projection precedes reduction; normalization must follow it.
            joined = torch.stack((query, key)).contiguous()
            params["backend"].all_reduce(joined)
            query, key = joined.unbind(0)
        return F.normalize(query, dim=-1, eps=1e-12), F.normalize(key, dim=-1, eps=1e-12)

    def empty_project(self, x, params):
        """Headless TP ranks participate in the same index projection collective."""
        cached = params.get("attn_mode") == "flash_attn"
        active = cached or (params.get("causal", True) and params.get("cu_seqlens") is None and not self.can_read_dense(x.shape[1]))
        if self.tp_mode and active:
            params["backend"].all_reduce(torch.zeros((2, *x.shape[:2], self.index_dim),
                                                     dtype=torch.float32, device=x.device))

    def can_read_dense(self, end):
        """A dense read is exact while every closed bucket and historical key fits."""
        s = self.settings
        buckets = max(0, (end - 1) // s.window_size) * (s.window_size // s.bucket_size)
        return end <= s.top_k + s.window_size + s.sink_size and buckets <= s.top_p

    def cluster(self, keys):
        """Closed blocks [G, window, dim] -> normalized centroids and local member ids."""
        s = self.settings
        pool_size = s.pool_size or 1
        pools = keys.reshape(keys.shape[0], -1, pool_size, self.index_dim).sum(2)
        sizes = torch.full(pools.shape[:2], float(pool_size), device=keys.device)
        buckets = s.window_size // s.bucket_size
        n = pools.shape[1]
        if keys.is_cuda and 16 <= n <= 4096 and not n & (n - 1) and 16 <= self.index_dim <= 256 and not self.index_dim & (self.index_dim - 1):
            from ..vendor.lsa.ward import ward_merge, bucket_members_from_labels
            labels, reached = ward_merge(pools, sizes, buckets, s.bucket_cap, precision="ieee")
            if not bool(reached.all()):
                raise RuntimeError("LSA Ward clustering did not reach the requested bucket count")
        else:
            from ..vendor.lsa.ward_reference import ward_merge
            labels = ward_merge(pools, sizes, buckets, s.bucket_cap)
            from ..vendor.lsa.ward import bucket_members_from_labels
        pool_members = bucket_members_from_labels(labels, buckets, s.bucket_cap // pool_size)
        block_ids = torch.arange(keys.shape[0], device=keys.device)[:, None, None]
        gathered = pools[block_ids, pool_members.clamp_min(0).long()]
        sums = (gathered * (pool_members >= 0)[..., None]).sum(2)
        centroids = F.normalize(sums, dim=-1, eps=1e-30)
        members = bucket_members_from_labels(labels.repeat_interleave(pool_size, 1), buckets, s.bucket_cap)
        return centroids, members

    def select(self, queries, positions, centroids, members):
        s = self.settings
        scores = queries.float() @ centroids.float().T
        bucket_ids = torch.arange(centroids.shape[0], device=queries.device)
        visible = positions[:, None] // s.window_size * (s.window_size // s.bucket_size)
        scores.masked_fill_(bucket_ids[None, :] >= visible, -torch.inf)
        # Stable order is the reference design's tie break, including zero-vector buckets.
        picks = scores.argsort(dim=1, descending=True, stable=True)[:, :s.top_p]
        picks = torch.where(picks < visible, picks, -1).int().contiguous()
        if queries.is_cuda:
            from ..vendor.lsa.expansion import expand_in_order
            cap = members.shape[1]
            if cap & (cap - 1):
                members = F.pad(members, (0, (1 << (cap - 1).bit_length()) - cap), value=-1)
            return expand_in_order(picks, members, positions.int(), s.window_size, s.sink_size, s.top_k)[0]
        out = torch.full((len(positions), s.top_k + s.window_size + s.sink_size), -1,
                         device=queries.device, dtype=torch.int32)
        for row, position in enumerate(positions.tolist()):
            ids = members[picks[row][picks[row] >= 0].long()].flatten()
            start = max(0, position + 1 - s.window_size)
            ids = ids[(ids >= s.sink_size) & (ids < start)][:s.top_k]
            ids = torch.cat((ids, torch.arange(start, position + 1, device=ids.device),
                             torch.arange(min(s.sink_size, start), device=ids.device)))
            out[row, :len(ids)] = ids.int()
        return out

    def attend(self, attention, q, k, v, params, cache_layer=None, block_table=None, past=None, cache_seqlens=None):
        from .attention_fn.qsa_triton import qsa_sparse_attend_rows
        from ..constants import PAGE_SIZE
        query, key = self.project(q, k, params)
        b, t = q.shape[:2]
        if cache_layer is not None:
            cache_layer.append_index(key, block_table, past)
            cache_layer.update_kv_direct(cache_seqlens, block_table, k, v, t)
        outputs = []
        for batch in range(b):
            start = int(past[batch]) if past is not None else 0
            positions = torch.arange(start, start + t, dtype=torch.int32, device=q.device)
            closed = (start + t - 1) // self.settings.window_size
            if cache_layer is None:
                keys = key[batch, :closed * self.settings.window_size].reshape(closed, self.settings.window_size, -1)
                if closed:
                    centroids, members = self.cluster(keys)
                    offsets = torch.arange(closed, device=q.device)[:, None, None] * self.settings.window_size
                    members = torch.where(members >= 0, members + offsets, -1)
                    centroids, members = centroids.flatten(0, 1), members.flatten(0, 1)
                else:
                    centroids = q.new_empty((0, self.index_dim), dtype=torch.float32)
                    members = torch.empty((0, self.settings.bucket_cap), dtype=torch.int32, device=q.device)
            else:
                centroids, members = cache_layer.buckets(self, block_table[batch], closed)
            indices = self.select(query[batch], positions, centroids, members)
            qc = None
            bt = None
            if cache_layer is None:
                keys, values = k[batch].contiguous(), v[batch].contiguous()
            else:
                bt = block_table[batch:batch + 1].expand(t, -1).contiguous()
                if hasattr(cache_layer, "get_qkv") and not cache_layer.compand_a:
                    keys, sk, values, sv, kb, vb = cache_layer.get_qkv()
                    qc = (sk.flatten(0, 1), sv.flatten(0, 1), kb, vb)
                else:
                    keys, values = cache_layer.get_kv(cache_seqlens + t, block_table)
                keys, values = keys.flatten(0, 1), values.flatten(0, 1)
            outputs.append(qsa_sparse_attend_rows(q[batch].contiguous(), keys, values, indices,
                attention.sm_scale, bt, PAGE_SIZE if bt is not None else 0, qc, attention.num_kv_heads))
        return torch.stack(outputs)
