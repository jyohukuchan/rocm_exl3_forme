"""Distilled LSA for grouped-query attention, independent of the KV storage dtype.

The index uses the position-free, QK-normalized heads, as in the official Qwen
SGLang backend. Only the small projected keys persist alongside paged KV; the
original keys and values continue to use the caller's cache quantization.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from contextlib import nullcontext
from pathlib import Path
import re
import torch
import torch.nn.functional as F

# Bound stable-sort scratch by batching independent query rows, not by
# restricting the history or changing the selected key budget.
_FLAT_SORT_MAX_ELEMENTS = 1 << 24


def mp_lsa_stats(context):
    device = context["device"]
    records = []
    for root in context["modules"]:
        if root is None:
            continue
        for module in root:
            idx = getattr(module, "lsa_indexer", None)
            if idx is None:
                continue
            records.append(dict(key=module.key, kv_heads=module.num_kv_heads,
                index_owner=idx.index_owner, owns_index=idx.owns_index,
                index_bytes=sum(cl.index_storage_size() for cl in module.cache_layers),
                selection_graph_replays=sum(g.replays for g in getattr(idx, "_tp_read_graphs", {}).values()),
                attention_graph_replays=sum(g.replays for g in getattr(idx, "_tp_attention_graphs", {}).values())))
    return dict(device=device, graph_pool=context.get("lsa_graph_pool"),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                allocated_gib=torch.cuda.memory_allocated(device)/2**30,
                reserved_gib=torch.cuda.memory_reserved(device)/2**30,
                free_gib=torch.cuda.mem_get_info(device)[0]/2**30, layers=records)


def _input_device(function):
    @wraps(function)
    def guarded(self, tensor, *args, **kwargs):
        if tensor.is_cuda:
            with torch.cuda.device(tensor.device):
                return function(self, tensor, *args, **kwargs)
        return function(self, tensor, *args, **kwargs)
    return guarded


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
    top_level: int = 0
    node_bucket_size: int | list[int] | None = None
    node_cluster_block_size: int | list[int] | None = None
    beam_width: int | list[int] | None = None
    start_nodes: int | None = None
    node_centroid_precision: str = "bf16"
    query_heads: str = "all"
    decode: str = "flat"

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
        if not isinstance(self.top_level, int) or isinstance(self.top_level, bool) or self.top_level < 0:
            raise ValueError("LSA top_level must be a nonnegative integer")
        if self.query_heads not in ("all", "rank"):
            raise ValueError("LSA query_heads must be all or rank")
        if self.decode not in ("flat", "descend"):
            raise ValueError("LSA decode must be flat or descend")
        if self.node_centroid_precision not in ("bf16", "fp8"):
            raise ValueError("LSA node_centroid_precision must be bf16 or fp8")
        if self.start_nodes is not None and self.start_nodes < 1:
            raise ValueError("LSA start_nodes must be positive")
        for level in range(1, self.top_level + 1):
            size, block, beam = (self.at(name, level) for name in (
                "node_bucket_size", "node_cluster_block_size", "beam_width"))
            if size < 2 or size & (size - 1) or block % size or beam < 1:
                raise ValueError("LSA tree requires a power-of-two node size, divisible cluster blocks and positive beams")

    def at(self, name, level):
        value = getattr(self, name)
        if value is None:
            value = {"node_bucket_size": 4, "node_cluster_block_size": 4096, "beam_width": 2048}[name]
        if isinstance(value, (list, tuple)):
            if len(value) != self.top_level:
                raise ValueError(f"LSA {name} needs one value or {self.top_level} values")
            value = value[level - 1]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"LSA {name} must be a positive integer")
        return value


def settings_from_config(config):
    body = getattr(config, "lsa", None)
    if not body:
        return None
    body = dict(body)
    # These describe the published Qwen maps; they are accepted for config portability.
    for key, expected in (("input", "key_and_query"),):
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
        self.index_owner = None
        self.owns_index = True
        self.tp_graph_pool = None
        self.tp_graph_stream = None
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
        self.decode_graph = None
        self._tp_read_tables = {}
        self._tp_read_graphs = {}
        self._tp_attention_graphs = {}

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
            key = joined[1]
            if self.settings.query_heads == "all":
                query = joined[0]
        return F.normalize(query, dim=-1, eps=1e-12), F.normalize(key, dim=-1, eps=1e-12)

    def empty_project(self, x, params):
        """Headless TP ranks participate in the same index projection collective."""
        cached = params.get("attn_mode") == "flash_attn"
        active = cached or (params.get("causal", True) and params.get("cu_seqlens") is None and not self.can_read_dense(x.shape[1]))
        if self.tp_mode and active:
            params["backend"].all_reduce(torch.zeros((2, *x.shape[:2], self.index_dim),
                                                     dtype=torch.float32, device=x.device))
            if self.index_owner is not None:
                from ..util.tensor import get_for_device
                past = get_for_device(params, "cache_seqlens", "cpu") if cached else None
                end = int(past.max()) + x.shape[1] if cached else x.shape[1]
                if (params.get("causal", True) and params.get("non_causal_spans") is None
                        and not self.can_read_dense(end)):
                    for _ in range(x.shape[0]):
                        self.share_indices(None, x.shape[1], x.device, params)

    def share_indices(self, indices, rows, device, params):
        if self.index_owner is None:
            return indices
        width = self.settings.top_k + self.settings.window_size + self.settings.sink_size
        # Native broadcast transfers 16-byte lanes. Padding is transport only.
        padded = (width + 3) // 4 * 4
        if indices is None:
            transfer = torch.empty((rows, padded), dtype=torch.int32, device=device)
        elif padded != width:
            transfer = F.pad(indices, (0, padded - width), value=-1)
        else:
            transfer = indices
        params["backend"].broadcast(transfer, self.index_owner)
        return transfer[:, :width].contiguous()

    def can_read_dense(self, end):
        """A dense read is exact while every closed bucket and historical key fits."""
        s = self.settings
        if s.top_level:
            import math
            block = s.at("node_cluster_block_size", 1)
            page = math.lcm(8192, s.window_size)
            first = math.ceil(block * max(1, 4096 // block) * s.bucket_size / page) * page
            if end > first:
                return False  # A beam can prune even when the flat budget covers the context.
        buckets = max(0, (end - 1) // s.window_size) * (s.window_size // s.bucket_size)
        return end <= s.top_k + s.window_size + s.sink_size and buckets <= s.top_p

    @_input_device
    def cluster(self, keys, return_sums=False):
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
        return (centroids, members, sums) if return_sums else (centroids, members)

    @_input_device
    def select(self, queries, positions, centroids, members, tree=None):
        s = self.settings
        if tree is None and len(queries) > 1 and len(queries) * len(centroids) > _FLAT_SORT_MAX_ELEMENTS:
            rows = max(1, _FLAT_SORT_MAX_ELEMENTS // max(1, len(centroids)))
            rows = 1 << (rows.bit_length() - 1)
            return torch.cat([self.select(queries[first:first + rows], positions[first:first + rows], centroids, members)
                              for first in range(0, len(queries), rows)])
        if tree is not None:
            picks = tree.pick(queries, positions, centroids)
        else:
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
        with torch.cuda.device(q.device) if q.is_cuda else nullcontext():
            return self._attend(attention, q, k, v, params, cache_layer, block_table, past, cache_seqlens)

    def _attend(self, attention, q, k, v, params, cache_layer=None, block_table=None, past=None, cache_seqlens=None):
        from .attention_fn.qsa_triton import qsa_sparse_attend_rows
        from ..constants import PAGE_SIZE
        query, key = self.project(q, k, params)
        b, t = q.shape[:2]
        if cache_layer is not None:
            cache_layer.append_index(key, block_table, past, mark_versions=params.get("_lsa_graph_state") is None)
            cache_layer.update_kv_direct(cache_seqlens, block_table, k, v, t)
        outputs = []
        for batch in range(b):
            tree = None
            graph_state = params.get("_lsa_graph_state")
            start = past[batch] if graph_state is not None else int(past[batch]) if past is not None else 0
            positions = (past[batch:batch + 1] if graph_state is not None else
                         torch.arange(start, start + t, dtype=torch.int32, device=q.device))
            if not self.owns_index:
                indices = self.share_indices(None, t, q.device, params)
            elif graph_state is not None:
                positions = past[batch:batch + 1]
                centroids, members = graph_state.buckets[batch]
            elif cache_layer is None:
                closed = (start + t - 1) // self.settings.window_size
                keys = key[batch, :closed * self.settings.window_size].reshape(closed, self.settings.window_size, -1)
                if closed:
                    clustered = self.cluster(keys, return_sums=bool(self.settings.top_level))
                    centroids, members = clustered[:2]
                    if self.settings.top_level:
                        from .lsa_tree import TreeStore
                        sums = clustered[2].flatten(0, 1)
                        store = TreeStore(self.settings, len(sums), self.index_dim, q.device)
                        tree = store.view(sums, list(range(len(sums))))
                    offsets = torch.arange(closed, device=q.device)[:, None, None] * self.settings.window_size
                    members = torch.where(members >= 0, members + offsets, -1)
                    centroids, members = centroids.flatten(0, 1), members.flatten(0, 1)
                else:
                    centroids = q.new_empty((0, self.index_dim), dtype=torch.float32)
                    members = torch.empty((0, self.settings.bucket_cap), dtype=torch.int32, device=q.device)
            else:
                closed = (start + t - 1) // self.settings.window_size
                use_tree = bool(self.settings.top_level) and (t != 1 or self.settings.decode == "descend")
                bucketed = cache_layer.buckets(self, block_table[batch], closed, with_tree=use_tree)
                centroids, members = bucketed[:2]
                tree = bucketed[2] if use_tree else None
            if self.owns_index:
                if self.index_owner is not None and tree is None:
                    from .attention_fn.lsa_decode import tp_select
                    indices = tp_select(self, query[batch], positions, centroids, members, batch)
                else:
                    indices = self.select(query[batch], positions, centroids, members, tree)
                indices = self.share_indices(indices, t, q.device, params)
            qc = None
            bt = None
            if cache_layer is None:
                keys, values = k[batch].contiguous(), v[batch].contiguous()
            else:
                bt = block_table[batch:batch + 1].expand(t, -1).contiguous()
                from ..cache.quant import CacheLayer_quant
                if isinstance(cache_layer, CacheLayer_quant) and not cache_layer.compand_a:
                    keys, sk, values, sv, kb, vb = cache_layer.get_qkv()
                    qc = (sk.flatten(0, 1), sv.flatten(0, 1), kb, vb)
                else:
                    keys, values = cache_layer.get_kv(cache_seqlens + t, block_table)
                keys, values = keys.flatten(0, 1), values.flatten(0, 1)
            if self.index_owner is not None and bt is not None and (qc is not None or keys.dtype == torch.float16):
                from .attention_fn.lsa_decode import tp_attend
                output = tp_attend(self, cache_layer, q[batch].contiguous(), keys, values, indices,
                    attention.sm_scale, bt, qc, attention.num_kv_heads, batch)
            else:
                output = qsa_sparse_attend_rows(q[batch].contiguous(), keys, values, indices,
                    attention.sm_scale, bt, PAGE_SIZE if bt is not None else 0, qc, attention.num_kv_heads)
            outputs.append(output)
        return torch.stack(outputs)
