"""Paged LSA index keys, shared by unquantized and quantized KV layers."""
from __future__ import annotations

from collections import OrderedDict
import itertools
import math
import torch

from ..constants import PAGE_SIZE
from .fp16 import CacheLayer_fp16
from .quant import CacheLayer_quant

_versions = itertools.count(1)


class LSAPlanes:
    def _init_lsa(self, attention, max_num_tokens):
        indexer = attention.lsa_indexer
        self.index_dim = indexer.index_dim
        self.lsa_settings = indexer.settings
        self.index_shape = (max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.index_dim)
        self.block_capacity = max(1, math.ceil(max_num_tokens / self.lsa_settings.window_size))
        self.index_keys = self.page_versions = self.centroids = self.members = None
        self.bucket_cache = OrderedDict()

    def alloc(self, device):
        super().alloc(device)
        self.index_keys = torch.zeros(self.index_shape, dtype=torch.float32, device=device)
        # cache_rotate copies uint4 lanes, so every persisted page needs 16 bytes.
        self.page_versions = torch.zeros((self.index_shape[0], 2), dtype=torch.int64, device=device)
        s = self.lsa_settings
        n = s.window_size // s.bucket_size
        self.centroids = torch.empty((self.block_capacity, n, self.index_dim), dtype=torch.float32, device=device)
        self.members = torch.empty((self.block_capacity, n, s.bucket_cap), dtype=torch.int32, device=device)
        self.bucket_cache.clear()

    def free(self):
        super().free()
        self.index_keys = self.page_versions = self.centroids = self.members = None
        self.bucket_cache.clear()

    def get_tensors(self):
        # Versions travel with index keys through CPU paging and defragmentation.
        # Derived buckets are an LRU workspace, never part of a persisted page.
        return super().get_tensors() + [self.index_keys, self.page_versions]

    def copy_page(self, source, from_page, to_page, num_tokens):
        super().copy_page(source, from_page, to_page, num_tokens)
        self.index_keys[to_page, :num_tokens].copy_(source.index_keys[from_page, :num_tokens], non_blocking=True)
        # A partial copy may leave a different suffix; give the destination a new version.
        self.page_versions[to_page] = next(_versions)

    def storage_size(self):
        return super().storage_size() + self.index_storage_size()

    def index_storage_size(self):
        s = self.lsa_settings
        derived = self.block_capacity * (s.window_size // s.bucket_size) * (self.index_dim + s.bucket_cap) * 4
        return math.prod(self.index_shape) * 4 + self.index_shape[0] * 16 + derived

    def append_index(self, keys, block_table, past):
        b, t, _ = keys.shape
        positions = past.to(keys.device)[:, None] + torch.arange(t, device=keys.device)[None, :]
        pages = block_table.gather(1, positions // PAGE_SIZE).long()
        slots = pages * PAGE_SIZE + positions % PAGE_SIZE
        self.index_keys.flatten(0, 1).index_copy_(0, slots.flatten(), keys.flatten(0, 1))
        self.page_versions[pages.flatten()] = next(_versions)

    def buckets(self, indexer, block_table, closed):
        s = self.lsa_settings
        device = self.index_keys.device
        if not closed:
            return (self.index_keys.new_empty((0, self.index_dim)),
                    self.members.new_empty((0, s.bucket_cap)))
        end = closed * s.window_size
        page_count = math.ceil(end / PAGE_SIZE)
        pages = block_table[:page_count].long()
        versions = self.page_versions[pages, 0].cpu().tolist()
        page_ids = pages.cpu().tolist()
        signatures = []
        missing = []
        for block in range(closed):
            first = block * s.window_size // PAGE_SIZE
            last = math.ceil((block + 1) * s.window_size / PAGE_SIZE)
            signature = (block * s.window_size % PAGE_SIZE,
                         tuple(zip(page_ids[first:last], versions[first:last])))
            signatures.append(signature)
            if signature not in self.bucket_cache:
                missing.append(block)
        if missing:
            positions = (torch.tensor(missing, device=device)[:, None] * s.window_size
                         + torch.arange(s.window_size, device=device)[None, :])
            slots = block_table[positions // PAGE_SIZE].long() * PAGE_SIZE + positions % PAGE_SIZE
            keys = self.index_keys.flatten(0, 1)[slots]
            centroids, members = indexer.cluster(keys)
            # Protect all resident blocks of this request until its read has been assembled.
            for signature in signatures:
                if signature in self.bucket_cache:
                    self.bucket_cache.move_to_end(signature)
            for row, block in enumerate(missing):
                if len(self.bucket_cache) < self.block_capacity:
                    slot = len(self.bucket_cache)
                else:
                    _, slot = self.bucket_cache.popitem(last=False)
                self.centroids[slot].copy_(centroids[row])
                self.members[slot].copy_(members[row])
                self.bucket_cache[signatures[block]] = slot
        slots = []
        for signature in signatures:
            slot = self.bucket_cache[signature]
            slots.append(slot)
            self.bucket_cache.move_to_end(signature)
        slots = torch.tensor(slots, device=device)
        centroids = self.centroids[slots].flatten(0, 1)
        members = self.members[slots]
        offsets = torch.arange(closed, device=device)[:, None, None] * s.window_size
        members = torch.where(members >= 0, members + offsets, -1).int().flatten(0, 1)
        return centroids, members


class CacheLayer_lsa(LSAPlanes, CacheLayer_fp16):
    def __init__(self, config, attention, cache_id, max_num_tokens):
        super().__init__(config, attention, cache_id, max_num_tokens)
        self._init_lsa(attention, max_num_tokens)

    def tp_export(self, plan):
        return {"cls": CacheLayer_lsa, "args": {"cache_id": self.cache_id, "max_num_tokens": self.max_num_tokens}}


class CacheLayer_lsa_quant(LSAPlanes, CacheLayer_quant):
    def __init__(self, config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a=0.0):
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a)
        self._init_lsa(attention, max_num_tokens)

    def tp_export(self, plan):
        return {"cls": CacheLayer_lsa_quant, "args": {
            "cache_id": self.cache_id, "max_num_tokens": self.max_num_tokens,
            "k_bits": self.k_bits, "v_bits": self.v_bits, "compand_a": self.compand_a}}
