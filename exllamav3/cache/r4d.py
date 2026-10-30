"""Native libr4d KV cache layer (Stage A bf16, opt-in).

Stores K/V directly in r4d's paged HND bf16 layout at write time, so the
prefill/decode kernels read the cache with no per-call gather or repack:
EXL3 page p (256 tokens) occupies 16 consecutive r4d 16-blocks
``[p*16, p*16+15]``, with K in ``[..., :dim]`` and V in ``[..., dim:]``
of each ``(hkv, 16, 2*dim)`` block. Enabled per model build by passing
``layer_type=CacheLayer_r4d`` (see ``model_init.py``: ``EXL3_R4D_ATTN=bf16``
implies it when the library is present); only the 27B full-attention
geometry (head_dim 256, 6 q/kv, causal, gfx1201) is served, everything else
fails loudly at first use rather than silently falling back to a layout
this layer cannot provide.

License boundary: this file contains only our own adapter code. It must
never vendor libr4d or vllm-mxfp4 sources, patches, or binaries. The
``r4d.so`` is built on the user's machine by ``rocm_tools/build_r4d.sh``
and located via ``EXL3_R4D_LIB``.
"""
from __future__ import annotations
from typing_extensions import override
import torch
import numpy as np
from ..constants import PAGE_SIZE
from .cache import CacheLayer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention
    from ..model import Model, Config

_R4D_BLOCK = 16


class CacheLayer_r4d(CacheLayer):

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        self.hkv = attention.num_kv_heads if attention else 0
        self.dim = attention.head_dim if attention else 0
        self.num_pages = max_num_tokens // PAGE_SIZE if attention else 0
        # One EXL3 page = 16 consecutive r4d 16-blocks: block ids are
        # physical_page * 16 + (pos_in_page // 16), sub-row pos_in_page % 16.
        self.shape = (
            (self.num_pages * _R4D_BLOCK, self.hkv, _R4D_BLOCK, 2 * self.dim)
            if attention else None
        )
        self.kv = None
        self.device = None

    @override
    def alloc(self, device: torch.device):
        self.device = device
        # Zero-init: padding (unaligned tails) is never read (kernels bound
        # reads by seqused), but it must not hold NaNs.
        self.kv = torch.zeros(self.shape, dtype=torch.bfloat16, device=device) \
            if self.shape else None

    @override
    def free(self):
        self.device = None
        self.kv = None

    def write_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ):
        """Pack new fp16 K/V rows into the native layout at positions
        cache_seqlens..+length. All device-side (no host sync): positions,
        page/block/sub indices and the bf16 scatter stay on GPU."""
        if length <= 0:
            return
        with torch.no_grad():
            dev = self.kv.device
            bsz, n = k.shape[0], k.shape[1]
            assert n == length and k.shape[2] == self.hkv and v.shape == k.shape, \
                f"r4d write shape mismatch: k {tuple(k.shape)} v {tuple(v.shape)}"
            pos = cache_seqlens.reshape(bsz, 1).to(torch.int64) + torch.arange(
                n, device=dev, dtype=torch.int64).view(1, n)
            logical = pos // PAGE_SIZE
            phys = block_table.to(torch.int64).gather(1, logical)
            blk16 = (phys * _R4D_BLOCK + (pos % PAGE_SIZE) // _R4D_BLOCK).reshape(-1)
            sub = ((pos % PAGE_SIZE) % _R4D_BLOCK).reshape(-1)
            kb = k.reshape(bsz * n, self.hkv, self.dim).bfloat16()
            vb = v.reshape(bsz * n, self.hkv, self.dim).bfloat16()
            # In-place scatters into the packed halves (views share storage
            # with self.kv; no temp copies, no host sync).
            self.kv[..., :self.dim][blk16, :, sub] = kb
            self.kv[..., self.dim:][blk16, :, sub] = vb

    @override
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        self.write_kv(cache_seqlens, block_table, k, v, length)

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1):
        raise RuntimeError(
            "CacheLayer_r4d has no fp16 cache to return: serve these layers "
            "through the r4d dispatch path (EXL3_R4D_ATTN=bf16).")

    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        # No-op: every write goes through write_kv before the attention call.
        pass

    @override
    def copy_page(self, source: CacheLayer_r4d, from_page: int, to_page: int, num_tokens: int):
        assert isinstance(source, CacheLayer_r4d) and source.kv is not None
        assert self.kv is not None and source.shape == self.shape
        # Move whole 16-token blocks covering [0, num_tokens); rows past
        # num_tokens travel as stale padding, which is never read (all reads
        # are seqused-bounded) and gets overwritten by the next write_kv at
        # those exact positions.
        nblocks = (num_tokens + _R4D_BLOCK - 1) // _R4D_BLOCK
        self.kv[to_page * _R4D_BLOCK:to_page * _R4D_BLOCK + nblocks].copy_(
            source.kv[from_page * _R4D_BLOCK:from_page * _R4D_BLOCK + nblocks],
            non_blocking=True)

    @override
    def get_tensors(self):
        return [self.kv]

    @override
    def storage_size(self):
        return int(np.prod(self.shape)) * torch.bfloat16.itemsize

    @override
    def overhead_size(self):
        return 0

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_r4d,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }
