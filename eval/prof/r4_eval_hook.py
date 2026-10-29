#!/usr/bin/env python3
"""Eval-only hook: run supported prefill attention through r4-int8/fp8.

Monkeypatches exllamav3.modules.attn.attn_dispatch (same technique as
capture_qkv_27b.py). NOT a dispatch change: the repo path is untouched and
default-off; this exists so KLD/needle evals can measure the Round 4 scheme
on real model tensors.

Firing rules (mirror the prototype limits; everything else → orig fp16):
- fp16 q, prefill (q_len > 16), dim <= 512 and a power of two,
- no varlen / window / softcap / sinks / spans,
- cache-less bulk: full K/V are the call's k/v (KLD dump path),
- fp16 paged cache: orig runs first (fp16 compute + cache update, result
  discarded), then the full window is gathered from the updated cache and
  run through r4 (needle chunked-prefill path). Decode (q_len <= 16) and
  non-fp16 caches → orig.

Usage:
    from r4_eval_hook import install_r4_hook
    info = install_r4_hook(mode="int8")  # returns counter dict
    ... run model ...
    print(info)  # {"calls": N, "layers": {n_q_heads: count}}
"""
import sys

sys.path.insert(0, "/src")

import torch
import torch.nn.functional as F

INFO = {"calls": 0, "fallback": 0, "layers": {}}


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def install_r4_hook(mode="int8", block_m=128, block_n=64):
    assert mode in ("fp8", "int8")
    import exllamav3.modules.attn as attn_mod
    from exllamav3.cache import CacheLayer, Cache
    from exllamav3.cache.fp16 import CacheLayer_fp16
    from exllamav3.modules.attention_fn.r4_fp8 import (
        r4_quantize, r4_to_paged, r4_prefill)

    orig = attn_mod.attn_dispatch

    def _run_r4(q, kf, vf, sm_scale):
        bsz, tq, hq, dim = q.shape
        tk = kf.shape[1]
        if sm_scale is None:
            sm_scale = dim ** (-0.5)
        quant = r4_quantize(q.contiguous(), kf.contiguous(), vf.contiguous(),
                            sm_scale, mode)
        # Pad the window to whole pages; padded rows are never attended
        # (cache_seqlens stays Tk).
        tp = ((tk + 255) // 256) * 256
        if tp != tk:
            pad = torch.zeros((bsz, tp - tk, kf.shape[2], dim),
                              dtype=torch.float16, device=q.device)
            kq = torch.cat([quant["k_q"], pad.to(quant["k_q"].dtype)], dim=1)
            vq = torch.cat([quant["v_q"], pad.to(quant["v_q"].dtype)], dim=1)
            quant = dict(quant, k_q=kq, v_q=vq)
        kc, vc, bt, cs = r4_to_paged(quant)
        return r4_prefill(quant["q_q"], kc, vc, bt, cs,
                          quant["row_log2e"], quant["v_scale"],
                          causal=True, mode=mode,
                          block_m=block_m, block_n=block_n)

    def _supported(q, k, window_size, softcap, sinks, cu_seqlens, non_causal_spans):
        if q.dtype != torch.float16 or not q.is_cuda:
            return False
        bsz, q_len, hq, dim = q.shape
        if q_len <= 16 or dim > 512 or not _is_pow2(dim):
            return False
        if cu_seqlens is not None or non_causal_spans is not None:
            return False
        if sinks is not None or (softcap or 0.0) != 0.0:
            return False
        if window_size is not None and window_size != -1:
            return False
        if k is None or k.dtype != torch.float16:
            return False
        return True

    def patched(q, k, v, cache=None, cache_idx=None, cache_instance=None,
                causal=True, sm_scale=None, cu_seqlens=None, max_seqlen=None,
                window_size=None, softcap=0.0, block_table=None,
                cache_seqlens=None, non_causal_spans=None, sinks=None,
                dispatch_cache=None, max_kv_len=None):
        kw = dict(q=q, k=k, v=v, cache=cache, cache_idx=cache_idx,
                  cache_instance=cache_instance, causal=causal,
                  sm_scale=sm_scale, cu_seqlens=cu_seqlens,
                  max_seqlen=max_seqlen, window_size=window_size,
                  softcap=softcap, block_table=block_table,
                  cache_seqlens=cache_seqlens,
                  non_causal_spans=non_causal_spans, sinks=sinks,
                  dispatch_cache=dispatch_cache, max_kv_len=max_kv_len)
        if not _supported(q, k, window_size, softcap, sinks, cu_seqlens,
                          non_causal_spans):
            return orig(**kw)
        bsz, q_len, hq, dim = q.shape
        if cache is None:
            # Bulk (KLD dump): full window is k/v.
            if (v is None or v.dtype != torch.float16 or k.ndim != 4
                    or k.shape[0] != bsz or k.shape[1] < q_len
                    or v.shape != k.shape):
                return orig(**kw)
            INFO["calls"] += 1
            INFO["layers"][hq] = INFO["layers"].get(hq, 0) + 1
            try:
                return _run_r4(q, k, v, sm_scale)
            except Exception as ex:
                INFO["fallback"] += 1
                print(f"[r4-hook] bulk fallback: {type(ex).__name__} {str(ex)[:120]}", flush=True)
                return orig(**kw)
        # Cached (needle chunked prefill): orig first (cache update), then
        # gather the full window from the updated fp16 cache.
        if block_table is None or cache_seqlens is None:
            return orig(**kw)
        try:
            layer = cache if isinstance(cache, CacheLayer) else cache.layers[cache_idx, cache_instance or 0]
        except Exception:
            return orig(**kw)
        if not isinstance(layer, CacheLayer_fp16):
            return orig(**kw)
        out_fp16 = orig(**kw)  # cache now holds past + new rows
        try:
            t_new = k.shape[1]
            hkv = k.shape[2]
            pages = layer.k
            page_size = pages.shape[1]
            kf_b, vf_b = [], []
            for b in range(bsz):
                total = int(cache_seqlens[b].item()) + t_new
                pos = torch.arange(total, device=q.device)
                pg = block_table[b][pos // page_size]
                off = pos % page_size
                kf_b.append(layer.k[pg, off])
                vf_b.append(layer.v[pg, off])
            kf = torch.stack(kf_b, dim=0)
            vf = torch.stack(vf_b, dim=0)
        except Exception:
            return out_fp16
        INFO["calls"] += 1
        INFO["layers"][hq] = INFO["layers"].get(hq, 0) + 1
        try:
            return _run_r4(q, kf, vf, sm_scale)
        except Exception as ex:
            INFO["fallback"] += 1
            print(f"[r4-hook] cached fallback: {type(ex).__name__} {str(ex)[:120]}", flush=True)
            return out_fp16

    attn_mod.attn_dispatch = patched
    return INFO
