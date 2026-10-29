"""Opt-in libr4d prefill attention backend (Stage A/B).

Off by default. When enabled, paged-cache fp16 prefill calls at the 27B geometry
(head_dim 256, 6 q heads per kv head, causal, no window/sinks/softcap) run through
a locally built ``r4d.so`` instead of the Triton kernel. Everything else -- decode,
quantized caches, varlen, windowed/softcapped/sinked calls, other geometries --
declines silently and falls back to the existing kernels.

License boundary: this file contains only our own loader/adapter code. It must
never vendor libr4d or vllm-mxfp4 sources, patches, or binaries. The ``.so`` is
built on the user's machine by ``rocm_tools/build_r4d.sh`` (clone at pinned
commit + patch fetched from the upstream repo at build time) and located via
``EXL3_R4D_LIB`` (fallback: ``R4D_SO`` for bench compat). When the library is
absent or unloadable, every call declines and the default path is bit-for-bit
unchanged.

Stages (see doc/r4d_attention.md for numbers and gates):

- Stage A (``EXL3_R4D_ATTN=1`` or ``bf16``): bf16 KV + f16 legs. Max-rel ~3.7e-3
  vs the fp16 Triton path; short-context KLD ~1e-4, but the 32K needle gate
  fails (see doc/r4d_attention.md) -- long-context serving not recommended.
- Stage B (``EXL3_R4D_ATTN=fp8``): fp8 KV. The leg mode is the library's own
  ``R4D_ATTN_FP8`` switch (set before process start, read once per process by
  the library): mode 0 (f16 legs) or mode 2 (O_PV8) only. Modes 1/3 (O_QK8,
  max-rel ~0.5) stay off unless ``EXL3_R4D_ALLOW_QK8=1`` is set explicitly for
  KLD measurement; serving keeps them off and the call declines.

KV adapter: r4d wants block-16 K/V-packed (HND, bf16/fp8) caches while EXL3
keeps PAGE-256 fp16 caches, so each prefill call gathers the referenced window
(past + appended rows) and packs it into the r4d layout. A 256-page maps to 16
consecutive 16-blocks. At 128K this is ~1GB of traffic (a few ms) against a
~400ms kernel -- negligible, and a write-path cache variant can remove it later.
Q fp16->bf16 per call is cheap; the bf16 output is converted back to fp16.
Decode keeps the existing path (r4d decode kernel not wired: prefill focus).
"""

import ctypes
import os

import torch

from .common import AttnArgs

_R4D_BLOCK = 16
_FP8_MAX = 448.0


def r4d_lib_path() -> str | None:
    """Configured r4d.so location, or None when the backend is not installed."""
    for var in ("EXL3_R4D_LIB", "R4D_SO"):
        path = os.environ.get(var, "")
        if path.strip():
            return path.strip()
    return None


def r4d_mode() -> str:
    """Requested backend mode: 'off', 'bf16' (Stage A) or 'fp8' (Stage B)."""
    raw = os.environ.get("EXL3_R4D_ATTN", "0").strip().lower()
    if raw in ("1", "on", "true", "bf16", "bf16kv"):
        return "bf16"
    if raw in ("2", "fp8", "fp8kv"):
        return "fp8"
    return "off"


def _legs_mode() -> int:
    try:
        return int(os.environ.get("R4D_ATTN_FP8", "0") or "0")
    except ValueError:
        return 0


_lib = None
_lib_failed = False


class _R4DArgs(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p),
        ("kv", ctypes.c_void_p),
        ("block_table", ctypes.c_void_p),
        ("seqused_k", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("k_descale", ctypes.c_void_p),
        ("v_descale", ctypes.c_void_p),
        ("q_descale", ctypes.c_void_p),
        ("scratch", ctypes.c_void_p),
        ("num_seqs", ctypes.c_int32),
        ("q_len", ctypes.c_int32),
        ("q_heads", ctypes.c_int32),
        ("kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("block_size", ctypes.c_int32),
        ("max_blocks", ctypes.c_int32),
        ("kv_block_stride", ctypes.c_int64),
        ("kv_head_stride", ctypes.c_int64),
        ("scale", ctypes.c_float),
        ("splits", ctypes.c_int32),
        ("max_ctx", ctypes.c_int32),
    ]


def _load():
    """Load r4d.so once; return None (clean fallback) when unavailable."""
    global _lib, _lib_failed
    if _lib is not None:
        return _lib
    if _lib_failed:
        return None
    path = r4d_lib_path()
    if path is None:
        return None
    try:
        lib = ctypes.CDLL(path)
        for name in (
            "r4d_attn_prefill_h256_gqa6_bf16kv",
            "r4d_attn_prefill_h256_gqa6_fp8kv",
        ):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.POINTER(_R4DArgs), ctypes.c_void_p]
            fn.restype = ctypes.c_int
        _lib = lib
        return _lib
    except Exception:
        _lib_failed = True
        return None


def _arch_is_gfx1201(device: torch.device) -> bool:
    try:
        from .triton_paged import _gfx_arch

        return _gfx_arch(device).startswith("gfx12")
    except Exception:
        try:
            arch = str(torch.cuda.get_device_properties(device).gcnArchName or "")
        except Exception:
            return False
        return arch.split(":", 1)[0].strip().lower().startswith("gfx12")


def fn_r4d_attn_prefill(args: AttnArgs) -> torch.Tensor | None:
    """Opt-in r4d prefill over the paged fp16 cache (or cache-less K/V).

    Serves the 27B geometry (head_dim 256, GQA 6, causal, gfx1201); returns
    None when disabled or inapplicable so the default path is bit-for-bit
    unchanged.
    """
    if r4d_mode() == "off":
        return None
    if (
        args.is_varlen()
        or args.q_len <= 16
        or args.q.dtype != torch.float16
        or args.dim != 256
        or args.num_q_heads % args.num_kv_heads != 0
        or args.num_q_heads // args.num_kv_heads != 6
        or not args.causal
        or args.is_swa()
        or (args.softcap or 0.0) != 0.0
        or args.sinks is not None
        or args.non_causal_spans
        or args.q_cache is not None
        or args.i8f8_cache is not None
    ):
        return None
    use_cache = args.has_kv_cache()
    if use_cache and args.k_cache.dtype != torch.float16:
        return None
    if args.k is None or args.v is None:
        return None
    if args.k.shape[1] != args.q_len or args.v.shape[1] != args.q_len:
        return None
    if args.k.shape[0] != args.q.shape[0] or args.v.shape != args.k.shape:
        return None
    if not (args.q.is_cuda and args.k.is_cuda and args.q.is_contiguous()):
        return None
    if use_cache and not args.k_cache.is_cuda:
        return None
    if not _arch_is_gfx1201(args.q.device):
        return None

    mode = r4d_mode()
    legs = _legs_mode()
    if mode == "fp8" and legs in (1, 3):
        if os.environ.get("EXL3_R4D_ALLOW_QK8", "0") != "1":
            return None

    lib = _load()
    if lib is None:
        return None

    bsz, tq, hq, dim = args.q.shape
    hkv = args.num_kv_heads
    if use_cache:
        try:
            past_lens = [int(args.cache_seqlens[b].item()) for b in range(bsz)]
        except Exception:
            return None
    else:
        # Cache-less (k_new/v_new direct, or plain non-cached attention): the
        # whole attended sequence is the new K/V.
        past_lens = [0] * bsz
    total_lens = [past_lens[b] + tq for b in range(bsz)]
    if any(t <= 0 for t in total_lens):
        return None
    # The kernel masks q tails (live = qpos < q_len, early return before the
    # output store) and clamps K reads to seqused_k - 1, so any length is
    # servable: the adapter pads the packed KV buffer up to a whole number of
    # 16-blocks and passes the exact length in seqused_k. Padding blocks are
    # zeroed and never read.

    dev = args.q.device
    try:
        if use_cache:
            # Gather the full attended window per sequence through the block table.
            pages_per_seq = args.block_table.shape[1]
            flat_k = args.k_cache[args.block_table.long().view(-1)].view(
                bsz, pages_per_seq * args.k_cache.shape[1], hkv, dim
            )
            flat_v = args.v_cache[args.block_table.long().view(-1)].view(
                bsz, pages_per_seq * args.v_cache.shape[1], hkv, dim
            )
        nb_max = max(-(-t // _R4D_BLOCK) for t in total_lens)
        # Zero-init: padding (unaligned tails, shorter batch sequences) is never
        # read (seqused_k bounds the kernel), but it must not hold NaNs.
        k_full = torch.zeros((bsz, nb_max * _R4D_BLOCK, hkv, dim),
                             dtype=torch.float16, device=dev)
        v_full = torch.zeros_like(k_full)
        for b in range(bsz):
            p, t = past_lens[b], total_lens[b]
            if use_cache and p > 0:
                k_full[b, :p].copy_(flat_k[b, :p])
                v_full[b, :p].copy_(flat_v[b, :p])
            k_full[b, p:t].copy_(args.k[b])
            v_full[b, p:t].copy_(args.v[b])
        if mode == "bf16":
            kv = torch.empty((bsz * nb_max, hkv, _R4D_BLOCK, 2 * dim),
                             dtype=torch.bfloat16, device=dev)
            kf = k_full.bfloat16().view(bsz, -1, nb_max, _R4D_BLOCK, hkv, dim)
            vf = v_full.bfloat16().view(bsz, -1, nb_max, _R4D_BLOCK, hkv, dim)
            kv[..., :dim].copy_(
                kf.permute(0, 2, 4, 3, 1, 5).reshape(bsz * nb_max, hkv, _R4D_BLOCK, dim))
            kv[..., dim:].copy_(
                vf.permute(0, 2, 4, 3, 1, 5).reshape(bsz * nb_max, hkv, _R4D_BLOCK, dim))
            kd = vd = None
            entry = lib.r4d_attn_prefill_h256_gqa6_bf16kv
        else:
            kb = k_full.bfloat16()
            vb = v_full.bfloat16()
            sk = kb.abs().amax().clamp_min(1e-12).item() / _FP8_MAX
            sv = vb.abs().amax().clamp_min(1e-12).item() / _FP8_MAX
            k8 = (kb / sk).to(torch.float8_e4m3fn)
            v8 = (vb / sv).to(torch.float8_e4m3fn)
            kv = torch.empty((bsz * nb_max, hkv, _R4D_BLOCK, 2 * dim),
                             dtype=torch.float8_e4m3fn, device=dev)
            k8r = k8.view(bsz, -1, nb_max, _R4D_BLOCK, hkv, dim)
            v8r = v8.view(bsz, -1, nb_max, _R4D_BLOCK, hkv, dim)
            kv[..., :dim].copy_(
                k8r.permute(0, 2, 4, 3, 1, 5).reshape(bsz * nb_max, hkv, _R4D_BLOCK, dim))
            kv[..., dim:].copy_(
                v8r.permute(0, 2, 4, 3, 1, 5).reshape(bsz * nb_max, hkv, _R4D_BLOCK, dim))
            kd = torch.full((bsz, hkv), sk, dtype=torch.float32, device=dev)
            vd = torch.full((bsz, hkv), sv, dtype=torch.float32, device=dev)
            entry = lib.r4d_attn_prefill_h256_gqa6_fp8kv

        qb = args.q.bfloat16().contiguous().view(bsz * tq, hq, dim)
        if not qb.is_contiguous():
            qb = qb.contiguous()
        out_bf16 = torch.empty((bsz * tq, hq, dim), dtype=torch.bfloat16, device=dev)
        rbt = torch.arange(nb_max, dtype=torch.int32, device=dev).view(1, nb_max)
        rbt = rbt.expand(bsz, nb_max)
        # Block indices are per-sequence contiguous in the packed buffer.
        if bsz > 1:
            base = (torch.arange(bsz, dtype=torch.int32, device=dev) * nb_max).view(bsz, 1)
            rbt = (rbt + base).contiguous()
        rseq = torch.tensor(total_lens, dtype=torch.int32, device=dev)

        r4args = _R4DArgs(
            qb.data_ptr(), kv.data_ptr(), rbt.data_ptr(), rseq.data_ptr(),
            out_bf16.data_ptr(),
            kd.data_ptr() if kd is not None else None,
            vd.data_ptr() if vd is not None else None, None, None,
            bsz, tq, hq, hkv, dim, _R4D_BLOCK, nb_max,
            kv.stride(0), kv.stride(1),
            float(args.sm_scale), 0, max(total_lens),
        )
        stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)
        rc = entry(ctypes.byref(r4args), stream)
        if rc != 0:
            return None
    except Exception:
        return None

    out = torch.empty_like(args.q)
    out.copy_(out_bf16.view_as(out).to(args.q.dtype))
    return out
