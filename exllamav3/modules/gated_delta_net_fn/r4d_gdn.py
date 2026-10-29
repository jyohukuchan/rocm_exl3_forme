"""Opt-in libr4d GDN chunk scan (kkt_solve + chunk_scan) for long prefill.

Off by default (``EXL3_R4D_GDN=1`` enables). Replaces the vendored FLA
``chunk_gated_delta_rule`` call (recompute_w_u + chunk_fwd_h + chunk_fwd_o in
one WMMA-resident kernel) at the compiled geometry K=128/V=128/chunk=64; the
conv/gating prep and the gated norm stay on the existing path (r4d's
conv_prep fusion needs vLLM-style glue and is deliberately not wired: heavy,
see doc/r4d_exploration.md). Decode/recurrent, KDA, RDNA2-native and LAZY
paths are untouched (LAZY was reverted upstream: it breaks multi-turn).

License boundary: this file is only our own loader/adapter. It must never
vendor libr4d/vllm-mxfp4 sources, patches, or binaries. The ``.so`` is built
on the user's machine by ``rocm_tools/build_r4d.sh`` and located via
``EXL3_R4D_LIB`` (fallback ``R4D_SO``); when absent/unloadable every call
declines and the FLA path runs unchanged.

Accuracy gate: r4d's scan carries decay on a per-chunk midpoint reference
(see the kernel header), and libr4d issue #4 reports inexactness for in-chunk
decay span > 160 on the base pin. The committed test compares output AND
final state against the FLA chunk kernel, including a wide-span synthetic
case. A mismatch there means this backend stays measurement-only.
"""

import ctypes
import os
import time

import torch

_CHUNK = 64
_KD = 128
_VD = 128
_KKT_RATIOS = (1, 2, 3, 4, 6, 8)
# In-chunk cumsum gate-span limit. The scan carries decay on a per-chunk
# midpoint reference (halves clamped at e^80); past a cumsum span of ~100 the
# fp32 triangular solve loses the small entries and the result diverges from
# the FLA chunk kernel (measured knee between span ~97/rel 6e-3 and ~129/1e-2
# on a real 27B layer; synthetic spans <= 3 match at rel ~5e-3). Calls with any
# chunk above the limit decline to FLA.
_GDN_SPAN_LIMIT = 100.0


def r4d_gdn_enabled() -> bool:
    return os.environ.get("EXL3_R4D_GDN", "0") not in ("0", "", "off", "false")


_lib = None
_lib_failed = False

# Cumulative per-process stats when EXL3_R4D_STATS=1 (calls, engaged, declined,
# adapter seconds incl. any host sync). Read via r4d_gdn_stats().
_stats = {"calls": 0, "engaged": 0, "declined": 0, "adapter_s": 0.0}


def r4d_gdn_stats() -> dict:
    return dict(_stats)


def _load():
    global _lib, _lib_failed
    if _lib is not None:
        return _lib
    if _lib_failed:
        return None
    path = os.environ.get("EXL3_R4D_LIB", "").strip() or os.environ.get("R4D_SO", "").strip()
    if not path:
        return None
    try:
        lib = ctypes.CDLL(path)
        scan = getattr(lib, "r4d_gdn_chunk_scan_k128_v128_c64_bf16")
        scan.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_void_p,
        ]
        scan.restype = ctypes.c_int
        kkt = getattr(lib, "r4d_gdn_kkt_solve_k128_c64_bf16")
        kkt.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        kkt.restype = ctypes.c_int
        _lib = lib
        return _lib
    except Exception:
        _lib_failed = True
        return None


def _is_gfx1201(device: torch.device) -> bool:
    try:
        from ..attention_fn.triton_paged import _gfx_arch

        return _gfx_arch(device).startswith("gfx12")
    except Exception:
        return False


def _l2norm_bf16(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


def _record(engaged: bool, _t0: float | None, out):
    if _t0 is not None:
        _stats["calls"] += 1
        _stats["engaged" if engaged else "declined"] += 1
        _stats["adapter_s"] += time.perf_counter() - _t0
    return out


def fn_r4d_gdn_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0_kv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Timed wrapper: records engage/decline counts + adapter seconds in
    ``_stats`` when ``EXL3_R4D_STATS=1`` (zero overhead otherwise)."""
    _t0 = time.perf_counter() if os.environ.get("EXL3_R4D_STATS") == "1" else None
    out = _fn_r4d_gdn_chunk_inner(q, k, v, g, beta, h0_kv)
    return _record(out is not None, _t0, out)


def _fn_r4d_gdn_chunk_inner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0_kv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One-sequence GDN chunk forward via r4d kkt_solve + chunk_scan.

    q/k: [1, T, Hg, 128] bf16; v: [1, T, H, 128] bf16; g: [1, T, H] fp32;
    beta: [1, T, H] bf16/fp32; h0_kv: [1, H, 128, 128] fp32 or None.
    Returns (o_bf16 [1, T, H, 128], ht_kv [1, H, 128, 128] fp32) or None when
    the shape is unsupported, the library is missing, or a kernel reports an
    error (caller falls back to the FLA chunk kernel).
    """
    if not r4d_gdn_enabled():
        return None
    try:
        if (
            q.ndim != 4 or k.ndim != 4 or v.ndim != 4
            or q.shape[0] != 1 or k.shape[0] != 1 or v.shape[0] != 1
            or q.shape != k.shape
            or q.shape[3] != _KD or v.shape[3] != _VD
            or q.shape[1] != v.shape[1]
            or g.shape != beta.shape
            or g.shape[:2] != q.shape[:2] or g.shape[2] != v.shape[2]
        ):
            return None
        t, hg, h = q.shape[1], q.shape[2], v.shape[2]
        if t == 0 or h % hg != 0 or (h // hg) not in _KKT_RATIOS:
            return None
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            return None
        dev = q.device
        if not (k.device == dev and v.device == dev and g.device == dev and beta.device == dev):
            return None
        if not _is_gfx1201(dev):
            return None
        lib = _load()
        if lib is None:
            return None

        gf = g.reshape(t, h).float()
        if not gf.is_contiguous():
            gf = gf.contiguous()
        # r4d consumes g summed along the chunk (per-chunk cumsum), while FLA
        # takes per-token log-decay: reshape into 64-wide chunks (plus a tail)
        # and cumsum within each. Full-sequence cumsum would overflow fp32.
        # The cumsum is needed by the kernel anyway, so the span gate below
        # reuses it; the gate runs BEFORE the expensive prep (l2norm,
        # transposes, output allocs) so declined calls pay only one cumsum,
        # one fused span reduction and a single host sync.
        n_full = (t // _CHUNK) * _CHUNK
        gc = torch.empty_like(gf)
        if n_full > 0:
            gc[:n_full] = (
                gf[:n_full].view(-1, _CHUNK, h).cumsum(dim=1).reshape(n_full, h)
            )
        if n_full < t:
            # The tail is its own (partial) chunk: the cumsum restarts there.
            gc[n_full:] = gf[n_full:].cumsum(dim=0)
        # Span gate: decline calls with any chunk above the kernel's exactness
        # range instead of returning a degraded result (see _GDN_SPAN_LIMIT).
        span_t = torch.zeros((), dtype=torch.float32, device=dev)
        if n_full > 0:
            gb = gc[:n_full].view(-1, _CHUNK, h)
            span_t = torch.maximum(span_t, (gb.amax(dim=1) - gb.amin(dim=1)).amax())
        if n_full < t:
            tail = gc[n_full:]
            span_t = torch.maximum(
                span_t, (tail.amax(dim=0) - tail.amin(dim=0)).amax()
            )
        if span_t.item() > _GDN_SPAN_LIMIT:
            return None
        bf = beta.reshape(t, h).float().contiguous()
        qn = _l2norm_bf16(q.reshape(t, hg, _KD)).contiguous()
        kn = _l2norm_bf16(k.reshape(t, hg, _KD)).contiguous()
        vv = v.reshape(t, h, _VD).contiguous()
        if vv.dtype != torch.bfloat16:
            vv = vv.to(torch.bfloat16)
        if h0_kv is None:
            h0 = torch.zeros((1, h, _VD, _KD), dtype=torch.float32, device=dev)
        else:
            if h0_kv.shape != (1, h, _KD, _VD):
                return None
            h0 = h0_kv.transpose(-2, -1).contiguous()
        a = torch.empty((t, h, _CHUNK), dtype=torch.bfloat16, device=dev)
        o = torch.empty((t, h, _VD), dtype=torch.bfloat16, device=dev)
        ht = torch.empty((1, h, _VD, _KD), dtype=torch.float32, device=dev)
        cu = torch.tensor([0, t], dtype=torch.int32, device=dev)
        stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

        gc = gc.contiguous()
        rc = lib.r4d_gdn_kkt_solve_k128_c64_bf16(
            kn.data_ptr(), bf.data_ptr(), gc.data_ptr(), a.data_ptr(),
            cu.data_ptr(), 1, t, h, hg, _KD, _CHUNK, stream,
        )
        if rc != 0:
            return None
        rc = lib.r4d_gdn_chunk_scan_k128_v128_c64_bf16(
            qn.data_ptr(), kn.data_ptr(), vv.data_ptr(), a.data_ptr(),
            gc.data_ptr(), bf.data_ptr(), h0.data_ptr(), o.data_ptr(),
            ht.data_ptr(), cu.data_ptr(),
            1, h, hg, _KD, _VD, _CHUNK, float(_KD**-0.5), stream,
        )
        if rc != 0:
            return None
        return o.view(1, t, h, _VD), ht.transpose(-2, -1).contiguous()
    except Exception:
        return None
