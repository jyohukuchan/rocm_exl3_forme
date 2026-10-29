"""Opt-in libr4d prefill attention backend (Stage A/B).

- Off by default: with EXL3_R4D_ATTN unset, the hook declines (None) and the
  default Triton path is untouched (also when the .so is absent).
- Unsupported geometries decline (non-256 dim, non-GQA6, non-causal, ...).
- Stage A (bf16 KV + f16 legs): matches the fp16 Triton prefill to rel ~4e-3.
  Needs EXL3_R4D_LIB pointing at a locally built r4d.so on gfx1201; skips
  cleanly otherwise (library absent or other arch).
- Stage B (fp8 KV): f16 legs (mode 0) and O_PV8 (mode 2) only; O_QK8 modes
  (1/3) decline unless EXL3_R4D_ALLOW_QK8=1 (measurement only).

The r4d.so itself is never vendored: build it on the user's machine with
rocm_tools/build_r4d.sh and point EXL3_R4D_LIB at it.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.modules.attention_fn.r4d_attn import (
    _legs_mode,
    fn_r4d_attn_prefill,
    r4d_lib_path,
    r4d_mode,
)
from exllamav3.modules.attention_fn.common import AttnArgs
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill

device = "cuda:0"


def _arch():
    try:
        return str(torch.cuda.get_device_properties(device).gcnArchName or "")
    except Exception:
        return ""


def _needs_lib():
    if r4d_lib_path() is None or not os.path.exists(r4d_lib_path()):
        pytest.skip("EXL3_R4D_LIB not set: build r4d.so with rocm_tools/build_r4d.sh")
    if not _arch().split(":", 1)[0].lower().startswith("gfx12"):
        pytest.skip(f"r4d prefill needs gfx1201, this is {_arch()}")


def _case(hq=12, hkv=2, dim=256, tk=2048, tq=256, seed=7):
    torch.manual_seed(seed)
    page = 256
    assert tk % page == 0 and tk % 16 == 0
    past = tk - tq
    pages = tk // page
    q = torch.randn((1, tq, hq, dim), dtype=torch.float16, device=device)
    k = torch.randn((1, tq, hkv, dim), dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    kc = torch.randn((pages, page, hkv, dim), dtype=torch.float16, device=device)
    vc = torch.randn_like(kc)
    # deterministic past content: write the first `past` rows through the cache
    kp = torch.randn((1, past, hkv, dim), dtype=torch.float16, device=device)
    vp = torch.randn_like(kp)
    kc.view(1, tk, hkv, dim)[:, :past].copy_(kp)
    vc.view(1, tk, hkv, dim)[:, :past].copy_(vp)
    bt = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    cs = torch.tensor([past], dtype=torch.int32, device=device)
    args = AttnArgs(
        1, tq, hq, dim, tq, hkv, q, k, v, kc, vc, True, dim**-0.5,
        None, None, None, 0.0, bt, cs,
    )
    return args, q, k, v, kc, vc, bt, cs


def _rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    return d / b.float().abs().amax().clamp_min(1e-12).item()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("EXL3_R4D_ATTN", raising=False)
    assert r4d_mode() == "off"
    args, *_ = _case()
    assert fn_r4d_attn_prefill(args) is None


def test_unsupported_geometry_declines(monkeypatch):
    monkeypatch.setenv("EXL3_R4D_ATTN", "bf16")
    assert r4d_mode() == "bf16"
    args, q, k, v, kc, vc, bt, cs = _case(hq=8, hkv=2)  # GQA 4, not 6
    assert fn_r4d_attn_prefill(args) is None
    args, q, k, v, kc, vc, bt, cs = _case()
    args = args._replace(causal=False)
    assert fn_r4d_attn_prefill(args) is None
    args, *_ = _case()
    args = args._replace(softcap=50.0)
    assert fn_r4d_attn_prefill(args) is None


def test_bf16_matches_triton(monkeypatch):
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_ATTN", "bf16")
    args, q, k, v, kc, vc, bt, cs = _case()
    got = fn_r4d_attn_prefill(args)
    assert got is not None, "r4d bf16 prefill declined on a supported shape"
    ref = paged_attn_triton_prefill(
        q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=256**-0.5,
        out=torch.empty_like(q),
    )
    assert _rel(got, ref) < 8e-3, "Stage A bf16 KV should match fp16 Triton"


def test_nocache_matches_triton(monkeypatch):
    """Cache-less full-sequence calls (KLD dumps, bench_prefill) also engage."""
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_ATTN", "bf16")
    torch.manual_seed(9)
    bsz, tq, hq, hkv, dim = 1, 256, 12, 2, 256
    q = torch.randn((bsz, tq, hq, dim), dtype=torch.float16, device=device)
    k = torch.randn((bsz, tq, hkv, dim), dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    args = AttnArgs(
        bsz, tq, hq, dim, tq, hkv, q, k, v, None, None, True, dim**-0.5,
        None, None, None, 0.0, None, None,
    )
    got = fn_r4d_attn_prefill(args)
    assert got is not None, "r4d bf16 declined on a supported cache-less shape"
    ref = paged_attn_triton_prefill(
        q, None, None, None, None, None, None, causal=True,
        softmax_scale=dim**-0.5, k_new=k, v_new=v, out=torch.empty_like(q))
    assert _rel(got, ref) < 8e-3, "cache-less Stage A should match fp16 Triton"


def test_unaligned_length_matches_triton(monkeypatch):
    """Non-multiple-of-16 lengths pad the packed buffer (kernel masks tails)."""
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_ATTN", "bf16")
    torch.manual_seed(11)
    bsz, tq, hq, hkv, dim = 1, 80, 12, 2, 256
    q = torch.randn((bsz, tq, hq, dim), dtype=torch.float16, device=device)
    k = torch.randn((bsz, tq, hkv, dim), dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    args = AttnArgs(
        bsz, tq, hq, dim, tq, hkv, q, k, v, None, None, True, dim**-0.5,
        None, None, None, 0.0, None, None,
    )
    got = fn_r4d_attn_prefill(args)
    assert got is not None, "r4d bf16 declined on an unaligned shape"
    ref = paged_attn_triton_prefill(
        q, None, None, None, None, None, None, causal=True,
        softmax_scale=dim**-0.5, k_new=k, v_new=v, out=torch.empty_like(q))
    assert _rel(got, ref) < 8e-3, "unaligned Stage A should match fp16 Triton"


def test_fp8_legs_gate(monkeypatch):
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_ATTN", "fp8")
    legs = _legs_mode()
    args, q, k, v, kc, vc, bt, cs = _case()
    if legs in (1, 3) and os.environ.get("EXL3_R4D_ALLOW_QK8", "0") != "1":
        assert fn_r4d_attn_prefill(args) is None
        pytest.skip(f"QK8 legs mode {legs} correctly declined without ALLOW_QK8")
    got = fn_r4d_attn_prefill(args)
    assert got is not None, f"r4d fp8 prefill declined (legs mode {legs})"
    ref = paged_attn_triton_prefill(
        q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=256**-0.5,
        out=torch.empty_like(q),
    )
    # fp8 KV quantization noise dominates (bench: mode0 ~3.3e-2, mode2 ~4.5e-2,
    # seed-dependent up to ~5e-2). The wiring bugs this guards against (wrong
    # layout/scales) show up at rel ~0.5-1.5, so 8e-2 still catches them while
    # staying seed-robust. Exact accuracy numbers live in doc/bench, gated by KLD.
    assert _rel(got, ref) < 0.08, f"Stage B fp8 KV legs mode {legs} exceeds tolerance"
