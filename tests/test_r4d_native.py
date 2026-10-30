"""Native libr4d KV cache layer (Stage A bf16, opt-in).

CacheLayer_r4d packs K/V into r4d's block-16 HND bf16 layout at write time;
prefill and decode kernels read it with no gather/repack and no host syncs.
Enabled per model build via layer_type=CacheLayer_r4d (model_init.py maps
EXL3_R4D_ATTN=bf16 to it when the library is present); only the 27B
full-attention geometry is served.

The r4d.so itself is never vendored: build it on the user's machine with
rocm_tools/build_r4d.sh and point EXL3_R4D_LIB at it.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.modules.attention_fn.r4d_attn import (
    fn_r4d_native_attn,
    r4d_lib_path,
    r4d_mode,
)
from exllamav3.modules.attention_fn.common import AttnArgs
from exllamav3.cache.r4d import CacheLayer_r4d

device = "cuda:0"
Hq, Hkv, D = 24, 4, 256
SM = D ** -0.5


def _arch():
    try:
        return str(torch.cuda.get_device_properties(device).gcnArchName or "")
    except Exception:
        return ""


def _needs_native():
    if r4d_mode() != "bf16":
        pytest.skip("needs EXL3_R4D_ATTN=bf16")
    if r4d_lib_path() is None or not os.path.exists(r4d_lib_path()):
        pytest.skip("EXL3_R4D_LIB not set: build r4d.so with rocm_tools/build_r4d.sh")
    if not _arch().split(":", 1)[0].lower().startswith("gfx12"):
        pytest.skip(f"native r4d cache needs gfx1201, this is {_arch()}")


class _FakeAttn:
    num_kv_heads = Hkv
    head_dim = D


def _args(q, k, v, bt, cs):
    return AttnArgs(1, q.shape[1], Hq, D, q.shape[1], Hkv,
                    q, k, v, None, None, True, SM,
                    None, None, None, 0.0, bt, cs,
                    None, None, None, None, None)


def _sdpa(qf, kf, vf):
    g = Hq // Hkv
    s = torch.einsum("bqhd,bkhd->bhqk", qf, kf.repeat_interleave(g, dim=2)) * SM
    tq, tk = qf.shape[1], kf.shape[1]
    qp = torch.arange(tq, device=qf.device).view(tq, 1)
    kp = torch.arange(tk, device=qf.device).view(1, tk)
    s = s.masked_fill(~(kp <= (tk - tq + qp)).view(1, 1, tq, tk), float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s.float(), -1),
                        vf.repeat_interleave(g, dim=2).float())


def _rel(a, b):
    d = (a.float() - b).abs()
    return (d.max() / b.abs().amax().clamp_min(1e-12)).item()


def test_native_prefill_decode_scrambled_pages():
    """Write-then-read over a scrambled page order; prefill, chunked prefill
    and two decode steps all match fp32 SDPA (bf16 pipeline noise ~5e-3)."""
    _needs_native()
    torch.manual_seed(0)
    layer = CacheLayer_r4d(None, _FakeAttn(), 0, 8 * 256)
    layer.alloc(torch.device(device))
    bt = torch.tensor([[5, 2, 7, 1, 0, 3, 6, 4]], dtype=torch.int32, device=device)

    q1 = torch.randn((1, 512, Hq, D), device=device, dtype=torch.float16)
    k1 = torch.randn((1, 512, Hkv, D), device=device, dtype=torch.float16)
    v1 = torch.randn_like(k1)
    o1 = fn_r4d_native_attn(_args(q1, k1, v1, bt, torch.tensor([0], dtype=torch.int32, device=device)), layer)
    assert _rel(o1, _sdpa(q1.float(), k1.float(), v1.float())) < 2e-2

    q2 = torch.randn((1, 256, Hq, D), device=device, dtype=torch.float16)
    k2 = torch.randn((1, 256, Hkv, D), device=device, dtype=torch.float16)
    v2 = torch.randn_like(k2)
    o2 = fn_r4d_native_attn(_args(q2, k2, v2, bt, torch.tensor([512], dtype=torch.int32, device=device)), layer)
    kf = torch.cat([k1, k2], dim=1)
    vf = torch.cat([v1, v2], dim=1)
    assert _rel(o2, _sdpa(q2.float(), kf.float(), vf.float())) < 2e-2

    for past in (768, 769):
        qd = torch.randn((1, 1, Hq, D), device=device, dtype=torch.float16)
        kd = torch.randn((1, 1, Hkv, D), device=device, dtype=torch.float16)
        vd = torch.randn_like(kd)
        od = fn_r4d_native_attn(
            _args(qd, kd, vd, bt, torch.tensor([past], dtype=torch.int32, device=device)), layer)
        kf = torch.cat([kf, kd], dim=1)
        vf = torch.cat([vf, vd], dim=1)
        assert _rel(od, _sdpa(qd.float(), kf.float(), vf.float())) < 2e-2
    layer.free()


def test_native_copy_page_partial():
    """copy_page moves whole 16-blocks covering [0, num_tokens)."""
    _needs_native()
    torch.manual_seed(1)
    a = CacheLayer_r4d(None, _FakeAttn(), 0, 8 * 256)
    b = CacheLayer_r4d(None, _FakeAttn(), 0, 8 * 256)
    a.alloc(torch.device(device))
    b.alloc(torch.device(device))
    a.kv.normal_()
    b.copy_page(a, 2, 5, 100)
    nblocks = (100 + 16 - 1) // 16
    assert torch.equal(
        b.kv[5 * 16:5 * 16 + nblocks], a.kv[2 * 16:2 * 16 + nblocks])
    a.free()
    b.free()


def test_native_get_kv_raises():
    """No fp16 cache exists: get_kv fails loudly instead of returning None."""
    _needs_native()
    layer = CacheLayer_r4d(None, _FakeAttn(), 0, 8 * 256)
    layer.alloc(torch.device(device))
    bt = torch.zeros((1, 8), dtype=torch.int32, device=device)
    cs = torch.zeros(1, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError):
        layer.get_kv(cs, bt)
    layer.free()


def test_r4d_switches(monkeypatch):
    """EXL3_R4D_OFF=1 kills everything; EXL3_R4D_DEFAULT=1 selects the
    bundle (bf16 mode + exact GDN); explicit env always wins over bundle."""
    import exllamav3.modules.attention_fn.r4d_attn as ra
    import exllamav3.modules.gated_delta_net_fn.r4d_gdn as rg

    monkeypatch.setenv("EXL3_R4D_ATTN", "bf16")
    monkeypatch.setenv("EXL3_R4D_OFF", "1")
    monkeypatch.delenv("EXL3_R4D_DEFAULT", raising=False)
    assert ra.r4d_mode() == "off"
    assert not rg.r4d_gdn_enabled()
    assert not rg.r4d_gdn_exact_enabled()

    monkeypatch.delenv("EXL3_R4D_ATTN", raising=False)
    monkeypatch.delenv("EXL3_R4D_OFF", raising=False)
    monkeypatch.setenv("EXL3_R4D_DEFAULT", "1")
    assert ra.r4d_mode() == "bf16"
    assert rg.r4d_gdn_exact_enabled()

    monkeypatch.setenv("EXL3_R4D_ATTN", "off")
    assert ra.r4d_mode() == "off"
    monkeypatch.setenv("EXL3_R4D_GDN_EXACT", "off")
    assert not rg.r4d_gdn_exact_enabled()
