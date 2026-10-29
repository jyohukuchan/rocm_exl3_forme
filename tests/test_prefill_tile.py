"""Retuned fp16 prefill tile defaults (Round-4 / r4d-explore).

- gfx1201 (RDNA4, R9700), head_dim 256: default BM128/BN64/S1, ~1.5x over BM64/BN32/S2.
- Other arches (incl. gfx1030/V620): default stays BM64/BN32/S2 (new tile 2-3x slower there).
- EXL3_PREFILL_TILE="BM,BN,STAGES[,WARPS]" pins the tile for A/B.

Checks outputs only (speed is covered by eval/prof benches, not unit tests):
default/old/new all match the fp32 reference, and the env override parses/pins.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import (
    _gfx_arch,
    _prefill_tile_override,
    paged_attn_triton_prefill,
)

device = "cuda:0"


def ref_attn(q, k, v, causal):
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    kk = k.repeat_interleave(g, dim=2).float()
    vv = v.repeat_interleave(g, dim=2).float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * D ** -0.5
    qpos = (T - Q + torch.arange(Q, device=q.device)).view(Q, 1)
    kpos = torch.arange(T, device=q.device).view(1, T)
    mask = kpos <= qpos if causal else torch.ones((Q, T), dtype=torch.bool, device=q.device)
    s = s.masked_fill(~mask.view(1, 1, Q, T), float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vv)


def make_case():
    torch.manual_seed(1234)
    B, past, q_len, Hq, Hkv, D = 1, 512, 256, 8, 4, 256
    T = past + q_len
    pages = 4
    kc = torch.randn((pages, 256, Hkv, D), dtype=torch.half, device=device)
    vc = torch.randn_like(kc)
    bt = torch.arange(pages, device=device, dtype=torch.int32).view(1, pages)
    sl = torch.full((B,), past, dtype=torch.int32, device=device)
    k = torch.randn((B, q_len, Hkv, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    q = torch.randn((B, q_len, Hq, D), dtype=torch.half, device=device)
    return q, k, v, kc, vc, bt, sl, T


def test_gfx_arch_probe():
    arch = _gfx_arch(torch.device(device))
    assert isinstance(arch, str) and arch.startswith("gfx")


def test_prefill_tile_override_parsing(monkeypatch):
    monkeypatch.delenv("EXL3_PREFILL_TILE", raising=False)
    assert _prefill_tile_override() is None
    monkeypatch.setenv("EXL3_PREFILL_TILE", "auto")
    assert _prefill_tile_override() is None
    monkeypatch.setenv("EXL3_PREFILL_TILE", "128,64,1")
    assert _prefill_tile_override() == (128, 64, 8, 1)
    monkeypatch.setenv("EXL3_PREFILL_TILE", "64,32,2,8")
    assert _prefill_tile_override() == (64, 32, 8, 2)
    monkeypatch.setenv("EXL3_PREFILL_TILE", "bogus")
    with pytest.raises(ValueError):
        _prefill_tile_override()


def _rel(a, b):
    d = (a.float() - b.float()).abs().max().item()
    return d / b.float().abs().amax().clamp_min(1e-12).item()


def test_prefill_tiles_match_reference():
    q, k, v, kc, vc, bt, sl, T = make_case()
    outs = {}
    outs["default"] = paged_attn_triton_prefill(
        q, k, v, kc, vc, bt, sl, causal=True)
    outs["old"] = paged_attn_triton_prefill(
        q, k, v, kc, vc, bt, sl, causal=True,
        block_m=64, block_n=32, num_warps=8, num_stages=2)
    outs["new"] = paged_attn_triton_prefill(
        q, k, v, kc, vc, bt, sl, causal=True,
        block_m=128, block_n=64, num_warps=8, num_stages=1)
    # reference over the logical cache window: append rows were written into kc/vc
    # by the call above, so gather the first T positions through the block table
    flat_k = kc[bt.long().view(-1)].view(1, -1, kc.shape[2], kc.shape[3])[:, :T]
    flat_v = vc[bt.long().view(-1)].view(1, -1, vc.shape[2], vc.shape[3])[:, :T]
    ref = ref_attn(q, flat_k, flat_v, True)
    for name, o in outs.items():
        assert _rel(o, ref) < 8e-3, f"{name} rel err vs ref"
    # old vs new agree to reduction-order noise (expected ~0, tolerance like hdpad tests)
    assert _rel(outs["old"], outs["new"]) < 8e-3
    # default equals whichever tile the arch selects (bit-identical: same launch config)
    arch = _gfx_arch(torch.device(device))
    expect = outs["new"] if arch.startswith("gfx12") else outs["old"]
    assert torch.equal(outs["default"], expect), f"default mismatch on {arch}"
