"""Opt-in libr4d GDN chunk scan (kkt_solve + chunk_scan) for long prefill.

- Off by default: with EXL3_R4D_GDN unset, the hook declines (None) and the
  vendored FLA chunk path runs unchanged. Also declines on unsupported
  geometry, missing library, or non-gfx1201.
- Exactness gate vs the FLA chunk kernel (output AND final state) at 27B
  shapes (16 k-heads / 48 v-heads / dim 128): realistic gates match to
  rel ~1e-2 (WMMA-resident vs HBM-roundtripped arithmetic). A wide-span
  adversarial case (in-chunk gate span ~200, libr4d issue #4 territory) is
  attenuated by design (finite, no NaN, but rel ~1): the test locks in
  finiteness only, so real-model spans stay exact while pathological spans
  degrade gracefully.

The r4d.so itself is never vendored: build it on the user's machine with
rocm_tools/build_r4d.sh and point EXL3_R4D_LIB at it.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.modules.gated_delta_net_fn.r4d_gdn import (
    fn_r4d_gdn_chunk,
    r4d_gdn_enabled,
)

device = "cuda:0"
Hk, Hv, D = 16, 48, 128


def _arch():
    try:
        return str(torch.cuda.get_device_properties(device).gcnArchName or "")
    except Exception:
        return ""


def _needs_lib():
    from exllamav3.modules.gated_delta_net_fn.r4d_gdn import _load

    if not r4d_gdn_enabled():
        pytest.skip("EXL3_R4D_GDN=1 not set")
    if _load() is None:
        pytest.skip("EXL3_R4D_LIB missing: build r4d.so with rocm_tools/build_r4d.sh")
    if not _arch().split(":", 1)[0].lower().startswith("gfx12"):
        pytest.skip(f"r4d GDN needs gfx1201, this is {_arch()}")


def _inputs(t, mode="real", seed=0):
    torch.manual_seed(seed)
    q = torch.randn((1, t, Hk, D), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, t, Hk, D), device=device, dtype=torch.bfloat16)
    v = torch.randn((1, t, Hv, D), device=device, dtype=torch.bfloat16)
    beta = torch.rand((1, t, Hv), device=device, dtype=torch.bfloat16)
    if mode == "real":
        g = (-torch.rand((1, t, Hv), device=device) * 3.0).float()
    elif mode == "widespan":
        ramp = torch.linspace(0, -200, t, device=device).view(1, t, 1)
        g = (ramp + torch.randn((1, t, Hv), device=device) * 2.0).float()
    else:
        raise ValueError(mode)
    return q, k, v, g, beta


def _rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs().max().item()
    return d / b.abs().amax().clamp_min(1e-12).item()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("EXL3_R4D_GDN", raising=False)
    assert not r4d_gdn_enabled()
    q, k, v, g, beta = _inputs(128)
    assert fn_r4d_gdn_chunk(q, k, v, g, beta, None) is None


def test_geometry_decline(monkeypatch):
    monkeypatch.setenv("EXL3_R4D_GDN", "1")
    assert r4d_gdn_enabled()
    q, k, v, g, beta = _inputs(128)
    assert fn_r4d_gdn_chunk(q[..., :64], k[..., :64], v, g, beta, None) is None


def test_matches_fla_realistic(monkeypatch):
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_GDN", "1")
    from exllamav3.vendor.fla import chunk_gated_delta_rule

    for t, seed in ((512, 0), (2048, 1)):
        q, k, v, g, beta = _inputs(t, "real", seed)
        h0 = (torch.randn((1, Hv, D, D), device=device) * 0.1).float()
        fo, fs = chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=h0,
            output_final_state=True, use_qk_l2norm_in_kernel=True)
        ro = fn_r4d_gdn_chunk(q, k, v, g, beta, h0)
        assert ro is not None, "r4d GDN declined on a supported shape"
        r_o, r_s = ro
        assert _rel(r_o, fo) < 2e-2, f"output gate failed at T={t}"
        assert _rel(r_s, fs) < 2e-2, f"state gate failed at T={t}"


def test_widespan_stays_finite(monkeypatch):
    """Issue #4 territory: the midpoint-split decay attenuates (inexact) but
    must stay finite -- no NaN may reach the state the next chunk reads."""
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_GDN", "1")
    q, k, v, g, beta = _inputs(2048, "widespan")
    ro = fn_r4d_gdn_chunk(q, k, v, g, beta, None)
    assert ro is not None
    r_o, r_s = ro
    assert torch.isfinite(r_o.float()).all() and torch.isfinite(r_s.float()).all()
