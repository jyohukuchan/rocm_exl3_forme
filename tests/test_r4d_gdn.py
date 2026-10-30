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


def _needs_exact():
    from exllamav3.modules.gated_delta_net_fn import r4d_gdn as gdn

    if not (os.environ.get("EXL3_R4D_GDN2_LIB") or os.environ.get("R4D_GDN2_SO")):
        pytest.skip("EXL3_R4D_GDN2_LIB missing: build r4d_gdn2.so with rocm_tools/build_r4d_gdn2.sh")
    if not _arch().split(":", 1)[0].lower().startswith("gfx12"):
        pytest.skip(f"r4d GDN needs gfx1201, this is {_arch()}")
    return gdn


def _inputs(t, mode="real", seed=0):
    torch.manual_seed(seed)
    q = torch.randn((1, t, Hk, D), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, t, Hk, D), device=device, dtype=torch.bfloat16)
    v = torch.randn((1, t, Hv, D), device=device, dtype=torch.bfloat16)
    beta = torch.rand((1, t, Hv), device=device, dtype=torch.bfloat16)
    if mode == "real":
        # Realistic log-decay: negative, small magnitudes (real 27B layers show
        # |g| mean ~0.1-0.6). Large per-token means drift the per-chunk cumsum
        # past the kernel's exactness range (see _GDN_SPAN_LIMIT), which the
        # span gate must decline -- so the realistic case stays well inside it.
        g = (-torch.rand((1, t, Hv), device=device) * 0.6).float()
    elif mode == "widespan":
        # Issue #4 territory: per-token |g| ~3 with consistent sign gives an
        # in-chunk cumsum span ~200, past the stock kernel's exactness range
        # (gate declines at >100) but inside the exact scan's validated range
        # (upstream: per-token up to 92). Per-token magnitudes far beyond that
        # (e.g. a 0..-200 ramp) break the stock kkt_solve first and are out
        # of scope for both paths.
        g = (-3.0 + torch.randn((1, t, Hv), device=device) * 0.5).float()
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


def test_widespan_declines_to_fla(monkeypatch):
    """Issue #4 territory (in-chunk cumsum span ~200): the kernel would
    attenuate instead of matching FLA, so the adapter declines and the caller
    runs the FLA chunk kernel. Exactness over speed for out-of-range spans."""
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_GDN", "1")
    q, k, v, g, beta = _inputs(2048, "widespan")
    assert fn_r4d_gdn_chunk(q, k, v, g, beta, None) is None


def test_exact_matches_fla_widespan(monkeypatch):
    """EXL3_R4D_GDN_EXACT=1 (libr4d issue #4 fix): a wide-span case that the
    stock scan cannot serve engages and matches the FLA chunk kernel in both
    output and final state. No span gate, no host sync on this path."""
    gdn = _needs_exact()
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_GDN", "1")
    monkeypatch.setenv("EXL3_R4D_GDN_EXACT", "1")
    from exllamav3.vendor.fla import chunk_gated_delta_rule

    q, k, v, g, beta = _inputs(2048, "widespan", seed=3)
    h0 = (torch.randn((1, Hv, D, D), device=device) * 0.1).float()
    fo, fs = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=h0,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    ro = fn_r4d_gdn_chunk(q, k, v, g, beta, h0)
    assert ro is not None, "exact scan declined a supported shape"
    r_o, r_s = ro
    assert _rel(r_o, fo) < 1e-2, "exact-scan output gate failed"
    assert _rel(r_s, fs) < 1e-2, "exact-scan state gate failed"


def test_exact_missing_lib_raises(monkeypatch):
    """Asked-for-but-missing exact module is fatal, never a silent switch
    back to the stock scan."""
    from exllamav3.modules.gated_delta_net_fn import r4d_gdn as gdn

    monkeypatch.setenv("EXL3_R4D_GDN_EXACT", "1")
    monkeypatch.delenv("EXL3_R4D_GDN2_LIB", raising=False)
    monkeypatch.delenv("R4D_GDN2_SO", raising=False)
    gdn._exact_mod = None
    q, k, v, g, beta = _inputs(128)
    with pytest.raises(RuntimeError):
        fn_r4d_gdn_chunk(q, k, v, g, beta, None)
    gdn._exact_mod = None


def test_span_gate_boundary(monkeypatch):
    """The gate trips on cumsum span, not on absolute gate level: a shifted
    gate with the same span behaves the same (both decline here)."""
    _needs_lib()
    monkeypatch.setenv("EXL3_R4D_GDN", "1")
    torch.manual_seed(0)
    t, h = 512, Hv
    q = torch.randn((1, t, Hk, D), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, t, Hk, D), device=device, dtype=torch.bfloat16)
    v = torch.randn((1, t, Hv, D), device=device, dtype=torch.bfloat16)
    beta = torch.rand((1, t, Hv), device=device, dtype=torch.bfloat16)
    ramp = torch.linspace(0, -200, t, device=device).view(1, t, 1)
    g = (ramp + torch.randn((1, t, Hv), device=device) * 2.0).float()
    assert fn_r4d_gdn_chunk(q, k, v, g, beta, None) is None
    # Small-mean gate (cumsum span ~ tens): engages.
    g_small = (-torch.rand((1, t, Hv), device=device) * 0.6).float()
    got = fn_r4d_gdn_chunk(q, k, v, g_small, beta, None)
    assert got is not None
