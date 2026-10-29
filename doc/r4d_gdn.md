# r4d GDN chunk scan (opt-in backend for long prefill)

Status (branch `feat/r4d-integration`): kkt_solve + chunk_scan wired as an
opt-in replacement for the vendored FLA `chunk_gated_delta_rule` call,
kernel-tested. Opt-in only; default path unchanged.

## What it is

`exllamav3/modules/gated_delta_net_fn/r4d_gdn.py` (our loader/adapter) calls
`r4d_gdn_kkt_solve_k128_c64_bf16` (FLA `chunk_scaled_dot_kkt_fwd` +
`solve_tril`, gram never in HBM) and `r4d_gdn_chunk_scan_k128_v128_c64_bf16`
(FLA `recompute_w_u_fwd` + `chunk_gated_delta_rule_fwd_h` + `chunk_fwd_o` in
one kernel, state WMMA-resident). Wired per sequence into
`gated_delta_rule_fn`'s chunk branch; declines (None) when disabled or
unsupported so FLA runs.

Stays on the existing path: conv/gating prep (r4d's conv_prep fusion needs
vLLM-style glue and is deliberately not wired -- heavy), the gated norm,
decode/recurrent, KDA, the RDNA2-native default, and LAZY (reverted upstream:
breaks multi-turn).

## Environment

| Var | Default | Meaning |
|---|---|---|
| `EXL3_R4D_GDN` | `0` | `1` enables the r4d chunk path (long prefill, GDN layers). |
| `EXL3_R4D_LIB` | empty | Same `.so` as the attention backend (`R4D_SO` fallback). |

Read per call, so one process can A/B FLA vs r4d chunk-for-chunk.

## Adapter notes (the two conventions that matter)

- **g is per-chunk cumsum'd**: r4d consumes "g summed along the chunk"
  (conv_prep's output form), while FLA takes per-token log-decay. The adapter
  reshapes into 64-wide chunks (+tail restarts) and cusums within each.
  Passing raw g mismatches at rel ~15 (measured); the committed test would
  catch it. Full-sequence cumsum would overflow fp32 -- never do that.
- **q/k l2norm is outside the kernel**: the scan has no norm; FLA was called
  with `use_qk_l2norm_in_kernel=True`, so the adapter norms in fp32
  (eps 1e-6, matching the reference) and passes bf16.
- State layout: EXL3 keeps `[H, K, V]` fp32, r4d `[H, V, K]` -- transposed on
  the way in/out. Scale `K^-0.5`. `h0=None` means zeros; a dummy `ht` is
  always provided (a few MB) and ignored unless the caller saves state.

## Accuracy gate (R9700, 27B shapes: 16 k / 48 v / dim 128)

| T | gate | out rel | state rel |
|---|---|---|---|
| 2048 | realistic | 7.8e-3 | 9.8e-3 |
| 8192 | realistic | 9.0e-3 | 6.0e-3 |
| 32768 | realistic | 1.0e-2 | 8.5e-3 |
| 2048 | adversarial span ~200 (issue #4) | ~1.0 (attenuated, finite) | ~1.0 (finite) |

Realistic gates pass at rel ~1e-2 (WMMA-resident vs HBM-roundtripped
arithmetic). The adversarial case confirms the documented behaviour: the
midpoint-split decay (`e^80` clamps) attenuates instead of NaN -- no poison
reaches the next chunk's state. Real-model spans (~tens) stay exact.

## Speed (R9700, layer ms; Hessian co-resident, interleaved)

| T | FLA chunk | r4d kkt+scan | speedup |
|---|---|---|---|
| 2048 | 1.30-1.34 | 0.64-0.66 | ~2.0x |
| 8192 | 5.40-5.44 | 3.23-3.27 | ~1.7x |
| 32768 | 29.5-30.6 | 12.7-12.8 | ~2.3-2.4x |

Matches the exploration estimate (2-2.5x from ~1/3 bytes). 27B end-to-end
pending VRAM (Hessian until ~05:00 JST).

## Open

- Batched single launch (one kkt+scan over N sequences with `cu`) instead of
  the current per-sequence loop -- same math, fewer launches.
- `gated_rmsnorm` wiring (plain tensors, trivial) and a decision on conv
  fusion (own glue; vLLM's is heavy and model-specific).
- 27B end-to-end prefill share after VRAM frees.
