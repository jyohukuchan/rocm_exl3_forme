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

Kernel-level vs FLA (output AND final state):

| T | gate | out rel | state rel |
|---|---|---|---|
| 2048 | realistic | 7.8e-3 | 9.8e-3 |
| 8192 | realistic | 9.0e-3 | 6.0e-3 |
| 32768 | realistic | 1.0e-2 | 8.5e-3 |
| 2048 | adversarial span ~200 (issue #4) | ~1.0 (attenuated, finite) | ~1.0 (finite) |

The ~1e-2 floor is inherent accumulation difference (WMMA-resident fp32 vs
HBM-roundtripped bf16 intermediates across kkt+solve+scan), not the norm: an
external-norm ablation gives byte-identical rel (7.84e-3 both ways), i.e. the
adapter's fp32 l2norm matches FLA's in-kernel norm. The adversarial case
confirms the documented behaviour: the midpoint-split decay (`e^80` clamps)
attenuates instead of NaN -- no poison reaches the next chunk's state.

27B real-model gate (full-model KLD vs FLA path, paged chunked prefill;
noise floor bit-identical 0.0):

| cases | mean KLD | top-1 |
|---|---|---|
| cases-v1 (65-1025 tok), pre-fix | 0.06-0.32 | 0.77-0.94 |
| cases-v1, span-gated | 1.1e-4 | 1.000 |
| 32K teacher-forced, pre-fix | 0.18 | 0.974 |
| 32K teacher-forced, span-gated | 0.019 | 0.991 |

Pre-fix the model-level drift far exceeded the kernel-level 1e-2. Root
cause (review follow-up, per-layer probe on real 27B vs FLA incl. final
state): NOT state carry, transpose, cumsum reset, or dtypes -- ht matches
while o diverges only on layers whose in-chunk gate-cumsum span exceeds
~100 (raw-g span knee 9.6 -> rel 0.27; cumsum-span knee 97 -> 6e-3,
129 -> 1.3e-2, 193 -> 0.27). The scan carries decay on a per-chunk midpoint
reference and the fp32 triangular solve loses small entries past span ~100
(cond ~ e^span); real 27B median cumsum span is 148.6, so 36/48 calls
diverged pre-fix. Fix: `_GDN_SPAN_LIMIT=100` -- calls with any chunk above
it decline to FLA (exactness over speed). Residual post-fix 32K KLD 0.019
is engaged-call ~1e-2 noise compounding (plus needle-region sensitivity,
same pattern as attention). 32K passkey needle (50% depth): PASS with the
gate (was never passing-relevant pre-fix). Per-call stats via
`EXL3_R4D_STATS=1` (`r4d_gdn_stats()`).

## Speed (R9700, layer ms; Hessian co-resident, interleaved)

| T | FLA chunk | r4d kkt+scan | speedup |
|---|---|---|---|
| 2048 | 1.30-1.34 | 0.64-0.66 | ~2.0x |
| 8192 | 5.40-5.44 | 3.23-3.27 | ~1.7x |
| 32768 | 29.5-30.6 | 12.7-12.8 | ~2.3-2.4x |

Matches the exploration estimate (2-2.5x from ~1/3 bytes).

27B end-to-end (chunked Generator path, max_chunk_size 2048, NEEDLE_STEPS=1
=> full 32K prefill + 1 token; GSQ quant job co-resident, A/B interleaved):

| backend | full 32K prefill + 1 tok (n=2) |
|---|---|
| FLA baseline | 24.14s / 25.11s (avg 24.6) |
| r4d GDN span-gated | 24.27s / 24.51s (avg 24.4, ~-1%, noise) |

No E2E win: the kernel-level 2x (0.77 vs 1.05 ms/call at T=2048) is eaten
by one host `.item()` sync per call (768 calls at 32K = 48 GDN layers x 16
chunks, 356 engaged / 412 declined; adapter ~19.9s host-side). Each sync
drains the queued layer work and serializes host/GPU launch, starving the
GPU between calls. The gate reorder (sync before l2norm/transpose/allocs)
halved prep waste but the sync itself is the floor. Batching the 48
per-chunk span checks into one sync per chunk (16 syncs instead of 768)
would cut most of it, but engaged savings are only ~0.28 ms/call on ~46%
of calls -- the net would stay ~neutral. Verdict: stays opt-in as a
measurement backend; not recommended for serving (no accuracy margin left
to trade and no speed gain at 32K).

## Open

- Batched single launch (one kkt+scan over N sequences with `cu`) instead of
  the current per-sequence loop -- same math, fewer launches.
- `gated_rmsnorm` wiring (plain tensors, trivial) and a decision on conv
  fusion (own glue; vLLM's is heavy and model-specific).
- 27B end-to-end prefill share after VRAM frees.

## Exact scan (libr4d #4 fix, 2026-10-30)

vllm-mxfp4 ships an exact scan (`moe-gdn2/radiance_gdn2_vs_v050.patch`,
enabled by `RADIANCE_GDN_SCAN_FIX=1`): same kernel without the midpoint
decay split, max error 98.6% -> 0.34% at the same speed, a diff against
libr4d v0.5.0's scan file. Our pin b9e42ab carries the byte-identical scan
file (only r4d.h differs: 0.4.0 vs 0.5.0), so the patch applies directly;
`rocm_tools/build_r4d_gdn2.sh` clones libr4d at the pin, fetches the patch
at build time, checks both md5s (scan + wmma header before, patched result
after -- fatal on mismatch, mirroring the author's policy), and compiles
the TU to `r4d_gdn2.so` with the same flags (`-O3 -ffp-contract=off
-mcumode`, gfx1201). Nothing vendored. `EXL3_R4D_GDN_EXACT=1` (implies the
hook; `EXL3_R4D_GDN2_LIB` points at the module) swaps the stock scan for
the exact one via its pybind surface (same 18 arguments; geometry asserted
128/128/64 at load). Asked-for-but-missing is fatal, never a silent
fallback. The exact scan is span-independent, so the span gate -- and with
it the per-call host sync -- is skipped.

Measured (R9700, GSQ quant job co-resident for E2E):

| gate | stock + span gate | exact scan |
|---|---|---|
| synthetic vs torch recurrent, mild | 1.1e-2 (engage) | 2.4e-2 (engage) |
| synthetic vs torch recurrent, span ~200 | decline | 8.5e-3 (engage) |
| synthetic vs torch recurrent, span ~640 | decline | 1.1e-2 (engage) |
| real 27B per-layer vs FLA @32K (768 calls) | 356 engage @~5e-3, 412 decline | 768 engage, mean 7.7e-3, max 1.9e-2 |
| cases-v1 KLD / top1 | 1.1e-4 / 1.000 | 3.1e-4 / 0.985 |
| 32K KLD / top1 | 0.019 / 0.991 | 0.035 / 0.987 |
| 32K needle (50%) | pass | pass |
| 8K full prefill + 1 tok | 7.2s (-21%) | 7.0s (-23%) |
| 32K full prefill + 1 tok | 24.4s (~0%) | 25.1s (~0%) |

Two honest caveats. First, at mild spans the exact kernel is ~2.5x noisier
than stock in our build (exact-vs-FLA 2.5e-2 vs stock 1.0e-2; stock-vs-exact
rel 2.3e-2 -- the author's "bit-identical" does not reproduce here, likely
rounding-order/codegen differences; both are O(1e-2) bf16 staging). So the
exact scan fixes the catastrophic tail but loses on average: gated KLD
(1.1e-4/0.019) beats exact (3.1e-4/0.035). Second, the sync removal pays at
8K (-23%, first-chunk still slower: init overhead) but 32K E2E stays neutral
for both -- the kkt_solve (unchanged, stock) and Amdahl dominate there.

Verdict: exact scan stays opt-in as the span-independent measurement
backend (no gate to tune, no syncs); the span-gated stock path keeps the
better KLD. Neither is recommended for serving.

## Error vs context length (2026-10-30)

Real 27B at 64K (captured inputs, offline): stock declines even at chunk 0
of layer 0 (real spans exceed 100 everywhere); exact engages at both ends
(~1e-2 vs FLA and vs the torch reference). GDN per-chunk error is
length-independent by construction (per-chunk cumsum restarts); the 64K
numbers match 32K. 128K capture infeasible (pool + trellis OOM, as above).
