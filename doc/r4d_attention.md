# r4d prefill attention (opt-in backend)

Status (branch `feat/r4d-integration`): Stage A (bf16 KV + f16 legs) implemented
and kernel-tested, opt-in. Stage B (fp8 KV + O_PV8) implemented but **pending
the accuracy gate** (27B KLD + 32K+ long-context check) before serving use.
O_QK8 modes (1/3) stay off (declined) unless explicitly allowed for measurement.

## What it is

An opt-in prefill attention backend that calls libr4d's
`r4d_attn_prefill_h256_gqa6_{bf16,fp8}kv` kernels (compiled for head_dim 256,
GQA 6, paged block 16, causal, bf16 query) through a ctypes loader, instead of
the built-in Triton prefill kernel. Decode keeps the existing path.

- Loader/adapter: `exllamav3/modules/attention_fn/r4d_attn.py` (our code).
- Dispatch: tried ahead of the INT8FP8 path in
  `exllamav3/modules/attention_fn/dispatch.py`; declines (returns None) for
  anything it does not serve, so the default path is bit-for-bit unchanged.
- Build: `rocm_tools/build_r4d.sh` (clone at pin + patch fetched at build
  time; nothing vendored). Tests: `tests/test_r4d_attn.py`.

## Environment

| Var | Default | Meaning |
|---|---|---|
| `EXL3_R4D_LIB` | empty (off) | Path to a locally built `r4d.so`. Fallback: `R4D_SO` (bench compat). Absent/unloadable = clean fallback to Triton. |
| `EXL3_R4D_ATTN` | `0` (off) | `1`/`bf16` = Stage A (bf16 KV). `fp8` = Stage B (fp8 KV). Anything else = off. |
| `R4D_ATTN_FP8` | `0` | **Library-owned** legs switch (read once per process by r4d.so; set before start): `0` f16 legs, `1` O_QK8, `2` O_PV8, `3` both. Applies to the fp8-KV entry only. |
| `EXL3_R4D_ALLOW_QK8` | `0` | `1` permits legs modes 1/3 for KLD measurement. Serving keeps them off: without this, fp8 calls with mode 1/3 decline to Triton. |

`EXL3_R4D_ATTN`/`EXL3_R4D_ALLOW_QK8` are read per call (A/B in one process);
`R4D_ATTN_FP8` is read once per process by the library.

## Constraints (decline to Triton when unmet)

head_dim 256, exactly 6 q heads per kv head, causal, `q_len > 16` with
appended K/V length equal to `q_len`, no window/sinks/softcap/alibi/per-(token,head)
scales, no varlen/quant-cache, gfx1201 only. Paged fp16 cache and cache-less
K/V are both served; any attended length works (the adapter pads the packed
buffer to whole 16-blocks and passes the exact length in seqused_k -- the
kernel masks q tails and clamps K reads). Q is converted fp16->bf16 per call
(cheap); the bf16 output is converted back to fp16.

## KV adapter (per-call conversion for now)

r4d wants block-16 K/V-packed (HND, bf16/fp8) caches; EXL3 keeps PAGE-256 fp16
caches. Each prefill call gathers the referenced window (past + appended rows)
and packs it into the r4d layout (a 256-page maps to 16 consecutive 16-blocks).
fp8 KV uses per-tensor scales (vLLM production style: `amax/448`, one descale
per (seq, kv head)). Measured loader overhead at 32K/2K: ~2.6 ms against a
~15.5 ms kernel (ctypes-direct 15.51 ms vs loader 18.11 ms, Hessian
co-resident). The overhead is O(Tk) vs the kernel's O(Tk*Tq), so the ratio
improves with query length; a write-path cache variant (store r4d layout/dtype
directly) can remove it later.

## Accuracy (R9700, 27B gates -- measured 2026-10-30)

Kernel-level: Stage A bf16 max-rel ~3.7e-3; Stage B fp8 mode 0 ~3.3-5.0e-2,
mode 2 (O_PV8) ~4.5e-2, modes 1/3 (O_QK8) ~0.5 (quantization noise included).

27B cases-v1 KLD vs fp16 attention (9 cases, 65-1025 tokens, paged path,
macro KLD / top-1 agreement; accepted INT8-path reference: 0.0 everywhere
except one case 0.0024/0.985):

| backend | macro KLD | top-1 | verdict |
|---|---|---|---|
| Stage A bf16 | 0.0001 | 0.996 | short-context clean |
| Stage B fp8 mode 2 (O_PV8) | 0.0031 | 0.971 | small but real shift |
| fp8 mode 1 (O_QK8) | 0.0176 | 0.934 | keep off |
| fp8 mode 3 (both) | 0.0168 | 0.946 | keep off |

32K teacher-forced prefill KLD over all positions (paged chunked path,
past>0 covered; noise floor bit-identical 0.0):

| backend | mean KLD | max KLD | top-1 |
|---|---|---|---|
| Stage A bf16 | 0.031 | 17.1 | 0.988 |
| Stage B fp8 mode 2 | 0.131 | 22.1 | 0.972 |

32K chat-templated passkey needle (same 32K/21K prompts for all backends):

| backend | 32K (50%) | 32K (25%) | 21K varied |
|---|---|---|---|
| fp16 baseline | pass | pass | pass |
| Stage A bf16 | **fail** (refusal) | **fail** (refusal) | **fail** (wrong digits) |
| Stage B fp8 mode 2 | **fail** (refusal) | -- | -- |

The needle failures are systematic (r4d 0/5 vs baseline 3/3) and the 32K KLD
drift concentrates at the needle region -- retrieval at long context moves
even for Stage A. Lesson: kernel rel 3.7e-3 is not accuracy-risk-free at
long context; per-layer error compounds over 32K positions x dozens of
layers. Both stages stay opt-in (default off); Stage B is not blessed for
serving. The user decides on modes 1/3 (KLD above) and on any long-context
use of Stage A (needs a wider long-context eval first).

### Review follow-up: adapter-bug hunt (2026-10-30) -- no adapter bug found

Parent review suspected the paged past>0 path (page-to-16-block mapping,
block-table order, seqused_k / q-offset for causal masking with Tq < Tk,
dedup'd pagetable pages), the Q fp16->bf16 scale, or the softmax scale.
Per-layer probe vs fp32 SDPA reference on real 27B Q/K/V at every 32K chunk
(all past lengths 0..30720, 256 layer-calls) plus a synthetic past=30720 /
Tq=2048 / Tk=32768 case:

| comparison | rel err |
|---|---|
| r4d vs Triton, real 27B, all past>0 | 3e-3 .. 2.5e-2, uniform (no past>0 blowup) |
| r4d vs Triton, synthetic past>0 | 4.5e-3 |
| r4d vs fp32 SDPA, synthetic past>0 | 4.6e-3 |
| Triton vs fp32 SDPA, synthetic past>0 | 3.9e-4 |

The adapter math is right (Triton matches fp32 SDPA to 4e-4 through the same
layout/block-table/seqused path); the r4d kernel is 10-50x noisier
everywhere, a precision floor of its bf16 accumulation, not a scale or
indexing bug. A scale bug would show as a large systematic error, not
uniform 1e-2-level noise. Conclusion stands: inherent kernel precision,
Stage A stays opt-in and not recommended for long-context serving.

## Speed

Layer ms on R9700 (Hessian co-resident early, GSQ quant job later; A/B
interleaved, medians):

| Tk | Tq | r4d bf16 (ctypes) | r4d bf16 (loader) | fp16 retuned (default) |
|---|---|---|---|---|
| 32768 | 2048 | 15.51 (1.56x) | 18.11 (1.34x) | 24.2 (1.00x) |

Exploration sweep (ctypes): bf16kv 1.5-1.6x, fp8kv+f16 legs 1.7-1.85x,
mode 1 ~1.9-2.0x, mode 2 ~2.0-2.1x, mode 3 ~2.6-2.8x over retuned fp16.

27B end-to-end (fair baseline `EXL3_HGEMM_FP16_OUT=1`, random tokens):

| length | fp16 default | r4d bf16 | r4d fp8 mode 2 |
|---|---|---|---|
| 8K full prefill | 1573 tok/s (5.21s) | 1600 tok/s (5.12s) | 1597 tok/s (5.13s) |

At 8K attention is ~9% of prefill (linears 51%, GDN 39%), so +2% is the
expected order -- kernel wins only move E2E where attention dominates.
Full-forward E2E past 8K is impossible in this harness on 34 GB: the
32Kx248K fp16 logits alone are a 16 GB single tensor (OOM, fundamental --
not fragmentation). Clean 32K E2E comes from the chunked Generator path
(max_chunk_size 2048, chat-templated passkey prompt, NEEDLE_STEPS=1 so
total = full 32K prefill + 1 decode token; pass=False expected -- one token
cannot contain the key; accuracy comes from the KLD/needle gates above):

| backend | full 32K prefill + 1 tok (n=2) |
|---|---|
| fp16 baseline | 24.14s / 25.11s (avg 24.6) |
| Stage A bf16 | 23.22s / 23.26s (avg 23.2, -5.5%) |
| Stage B fp8 mode 2 | 22.98s / 23.05s (avg 23.0, -6.4%) |

So the kernel wins (1.5-2.1x on attention) shrink to ~6% E2E at 32K: the
stateless adapter repacks the whole attended window per layer-chunk
(gather through the block table + fp16->bf16 + interleave into r4d layout,
O(window) per call = O(T^2) per prefill) plus one host `.item()` sync per
call. Measured adapter time at 32K: ~18.5s host-side over 288 calls (all
256 prefill layer-chunks = 16 full-attention layers x 16 chunks engaged;
avg ~64 ms/call, growing with window). A persistent
r4d-layout cache layer (pack once, append per chunk) would remove the
repack, but that redesign is only worth building if the kernel precision
gate ever passes -- it currently fails (needle 0/5), so speed is moot for
serving. Updated projections from measured kernel speedups x the
8K/128K/1M shares (attn 9%/69%/95%) -- upper bounds, adapter overhead not
included:

| config | 8K | 128K | 1M |
|---|---|---|---|
| attn Stage A loader (~1.4x @32K) | ~1.03x | ~1.25x | ~1.37x |
| attn fp8kv+f16 (~1.8x) | ~1.06x | ~1.44x | ~1.7x |
| attn mode 2 (~2.0x) | ~1.07x | ~1.5x | ~1.9x |
| attn mode 3 (~2.7x) | ~1.1x | ~1.8x | ~2.5x |
| + GDN r4d (~2x @32K) | ~1.2x | -- | -- (attn-bound at 1M) |

Rough (kernel speedups x 8K/128K/1M attn shares 9%/69%/95%; GDN share past
8K unmeasured, so no GDN projection past 8K).

## bf16 pipeline: intrinsic, not libr4d-specific (2026-10-30)

Parent review noted vllm-mxfp4's numbers (R4D 1.69e-3 vs AITER 2.28e-3
against an fp32 oracle, both bf16 pipelines) and asked whether the 32K
needle failure is a bf16 effect rather than an r4d bug. `EXL3_TRITON_BF16_IO=1`
rounds Q/K/V to bf16 values through our fp16 Triton kernel (fp32
accumulation) and rounds the output the same way -- a bf16-I/O pipeline with
no libr4d code anywhere. Per-layer probe on real 27B at 32K (256 calls, vs
fp32 SDPA on bf16-valued inputs; rel_F = ||out-ref||_F / ||ref||_F):

| backend | max-rel (mean) | rel_F (mean) |
|---|---|---|
| fp16 Triton (own oracle) | 3.9e-4 | 2.1e-4 |
| bf16-I/O Triton | 3.1e-3 | 1.68e-3 |
| r4d bf16 | 3.2e-3 | 1.68e-3 |
| r4d fp8 mode 2 | 9.8e-2 | 4.95e-2 |
| vllm-mxfp4 claim (their harness) | -- | 1.69e-3 |

bf16-I/O Triton and r4d bf16 are identical (1.681e-3 vs 1.676e-3, both ~ the
published 1.69e-3): the whole bf16 error budget is I/O quantization, the
kernel accumulation difference is invisible. Model level follows suit:

| gate | bf16-Triton | r4d bf16 |
|---|---|---|
| 32K teacher-forced KLD (mean / top1) | 0.0318 / 0.987 | 0.031 / 0.988 |
| 32K needle 5 configs (50/25/75% @32K, 25/50% @21K) | 0/5 (refusals) | 0/5 (refusals) |
| fp16 baseline needle | -- | 5/5 pass |

Same KLD, same refusal mode, without any libr4d code in the loop.
Conclusion: the needle failure is bf16-intrinsic (bf16 I/O alone breaks
long-context retrieval vs our fp16 path at rel 4e-4). Stage A/B stay opt-in;
no adapter bug to fix.

## Error vs context length (2026-10-30)

Synthetic 27B geometry, Tq=2048 last chunk, K-blocked fp32 SDPA reference;
rel_F vs the matching-dtype oracle:

| Tk | fp16 | bf16-I/O | r4d bf16 | r4d fp8m2 |
|---|---|---|---|---|
| 32K | 2.9e-4 | 1.71e-3 | 1.76e-3 | 4.65e-2 |
| 64K | 3.1e-4 | 1.71e-3 | 1.76e-3 | 4.64e-2 |
| 128K | 3.5e-4 | 1.72e-3 | 1.76e-3 | 4.61e-2 |
| 256K | 5.1e-4 | 1.74e-3 | 1.77e-3 | 4.63e-2 |

Real 27B at 64K (captured last-chunk Q + full window): fp16 2.2e-4,
bf16 1.69e-3, r4d bf16 1.66e-3, r4d fp8m2 6.6e-2 -- consistent with 32K and
synthetic. Answer: the error does NOT grow with context length (flat to
256K; fp16 max-rel creeps 4.3e-4 to 6.8e-4, rel_F flat). 128K capture was
infeasible in this harness (the 128K paged pool + trellis staging OOMs 32 GB;
documented limit, same wall as the 128K full-forward logits).

## License handling

libr4d and vllm-mxfp4 have no license (all rights reserved). This repo
contains only the loader/adapter and the build helper script. The script
clones `https://codeberg.org/StillDeadcode/libr4d.git` at pin `b9e42ab` and
fetches `r4d_radiance_extras_rx10.patch` from
`https://github.com/GGZ14/vllm-mxfp4` at `d616242` (verified byte-identical
to the patch the exploration build used) at build time on the user's machine.
Never commit their sources, patches, or binaries here. Read their code for
API/layout details; never modify those clones in place (copy to scratch to
patch/build). `r4d.so` path via env; never overwrite a loaded `.so` (temp +
rename in the script). A second script, `rocm_tools/build_r4d_gdn2.sh`, does
the same for the exact-scan module (same libr4d pin, the
`moe-gdn2/radiance_gdn2_vs_v050.patch` from the same vllm-mxfp4 ref, with
md5 checks before and after).
