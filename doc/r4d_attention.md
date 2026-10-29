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
not fragmentation). Chunked-path 32K TTFTs were measured but are not quoted:
repetitive prompts let the pagetable dedupe pages (less real work for the
baseline) and same-process repeats hit prefill-skip artefacts, so neither
side is a clean A/B. 128K has the same logits wall. Updated projections from
measured kernel speedups x the 8K/128K/1M shares (attn 9%/69%/95%):

| config | 8K | 128K | 1M |
|---|---|---|---|
| attn Stage A loader (~1.4x @32K) | ~1.03x | ~1.25x | ~1.37x |
| attn fp8kv+f16 (~1.8x) | ~1.06x | ~1.44x | ~1.7x |
| attn mode 2 (~2.0x) | ~1.07x | ~1.5x | ~1.9x |
| attn mode 3 (~2.7x) | ~1.1x | ~1.8x | ~2.5x |
| + GDN r4d (~2x @32K) | ~1.2x | -- | -- (attn-bound at 1M) |

Rough (kernel speedups x 8K/128K/1M attn shares 9%/69%/95%; GDN share past
8K unmeasured, so no GDN projection past 8K).

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
rename in the script).
