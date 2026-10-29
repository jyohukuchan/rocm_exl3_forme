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

head_dim 256, exactly 6 q heads per kv head, causal, paged fp16 cache with
`q_len > 16` and appended K/V length equal to `q_len`, attended length a
multiple of 16, no window/sinks/softcap/alibi/per-(token,head) scales, no
varlen/quant-cache, gfx1201 only. Q is converted fp16->bf16 per call
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

## Accuracy (R9700, kernel-level; 27B gates pending)

- Stage A bf16 KV: max-rel ~3.7e-3 vs fp16 Triton (exploration) and 5.3e-3 in
  the loader check -- quantization-free, no gate beyond unit tests.
- Stage B fp8 KV: mode 0 (f16 legs) ~3.3-5.0e-2, mode 2 (O_PV8) ~4.5e-2
  (fp8 KV quantization noise included); modes 1/3 (O_QK8) ~0.5 -- kernel
  error, end-task gate required. Stage B serving use waits for: 27B cases-v1
  KLD vs fp16 attention + a 32K+ long-context check (passkey/needle).
  Modes 1/3: KLD will be measured and reported; the user decides.

## Speed (R9700, layer ms; Hessian co-resident, A/B interleaved)

| Tk | Tq | r4d bf16 (ctypes) | r4d bf16 (loader) | fp16 retuned (default) |
|---|---|---|---|---|
| 32768 | 2048 | 15.51 (1.56x) | 18.11 (1.34x) | 24.2 (1.00x) |

Exploration sweep (ctypes): bf16kv 1.5-1.6x, fp8kv+f16 legs 1.7-1.85x,
mode 1 ~1.9-2.0x, mode 2 ~2.0-2.1x, mode 3 ~2.6-2.8x over retuned fp16
(current-default ratios are ~1.55x larger). 27B end-to-end at 8K/32K/128K
pending VRAM (Hessian until ~05:00 JST); projections in doc/r4d_exploration.md.

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
