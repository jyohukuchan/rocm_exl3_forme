# D-LSA Flat validation — 2026-11-04

Target: the local `qwen38-27b-exl3-305-int8grid` Qwen3.8-27B pack, using its
unchanged default RoPE. The official per-layer projection asset has SHA256
`24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b`.
Settings are the integration defaults: 64-dimensional index, window 512,
sink 4, pool 4, bucket size/cap 8/16, top-p 1024, top-k 4096.

All recorded jobs returned `739182`, the number embedded halfway through a
repeated filler document. Actual-vocabulary generation logits were finite.
The sampler's deliberately masked padded-vocabulary logits are excluded.

- `v620-8k-k5v4.json`: one V620, 8,192 input tokens, K5/V4. Both requests
  retrieve correctly; repeat reuses 7,936 prompt tokens.
- `v620-tp2-k5v4.json`: two V620s with the existing `nccl`/RCCL TP backend,
  8,192 input tokens, K5/V4. Both requests retrieve correctly; repeat reuses
  7,936 prompt tokens. The separate CUDA-native IPC TP backend is unavailable
  in the existing ROCm extension; this is not an LSA-specific restriction.
- `r9700-32k-k5v4.json`: one R9700, 32,768 input tokens, K5/V4. Both requests
  retrieve correctly; repeat reuses 32,512 prompt tokens.
- `r9700-batch2-k5v4.json`: one R9700, two concurrent 8,192-token requests,
  K5/V4, CPU cache tier enabled. Four jobs across two rounds retrieve correctly;
  both second-round jobs reuse 7,936 tokens. This run does not force CPU eviction.
- `r9700-fp16.json`: one R9700, 8,192 input tokens, ordinary FP16 KV. Correct
  retrieval without selecting cache quantization.

Reproduction uses `rocm_tools/lsa_smoke.py` and standard model loading flags;
see `doc/lsa_attention.md`. Native extensions were the existing
`/work/lib-context-mr-bounds` (V620) and `/work/lib-r9700-90k` (R9700), with
PyTorch `2.12.0+rocm7.2`. No new C++ extension build is needed for LSA.

The two `*-tests.txt` files record the 13 numerical/cache/TP tests on each GPU
family. These include independent gathered-attention references for FP16 and
K6/V4, causal full-budget coverage, Ward/reference agreement, page overwrite,
prefix copy/restore, native page rotation, and TP index reduction including
headless ranks. The real-model K5/V4 jobs cover the standard Generator path.

Timings are single smoke observations including warmup/compilation effects,
not controlled benchmarks or speedup comparisons. These retrieval cases do
not establish broad accuracy, image understanding, speculative generation
quality or ultra-long-context limits.
