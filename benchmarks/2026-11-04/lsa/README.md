# D-LSA Flat / Tree validation — 2026-11-04

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
- `r9700-tree-32k-k5v4.json` and `v620-tree-32k-k5v4.json`: official Qwen
  Tree settings (16 levels, nodes of 4, node cluster blocks 4096, beams 2048,
  start nodes 32768, FP8 node centroids). Each GPU retrieves correctly at
  32,768 input tokens and reuses 32,512 tokens on the repeated request. Both
  reports show 16 constructed node groups across the 16 attention layers,
  proving hierarchy construction rather than a Flat-only configuration.
- `v620-tp2-tree-rank.json`: the same Tree configuration on two V620s with
  RCCL and `query_heads=rank`. Correct retrieval and 32,512-token prefix reuse.
  Node groups live in workers, so the parent-side group counter is zero.
- `r9700-i8f8-kv.json` and `r9700-r4d-kv.json`: ordinary cache-selection
  arguments preserve `CacheLayer_lsa_i8f8` and `CacheLayer_lsa_r4d` storage,
  respectively. Correct 8K retrieval and prefix reuse. The r4d storage run
  does not require libr4d for the LSA Triton read.
- `r9700-pca-k5v4.json`: training-free public PCA map, same loader and K5/V4
  cache, correct 8K retrieval. PCA asset SHA256:
  `baaa28cb00b4378b3d33e02dc971e2781482d5c4d45e29e9140143bdf7832e65`.

Reproduction uses `rocm_tools/lsa_smoke.py` and standard model loading flags;
see `doc/lsa_attention.md`. Native extensions were the existing
`/work/lib-context-mr-bounds` (V620) and `/work/lib-r9700-90k` (R9700), with
PyTorch `2.12.0+rocm7.2`. No new C++ extension build is needed for LSA.

The two `*-tests.txt` files record the 26 numerical/cache/TP tests on each GPU
family. These include independent gathered-attention references for FP16 and
K6/V4, causal full-budget coverage, Ward/reference agreement, page overwrite,
prefix copy/restore, native page rotation, and TP index reduction including
headless ranks. The real-model K5/V4 jobs cover the standard Generator path.
Additional checks compare multilevel Tree beams to the vendored official CPU
oracle on CPU and GPU, for BF16 and FP8 nodes, including page visibility,
uncovered frontiers and start-level selection. They also cover node-cache
overwrite/restore/eviction, cache-memory accounting, INT8/FP8 and r4d native
page rotation, companding, r4d dense fallback storage, rank-local queries and
quantized dispatch validation. The quantized dispatch branch fix is committed;
the tests no longer rely on a pre-existing uncommitted edit to that file.

Timings are single smoke observations including warmup/compilation effects,
not controlled benchmarks or speedup comparisons. These retrieval cases do
not establish broad accuracy, image understanding, speculative generation
quality or ultra-long-context limits.
