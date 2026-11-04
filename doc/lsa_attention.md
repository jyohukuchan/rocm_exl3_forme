# Qwen3.8-27B: D-LSA on ROCm

Qwen3.8-27B can use the published **D-LSA Flat** indexer with ordinary EXL3
loading and generation. V620 (gfx1030) and R9700 (gfx1201) use the same Triton
implementation. FP16 KV and the existing independently configurable K/V 2–8 bit
caches are supported. Model weight quantization is independent of LSA.

The model's 16 full-attention layers use LSA. Its 48 Gated DeltaNet layers,
vision tower, output gates and native RoPE settings retain their usual behavior.
The released target-layer maps do not replace an independent MTP draft head.
No GPU-name check, minimum context requirement, forced KV bitrate, forced YaRN
factor, or additional serving approval/enable flag is imposed.

## Use with existing commands

Download the official map outside the source checkout:

```bash
mkdir -p ~/datapool/ai_models/lsa
curl -fL https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt \
  -o ~/datapool/ai_models/lsa/qwen38_27b.pt
```

SHA256: `24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b`.

Add the map to a command using the standard `model_init` arguments:

```bash
python examples/chat.py -m /path/to/Qwen3.8-27B-exl3 \
  --lsa-projections ~/datapool/ai_models/lsa/qwen38_27b.pt \
  -cs 32768 -cq 5,4
```

Omit `-cq` for FP16 KV. GPU splits, `-tp`, its existing backend choice, batch
size and CPU page-cache arguments continue to work. The underscore spellings
`--lsa_projections` and `--lsa_config` are also accepted.
For the existing ROCm tensor-parallel implementation, use `-tp --tp_backend nccl`.

Optional indexer settings use JSON, for example:

```bash
--lsa-config '{"top_k":2048,"top_p":1024,"window_size":512,"sink_size":4}'
```

Defaults are `top_k=4096`, `top_p=1024`, `window_size=512`, `sink_size=4`,
`pool_size=4`, `bucket_size=8`, `bucket_cap=16`. The map determines `index_dim`
(64 for the release). `index_dim`, if supplied, must match the map.
`pool_size: null` disables pooling, as in the upstream configuration. Pool,
bucket and window sizes must nest, and the Ward cap must be at least twice
the average bucket size, as in the official implementation.

## Python API and model config

```python
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.cache import CacheLayer_quant

config = Config.from_directory(
    "/path/to/Qwen3.8-27B-exl3",
    lsa={"projections": "/path/to/qwen38_27b.pt", "top_k": 4096},
)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=32768,
              layer_type=CacheLayer_quant, k_bits=5, v_bits=4)
model.load()
tokenizer = Tokenizer.from_config(config)
generator = Generator(model, cache, tokenizer)
```

Alternatively, add this block to the checkpoint's existing `config.json`:

```json
"lsa": {"projections": "qwen38_27b.pt", "top_k": 4096}
```

This lets applications that call `Config.from_directory` directly use LSA
without new application-specific arguments. A relative path in the model
config is relative to that model directory; the CLI map path is relative to
the working directory. A `text_config.lsa` block is accepted too. An explicit
`lsa=None` API override disables a checkpoint's LSA block. No LSA block/map
means the model uses its original attention.

## Implementation and memory

The integration follows the official SGLang fork's hybrid full/linear-attention
dispatch and the LSA Qwen backend's key-and-query index. The four KV heads'
192 position-free channels are concatenated, and each group's six query heads
are summed. Per-layer 768×64 query/key maps project these to unit index vectors.
Closed window blocks are pooled and spherical-Ward merged. Query scores choose
buckets; their members are expanded to long-range keys plus the sink/window,
without duplicates or future positions. Short contexts use dense attention
whenever the LSA budgets provably cover every causal key.

The index is built from normalized Q/K before KV cache packing. Quantized
attention reads only selected packed K/V rows, with online dequantization.
Nonzero cache companding uses the existing dequantization fallback. Selection
does not require a second unquantized copy of the original KV cache.

The projected keys and page versions are page-major cache tensors, so partial
prefix copies, defragmentation and CPU paging carry them with KV. A bounded
workspace caches derived buckets by page identity/version. Overwrites,
speculative rewinds and restored prefixes cannot reuse buckets of different
keys. Page-version storage respects the native 16-byte rotation alignment.
TP projects local heads before summing index partials and normalizing; headless
ranks participate in the same collective. KV remains sharded, while the small
index planes and maps are replicated and accounted for by the TP allocator.

LSA reduces the KV **read**, not the stored KV length. This implementation uses
FP32 projected keys and centroids. At the default settings the per-token index
storage is approximately 296 bytes per full-attention layer (about 148 MiB
for 32K across all 16 layers), plus small maps/page metadata and temporary
selection workspaces. K/V quantization still reduces the original KV storage.

LSA attention currently takes the eager path, rather than the existing fused
C++ attention CUDA graph. Other modules retain their existing dispatch. The
user does not need graph-disable flags. This port implements D-LSA Flat;
D-LSA Tree and training N-LSA are separate methods and are not implemented here.

## Validation

Run numerical/cache checks in the usual ROCm environment:

```bash
python -m pytest -q tests/test_lsa.py tests/test_tp_export_attention.py
```

These cover GQA projection, full-budget causal coverage, Ward/expansion versus
the reference, FP16/quantized sparse reads versus an independent gather-softmax,
page overwrite/copy/restore/rotation, and TP projection reduction including
headless participants. The GPU tests run on either V620 or R9700.

Exercise the standard loader and Generator with a retrieval prompt:

```bash
python rocm_tools/lsa_smoke.py -m /path/to/Qwen3.8-27B-exl3 \
  --lsa-projections /path/to/qwen38_27b.pt -cq 5,4 \
  -cs 36864 --input-tokens 32768 --repeat 2 --json-out /tmp/lsa.json
```

`--batch 2` exercises concurrent requests; size `-cs` for their total tokens.
`--repeat 2` exercises prefix reuse. Existing `-ccs` enables CPU paging.
The smoke records actual-vocabulary finite logits, the retrieved number,
prefix hits and timings. It is a smoke check, not a broad accuracy benchmark
or an end-to-end speedup claim. Raw local validation results are summarized in
`benchmarks/2026-11-04/lsa/`.

The official map was distilled with YaRN factor 40. The loader does not
override the checkpoint's RoPE: use its existing settings, or explicitly
configure YaRN when reproducing the paper. Accuracy across RoPE/weight/KV
quantization combinations still needs evaluation for the intended workload.

## Upstream and license

- [LSA implementation](https://github.com/Prefrontal-AI/latent-sparse-attention), revision `97b82d695b74ca341965a16bbc2a951c8b0c567a`.
- [Official SGLang fork](https://github.com/Prefrontal-AI/sglang/tree/lsa-v0.1.0), revision `042d955ba31b962911f63b114c2fb245effa86f2`.
- [Qwen backend](https://github.com/Prefrontal-AI/latent-sparse-attention/tree/main/src/lsa/sglang/models/qwen3_5).
- [Released projections](https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1).

Selected upstream Ward/expansion kernels and their reference are included under
`exllamav3/vendor/lsa/`, with Apache-2.0 LICENSE and NOTICE. SGLang and CuTeDSL
are not runtime dependencies of this EXL3 integration.
