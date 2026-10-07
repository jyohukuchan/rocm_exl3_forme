# JEV native EXL3 support

The verified mixed-precision checkpoint is publicly available at
[jyohukuchan/JEV-27B-VL-exl3-4bpw](https://huggingface.co/jyohukuchan/JEV-27B-VL-exl3-4bpw).
All 44 release files were checked against the upload manifest; see
[publication verification](jev_hf_release/PUBLICATION.md).

JEV-27B-VL combines an unchanged Qwen language/vision backbone for generation
(System 2) with a runtime decision LoRA (System 1). Merging the adapter into the
main checkpoint would change ordinary generation. Reading unadapted vocabulary
logits or applying a sampling filter would not implement its trained decisions.

This fork keeps both systems in one model, selects LoRA per forward, and starts
each request with fresh KV and recurrent state. The dedicated server serializes
GPU work, including client cancellation, so two systems cannot share mutable
state. JEV-27B-VL EXL3 inference is verified on a single V620 and a single
R9700. `--gpu-split 28,28` selects single-process layer split, but the early
unquantized V620 pair probe needed `AMD_SERIALIZE_KERNEL=3` during loading;
asynchronous source layer-split loading still needs investigation. The verified
quantized single-GPU runs need no serialized-kernel setting. Tensor-parallel
LoRA is explicitly rejected by the library.

## Conversion and precision

The completed JEV-27B-VL conversion uses:

```sh
ulimit -n 65536
EXL3_ROCM_MLP_RANGE_BALANCE=0 python convert.py \
  -i /models/safetensors/JEV-27B-VL -w /work/build-jev27-vl-4bpw \
  -o /work/models/JEV-27B-VL-exl3-4bpw \
  -b 4 -hb 6 -mb 6 -vb 16 -ss 1024
```

The language trunk uses 4bpw and the normal generation head and MTP component
use 6bit. The vision tower is unquantized because visual decisions are part of
JEV's intended use. The decision adapter remains unquantized, evaluated in
FP32. The converter detects `adapter_vllm/decision_head.json`, copies the
adapter/calibration, and saves 264 relevant source BF16 vocabulary rows in
`decision_rows.safetensors`. System 1 computes those exact source rows plus the
head LoRA in FP32. It bypasses the quantized generation head entirely.

This is a 4bpw **trunk** setting, not a claim that the whole pack is exactly
4bpw: embeddings, auxiliary weights, exact decision rows and LoRA are additional.
Calibration is 250 rows × 2048 tokens, with FP32 Hessians. The resulting pack
has 17 weight shards, 2,426 indexed tensors, and totals 17,445,036,040 bytes
(16.247 GiB), including the unmerged adapter. Its weight tensors occupy
16,966,676,548 bytes: 4.886 effective bits per original parameter; the entire
pack including metadata/adapter is 5.024. The 400 quantized language projections
are 4bit, the generation head and eight MTP projections are 6bit, and the 333
vision tensors remain BF16. Embeddings, norms and recurrent controls keep their
original storage precision. MTP is packaged but is not used by the JEV server.

The downloaded source is `autotrust/JEV-27B-VL` revision
`f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc`. All 18 source shards and the adapter
match their Hugging Face LFS SHA256 values. The output's 39 files have a saved
SHA256 manifest; the adapter/calibration match the source byte-for-byte and the
264 decision rows match the original BF16 head exactly.

On the validation host, the canonical model directory is
`/home/homelab1/datapool/ai_models/safetensors/JEV-27B-VL-exl3-4bpw`.
It shares weight inodes with the validated `/work/models/...` pack, avoiding a
second 17GB copy. Do not modify shared weight files in place.

## HTTP server

Install `fastapi`, `uvicorn` and `transformers` in the existing ROCm/PyTorch
EXL3 environment. Use the native extension compiled for the GPU being tested.

```sh
python -m rocm_tools.jev_server \
  -m /models/JEV-27B-VL-exl3-4bpw \
  --context 16384 --chunk-size 1024 --port 3960
```

The launcher raises its own open-file soft limit to 65536 when permitted; the
JEV adapter needs more descriptors than the common 1024 default. Library callers
should set `ulimit -n 65536` before starting their process. The default bind is loopback. `--api-key` enables Bearer authentication on POST
routes. This launcher disables the existing MLP metadata range-balance policy,
which rejects runtime LoRA. The original-scale JEV route is verified on both
GPUs below; this does not establish general R9700 coverage for other models.

`POST /v1/decide` accepts the official bare-v1 fields:

```json
{"kind":"noul","state":"Tokyo is the capital of Japan.","question":"Is this correct?"}
```

Kinds are `noul` (false/true), `score` (0..5), and `choice` (2..256 string
options). It returns the complete normalized probability vector, chosen index,
usage and timing. No token is generated for System 1. Bias is applied before
its calibrated per-kind temperature. For choices beyond the 16 trained labels,
`single`, `permute` and `tournament` follow the source model's extension
strategies; the default is `single`.

`thinking: "auto"` invokes System 2 when the largest System 1 probability is
below `threshold` (default 0.8). `thinking: "on"` always invokes it; the default
is `off`. `think_budget` defaults to 1024 tokens. The base model reasons over
the state/question/options, then its contextual answer-letter probabilities
are read without generation and mixed 50:50 with System 1, as in the official
server. `return_reasoning`, `debug` and `reasoning_effort` (low/medium/xhigh) are
supported. Scores have no adaptive-thinking route. A bounded reasoning budget
can force readout before reasoning completes; `finished_within_budget` reports
that condition. Adaptive HTTP execution and probability mixing are verified;
reasoning quality has only been screened on small examples. Source-model
calibration is not an EXL3 calibration guarantee.

Images may appear in state as a list of strings and `{"image":"data:image/png;base64,..."}`
or standard `image_url` parts. Native image embeddings and MRoPE are used for
both systems. Up to 16 images are accepted, with a 262144-pixel processing cap;
URLs are currently limited to data URLs.

`POST /v1/systemone` follows llama.cpp's TypeSafe batch request/answer shape:

```json
{"state":"The user needs a billing refund.","questions":{
  "route":{"type":"choice","instructions":"Choose the appropriate team.",
           "criteria":{"billing":"Payments and refunds","technical":"Software problems"}},
  "refund":{"type":"noul","instructions":"Does the user request a refund?"}
}}
```

Answers map back to the original criteria keys. Choice confidence and score
confidence use the formulas in llama.cpp's `server-decision.cpp`. JEV score
criteria must contain its six trained levels; incompatible scales are rejected.
Questions are evaluated sequentially. Shared-prefix caching across questions
is not implemented yet.

`POST /v1/chat/completions` offers ordinary text/vision System 2 generation
with `max_tokens`, `temperature`, and `chat_template_kwargs` containing
`enable_thinking`/`reasoning_effort`. This small JEV server currently returns
non-streaming responses. Tools, response-format constraints and top-p/top-k
controls are explicitly rejected; use the existing general EXL3 server when
those features are required for ordinary generation.

### Generation-only checkpoints

`--generation-only` skips the JEV adapter, calibration and decision rows so the
same serialized chat/session server can screen ordinary EXL3 VLM checkpoints:

```sh
python -m rocm_tools.jev_server -m /models/qwen3.5-2b-exl3 \
  --generation-only --model-name qwen35-2b --context 32768 --cache-quant 5,4
```

The info protocol is `exl3-vl-chat-v1`; `/v1/decide` and `/v1/systemone` return
400 because an ordinary checkpoint has no trained decision head. Chat responses
report `input_images` and `image_embedding_tokens` to check that images reached
the vision encoder. This is generation support, not an emulation of JEV System 1.

Session delimiters and later user fragments come from the checkpoint's own chat
template. Cached assistant token IDs remain unchanged when a template strips
historical thinking prefixes. Non-MRoPE models do not call the Qwen MRoPE helper;
bidirectional image spans are kept atomic across prefill chunks. Checkpoints
without `generation_config.json` use the tokenizer's EOS ID.

On 2026-10-08, R9700 tests exercised Qwen3.5-2B EXL3 with quantized 5/4 KV:
six Minecraft PNGs, actual image embeddings, and a two-turn continuing image
conversation that recalled the initial instruction. Gemma4-26B-A4B-it EXL3
4.10bpw also completed those probes with FP16 KV on the diagnostic reference
paths below. Its default fused projection/expert paths abort with an HSA
exception on this host; ordinary Gemma4 ROCm inference is **not yet validated**.
The reference-path timing must not be presented as optimized MoE performance.

`--vision-max-pixels N` controls image detail for preprocessors exposing a pixel
budget (including Qwen3.5). The default remains 262144; 524288 keeps an 800x600
Minecraft screenshot close to its original size. It cannot enlarge the
checkpoint's configured limit or go below its minimum. Architectures with a
soft-token budget instead reject non-default pixel overrides rather than
silently ignore them. Vision loading reserves scratch for the configured patch
budget. `/v1/decide/info` reports the configured budget, and stateless/session
chat responses report `image_preprocessing` with original/processed dimensions
and actual embedding-token counts where the architecture exposes them.

On R9700, Qwen3.5-2B processed the same 800x600 PNG at 576x416 (234 embedding
tokens) with the default, and at 800x608 (475 tokens) with 524288. All six fixed
PNG requests and a 16-turn live Minecraft session completed in both settings.
Qwen3.5-4B BF16 also completed the six higher-detail image requests. This checks
image transport/detail and execution; it does not establish gameplay quality.

For fault isolation only, `EXL3_VLM_UNFUSED_PROJECTIONS=1` disables Q/K/V and
gate/up projection bundles after loading. Pair it with `EXL3_BC_ATTN=0` and
`EXL3_QKV_SLICE=0` before importing EXL3 to disable graph-captured attention.
`EXL3_VLM_REFERENCE_MOE=1` runs routed experts through individual quantized
Linear projections. Neither switch changes the weights or substitutes a model
answer. `EXL3_VLM_TRACE=1` logs image/prefill stages and synchronizes traced
modules in the first language block; diagnostic synchronization perturbs timing.

### Append-only chat sessions

For a continuing System 2 conversation, `POST /v1/chat/sessions` accepts a
non-empty `system` string and returns `session_id`. Send only each new user
content (text/image parts) to `POST /v1/chat/sessions/{session_id}` with
`turn` starting at 1, `max_tokens`, and `temperature`. Delete that URL to close.
The initial instruction is transmitted once. The server retains live quantized
KV, GDN recurrent state, exact generated token IDs and image embeddings. It
prefills only the appended user fragment and commits each assistant turn's
closing delimiter before accepting the next turn.

Session setup also accepts `enable_thinking: true` (default false). This default
is applied to the initial template and later user fragments. An append request
may override `enable_thinking` for that turn, while preserving the actual
cached assistant token IDs. `enable_thinking_used` reports the selected mode.
The completion budget includes both generated reasoning and the final answer;
truncation still invalidates the session. The raw text retains the checkpoint's
native reasoning delimiters so a controller can record reasoning separately
and execute only the final JSON. A real two-image Qwen3.5-35B-A3B session on the
R9700 diagnostic reference paths recalled its initial marker/key on turn two
and reported 844 cached tokens. This validates continuing thinking generation,
not optimized MoE kernel performance or game-playing ability.

This permits an input planner to reason over a new screenshot and a following
GUI-location query to use an empty-thinking prefix in the same session. The
location request need not resend the screenshot, initial bindings or old text.
A live Qwen3.5-35B-A3B Minecraft trial completed16 control steps with15 location
queries; every location turn reported thinking false, zero new images, and a
nonzero cached prefix. The controller still verified the session/turn and input
revision before applying any input. No crafting table was acquired in that
trial; the integration check is not a claim of better game understanding.

Qwen templates with the native `</think>` token additionally accept
`reasoning_budget` on an append request. The overall `max_tokens` limit still
includes reasoning and the final answer. If the reasoning budget is reached,
the server commits the native closing delimiter through single-token decode,
then lets the model generate its final answer. Those formatting token IDs are
kept in the same KV/GDN history; no state is rewound and no action JSON is
substituted. `reasoning_budget_reached` identifies a forced boundary. Templates
without that supported delimiter reject the budget override. An R9700 2B image
probe with budget8 returned a complete JSON answer and consistent cached IDs.

Responses include `session`, cumulative prompt length,
`usage.prompt_tokens_details.cached_tokens`, and `usage.prefilled_tokens`.
Turns must be consecutive; repeated/out-of-order turns return 409 without
advancing the state. A truncated answer or partial inference failure invalidates
the session rather than trying to rewind destructive GDN updates.

One session exclusively owns the current single-batch cache. Stateless chat,
System 1 and additional session creation return 409 while it is active, avoiding
KV/adapter contamination without allocating a second copy. Info/health queries
remain available. Idle sessions expire after 180 seconds on the next request.
Context overflow requires a new session; automatic history compaction is not
implemented. Session generation currently keeps thinking disabled.

Two real image-bearing turns recalled a word supplied only in turn 1: turn 2
reused 288 tokens and prefilled 262 new tokens, with both image embeddings
retained. Three continuing Minecraft trials also kept one session each, but
did not collect a log or craft a table: the model repeatedly maintained attack
despite reporting no visible cracking. Cache continuity is therefore verified;
these trials do not demonstrate successful low-level gameplay.

### Quantized KV cache and context capacity

`--cache-quant k_bits,v_bits` selects independent 2..8-bit key/value caches.
Omitting it retains the original FP16 cache. These widths apply to the
full-attention KV layers, not to GDN recurrent state. `/v1/decide/info` reports
the requested policy, actual layer classes/widths/resident tensor bytes, and
`model_max_context`, so an accepted flag is not mistaken for actual quantization.

On 2026-10-07, JEV's 4bpw pack completed near-limit image/text prompts on one
R9700 using K5/V4, chunk size 1024, and the default 262144-pixel image cap:

| Context | Actual input tokens | KV bytes including scales | Peak torch allocation | Prefill + answer |
| ---: | ---: | ---: | ---: | ---: |
| 65,536 | 65,472 | 1.25 GiB | 15.96 GiB | 62.0 s |
| 131,072 | 131,008 | 2.50 GiB | 17.47 GiB | 163.5 s |
| 262,144 | 262,080 | 5.00 GiB | 20.48 GiB | 492.6 s |

Each prompt included one real 800x600 Minecraft PNG plus synthetic text
padding, followed by an instruction to answer `OK`; all three returned `OK`.
The final run left 10.79 GiB of whole-device free VRAM. Peak torch allocation
is not total driver/device usage. The timings exclude model loading and image
encoding. This establishes execution/capacity, not long-context task accuracy.
The model's configured maximum is 262144; no RoPE or weight changes were made.
All 16 full-attention cache layers were audited as `CacheLayer_quant` with
K5/V4; the 48 GDN recurrent layers remained at their existing precision.

The host-specific probe checks the R9700 UUID and single-device visibility:

```sh
python -m rocm_tools.jev_context_probe --context 262144 --cache-quant 5,4 \
  --image /path/to/game.png --output /path/to/new-probe-directory
```

The probe raises its file-descriptor soft limit to 65536, audits the live
cache tensors, and saves exact prompt length/hashes, generated output, memory,
timing and source hashes. It preserves failed attempts. Run each capacity
point in a separate process with the existing JEV server stopped to avoid
loading two copies of the model on the same GPU.

## Validation tools

`rocm_tools/jev_quality.py` creates eight frozen text/vision cases, collects an
unquantized BF16-body/FP32-head HF+PEFT reference, collects an EXL3 candidate,
checks System 1 → System 2 isolation and image generation, and compares full
probability distributions. It reports KL, max probability difference and top-1
agreement per case. These are a small regression screen, not broad model
quality or long-context validation.

The BF16 reference disables Transformers' optional allocator warmup because
a single ~26GiB allocation failed on V620. The collector uses a complete per-module device map, and supports a CPU-staged
`--reference-placement hybrid` alternative. No parent/root placement overrides
are used, because Accelerate can move all child weights to that parent device
during hook setup. All source weights and forward math remain unchanged. The oracle explicitly uses the upstream Torch GDN/conv
functions and SDPA MATH because the installed FLA BF16 dot kernel cannot compile
on gfx1030. Original failed logs must be retained beside successful runs.

Sources: [model and reference server](https://huggingface.co/autotrust/JEV-27B-VL/tree/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc),
[llama.cpp decision API](https://github.com/ggml-org/llama.cpp/blob/7049ff0cbeb1f5ead231de4522af6b75d8d773c0/tools/server/server-decision.cpp).

## Hardware evidence (2026-10-06)

Warmed System 1 latency on these GPUs is measured separately with serial HTTP
requests and the official browser demo; see [decision latency report](jev_latency_20261006/README.md).

Frozen regression cases cover Japanese/English binary decisions, six-level
scores, 4/16/256-way choices, two visual decisions, and four less decisive
examples. Both quantized single-GPU candidates match the BF16 oracle's top
choice on all 12 cases.

| Candidate | Basic top-1 / mean KL | Sensitive top-1 / mean KL | Maximum probability difference | Peak allocated VRAM |
|---|---|---|---|---|
| R9700, gfx1201 | 8/8 / 0.00085924 | 4/4 / 0.00039115 | 0.039731 | 15.60 GiB |
| V620, gfx1030 | 8/8 / 0.00083092 | 4/4 / 0.00038679 | 0.038927 | 15.52 GiB |

KL is `D_KL(reference || candidate)`. The largest probability shift is the
basic score example. This bounded screen supports retaining the 4bit trunk
with the higher precision components above; it does not establish broad
accuracy, probability calibration or long-context quality. Both candidates
produce `東京` before and after System 1 with identical token output and
` Red square.` for the image-generation probe.

Real HTTP checks on **both quantized GPUs** returned 200 for health/info,
bare-v1 decisions, TypeSafe noul/score/choice batch evaluation, image chat and
8-token adaptive thinking. They assert normalized probabilities, correct
criteria-key mapping, highest arithmetic grade at 5, and the 50:50 S1/S2 mix.
The 13 targeted CPU regressions cover sliced/strict LoRA, exact head selection,
calibration, TypeSafe mapping, inference mode and adaptive mixing.

Additional R9700 runtime checks select identifier 173 correctly among 256
options using both `permute` (four model requests) and `tournament` (17
requests), with normalized probabilities. High-confidence `thinking: "auto"`
also skips System 2 as intended. These extension checks are saved separately
from the 12-case BF16 comparison and do not expand that comparison's scope.

The tested environments use PyTorch `2.12.0+rocm7.2`, context 16,384,
chunk size 1,024, FP16 KV cache, and one GPU per model. R9700 uses
`PYTHONPATH=/work/lib-r9700-opt:/src`; V620 uses
`PYTHONPATH=/work/lib-expert-placement:/src` and `HSA_ENABLE_SDMA=0`.
`EXL3_ROCM_MLP_RANGE_BALANCE=0` applies to both. Neither quantized run sets
`AMD_SERIALIZE_KERNEL`. Native binary and source hashes are in the candidate
JSON metadata. The container's older default `/work/lib` was not used.

For the existing host containers, reproduce the collector with:

```sh
docker exec -e PYTHONPATH=/work/lib-r9700-opt:/src \
  -e EXL3_ROCM_MLP_RANGE_BALANCE=0 rocm-exl3-r9700-conv \
  python -m rocm_tools.jev_quality --backend exl3 \
  --model /work/models/JEV-27B-VL-exl3-4bpw \
  --cases /work/runs/jev-20261006/cases.json \
  --output /work/runs/jev-20261006/quantized-r9700.json
```

For V620 replace the container with `rocm-exl3-v620-pair` and native library
with `/work/lib-expert-placement:/src`. Use `cases-sensitive.json` for the
independent four-case screen. The server uses the same environment and model
path; HTTP checks ran on loopback inside each bridge container.

Evidence JSON, cases, integrity manifests and the HTTP verification script
are checked in under [jev_validation/](jev_validation/README.md). Large
conversion logs and source diagnostics remain at
`/home/homelab1/datapool/rocm-exl3-rdna2/runs/jev-20261006`.

The converter completed every decoder layer and printed `All done`, with no
excluded/nonfinite calibration rows. Its shell wrapper recorded exit 127
because editing that running wrapper shifted Bash's post-child read offset.
The original error/status is retained, alongside a separate artifact audit;
the conversion was not rerun or relabeled successful based on its exit code.
All shard headers/index entries, integrity checks and GPU runs passed.

### Unquantized baseline

The BF16 oracle and native unquantized V620 pair each completed all eight
decisions. Native-vs-oracle mean KL is 1.8139607e-6, maximum probability
difference .00103208, with 8/8 identical top choices. Base generation before
and after a decision is identical (`東京`); image chat returns ` red square`.
The native run used source weights cast to FP16, FP32 LoRA/exact head, the live
V620 native binary and serialized GPU operations. These are unquantized
baseline results, separate from the quantized single-GPU results above.

Real unquantized HTTP tests also exercised all decision/chat endpoints and an 8-token
adaptive-thinking budget. TypeSafe score descriptions are appended to the
question with their numeric mapping, preserving the trained 0..5 option lines.
A correct `2 + 2 = 4` answer then receives its highest probability at grade5.
