# Qwen3.8-27B 4bpw / LSA / TP=2 / MTP on V620×2

The target runs with tensor parallelism across the two V620s. The MTP model
runs on one GPU and borrows the target's embedding and output head. KV uses
K4/V4, batch 1. The target input window in this profile is 1,835,008 tokens
(1.75 Mi tokens); cache capacity is 1,835,520 to leave room for generation.

## Model

Use `turboderp/Qwen3.8-27B-exl3`, branch `4.00bpw`, pinned revision
`113cf7ab958054860e43fb7f3063b1af19171095`. It contains the 4.00bpw target,
6-bit output head, and 4-bit MTP weights. The downloaded safetensors were
checked against the Hub's LFS SHA256 metadata. Download/source manifests
remain beside the model. The model weights are not stored in this Git repo.

## Memory and work distribution

Global-query LSA uses a single index owner per attention layer. Every TP rank
contributes its Q/K projection, the owner selects the causal indices, and
broadcasts those indices for each rank to attend over its local KV heads.
The FP32 index precision is unchanged. The full index is no longer duplicated
on both ranks. Explicit `query_heads="rank"` keeps its separate, rank-local
selection semantics and therefore does not use shared-index ownership.

The TP allocator accounts for index storage and decode tables as a separate
single-owner component, restricted to ranks that own KV heads. It measures
available memory after loading the MTP model, so ownership and the remaining
splittable components can adapt to the MTP model's residency. In the 1.75Mi
capacity check, the MTP GPU owned six LSA indices and the other GPU ten; both
still held two KV heads in every target attention layer.

Short TP query batches capture the owner's flat selection and each rank's
local gathered attention in HIP graphs. Projection collectives and index
broadcast stay outside those graphs. This differs from the single-GPU graph,
which captures the whole attention block without TP collectives.
The serialized stages share a graph pool and capture stream within each TP
worker. Separate streams would keep sort scratch allocations resident per
layer even with a common pool, exhausting memory at the full window.

`--draft-device` chooses the MTP GPU. `--draft-attention-window` optionally
bounds the drafter's left attention window. It leaves the target's full
context and verification unchanged. The drafter can propose differently;
proposals are accepted only after the normal target verification. Omitting
the flag retains full draft attention. The profile below uses 65,536 tokens
to limit the single-GPU drafter's attention work. It still allocates the
full draft KV pool; this option reduces attention work, not KV residency.

## Reproduce

The model, LSA, TP, cache and draft flags use the ordinary `model_init`
interface, including `examples/chat.py`; no verification-specific loader is
required. `lsa_smoke.py` below only constructs the test prompt and records
the result.

Expose only the two V620s. On this ROCm environment, use the existing RCCL
backend (`-tpb nccl`). `EXL3_QC_STAGING=0` reads quantized KV directly rather
than expanding a long MTP prefix into a full FP16 temporary. The allocator
setting avoids the large-allocation failure observed with expandable
segments on this host. These are runtime settings, not context restrictions.

```bash
PYTHONPATH=/work/lib-context-mr-bounds:/src \
OMP_NUM_THREADS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
EXL3_QC_STAGING=0 \
python rocm_tools/lsa_smoke.py \
  -m /work/models/qwen38-27b-exl3-4bpw \
  --lsa-projections /work/models/lsa/qwen38_27b.pt \
  -tp -tpb nccl -mtp --draft-device 0 \
  --draft-attention-window 65536 \
  -cq 4,4 -cs 1835520 -chunk_size 4096 \
  --rope-scaling '{"rope_type":"yarn","factor":40.0,"original_max_position_embeddings":262144}' \
  --input-tokens 1835008 --new-tokens 16 --repeat 1 \
  --json-out /work/runs/lsa-tp2-mtp-175mi-full.json
```

The YaRN settings match the released LSA map's official model card. The
checkpoint's on-disk config is not rewritten. For a quick functional check,
use `--input-tokens 8192` while keeping the full cache capacity. Such a check
proves allocation and short-input generation, not full-window processing.
A complete 1.75Mi prefill is a separate long-running check.

Before repeating a full prefill after changes to graph memory, run the same
model/cache flags with `rocm_tools/lsa_tp_memory_probe.py`, replacing the
smoke-specific input/output flags with `--past-tokens 1835007 --json-out
/work/runs/lsa-tp2-mtp-175mi-memory.json`. It exercises actual target and MTP
forwards at the end of a full-sized cache with synthetic cached history,
including 1/5-token graph slots and recurrent rollback. Its output explicitly
labels that history synthetic; it cannot establish natural-prefill correctness
or long-context retrieval quality.

The smoke test records finite actual-vocabulary logits, output text, draft
acceptance/rejection, TP placement, physical GPU headroom, and graph replay
counts. It checks one synthetic number-retrieval prompt, not broad long-text
quality or general throughput.

## Observed result

The actual 1,835,008-token input completed with no prefix reuse, returned the
correct number `739182`, and exited with code 0. Returned generation logits
were finite. Target TP=2 and single-device MTP were active; MTP accepted two
and rejected eighteen proposed tokens. Both TP ranks replayed their attention
graphs (160 total), with 80 owner-selection graph replays.

Prefill took 8,993.48 seconds (149.9 minutes; 204.04 input tokens/s). Seven new
tokens took 8.16 seconds including first graph capture. These timings belong
to one functional request, not an MTP on/off speed comparison. The number
retrieval case does not establish general long-context quality.

Final target-worker allocation was 28.63 GiB on the MTP GPU and 28.97 GiB on
the other GPU; driver-reported free memory was 1.70/1.19 GiB. Physical VRAM use
sampled during prefill peaked at 31.98/31.56 GiB. The optional MTP window was
65,536; the target used the full input and both KV pools retained full capacity.
See the [result and raw artifacts](../benchmarks/2026-11-05/lsa-tp2-mtp/README.md).
