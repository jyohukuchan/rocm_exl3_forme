#!/usr/bin/env python3
"""Exercise normal model_init/Generator usage with a long-context retrieval prompt.

Accepts the standard --lsa-projections, -cq, -tp and GPU split arguments. No
special cache or model directory is needed. --repeat also exercises prefix reuse.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from exllamav3 import Generator, Job, GreedySampler, model_init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_init.add_args(parser, default_cache_size=16384, default_chunk_size=2048, add_draft_model_args=True)
    parser.add_argument("--input-tokens", type=int, default=8192)
    parser.add_argument("--new-tokens", type=int, default=24)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    args.autosplit_max_batch_size = max(args.autosplit_max_batch_size, args.batch)
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args, progress=False)
    generator = Generator(model, cache, tokenizer, max_batch_size=args.batch,
                          max_chunk_size=args.chunk_size, cpu_cache_size=int(args.cpu_cache_size * 1024**3),
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=args.num_draft_tokens or (draft_model.caps.get("default_draft_size", 4) if draft_model else 0),
                          dynamic_draft=args.dynamic_draft)
    prefix = tokenizer.encode("<|im_start|>user\nRead the document and remember its secret code.\n", encode_special_tokens=True)
    filler = tokenizer.encode("An ordinary record describes the weather, the road, and the nearby trees.\n")
    needle = tokenizer.encode("\nThe secret code is 739182. Remember this exact number.\n")
    suffix = tokenizer.encode("\nWhat is the secret code? Reply with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n", encode_special_tokens=True)
    count = args.input_tokens - prefix.shape[1] - needle.shape[1] - suffix.shape[1]
    if count < 0:
        raise ValueError("--input-tokens is shorter than the retrieval question")
    repeated = filler.repeat(1, (count + filler.shape[1] - 1) // filler.shape[1])[:, :count]
    ids = torch.cat((prefix, repeated[:, :count // 2], needle, repeated[:, count // 2:], suffix), 1)
    report = {"model": args.model_dir, "lsa": config.lsa, "cache_quant": args.cache_quant,
              "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "input_tokens": ids.shape[1], "batch": args.batch, "mtp": generator.mtp_draft,
              "cache_tokens": args.cache_size, "tp": model.loaded_tp, "runs": []}
    report["rope_scaling"] = config.rope_settings.rope_scaling
    report["draft_attention_window"] = args.draft_attention_window
    if model.loaded_tp:
        from exllamav3.modules.lsa import mp_lsa_stats
        report["tp_plan"] = model.plan
        report["loaded_stats"] = model.tp_worker_dispatch_wait_multi(model.active_devices, mp_lsa_stats, ())
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for repeat in range(args.repeat):
        jobs = [Job(ids.clone(), max_new_tokens=args.new_tokens, sampler=GreedySampler(), return_logits=True,
                    stop_conditions=config.eos_token_id_list)
                for _ in range(args.batch)]
        for job in jobs:
            generator.enqueue(job)
        completed, finite = [], True
        start = time.monotonic()
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                logits = result.get("logits")
                if logits is not None:
                    # Sampling masks the model's padded vocabulary with -inf.
                    finite = finite and bool(torch.isfinite(logits[..., :tokenizer.actual_vocab_size]).all())
                if result["stage"] == "streaming" and result.get("eos"):
                    completed.append({k: result.get(k) for k in (
                        "full_completion", "prompt_tokens", "new_tokens", "cached_tokens", "time_prefill", "time_generate",
                        "accepted_draft_tokens", "rejected_draft_tokens")})
                if result["stage"] == "prefill":
                    print(json.dumps({"prefill_progress": result.get("curr_progress"), "target": ids.shape[1],
                                      "elapsed_seconds": time.monotonic() - start}), flush=True)
        row = {"repeat": repeat, "seconds": time.monotonic() - start, "finite_logits": finite, "jobs": completed}
        report["runs"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if not finite or len(completed) != args.batch:
            raise RuntimeError("Generator did not complete all jobs with finite logits")
    if args.json_out:
        if model.loaded_tp:
            report["final_stats"] = model.tp_worker_dispatch_wait_multi(model.active_devices, mp_lsa_stats, ())
        report["cache_layer_types"] = sorted({type(layer).__name__ for layer in cache.layers.values()})
        report["tree_groups"] = sum(len(group.cached) for layer in cache.layers.values()
                                   if getattr(layer, "tree_store", None) is not None for group in layer.tree_store.groups)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    generator.clear_queue()
    model.unload()
    if draft_model is not None:
        draft_model.unload()


if __name__ == "__main__":
    main()
