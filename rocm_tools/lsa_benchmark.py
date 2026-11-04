#!/usr/bin/env python3
"""Controlled EXL3 dense / D-LSA Flat / Tree speed and retrieval comparison.

One warmup plus repeated cold-prefix requests at each length. Identical inputs,
weight/KV precision, cache capacity, chunks and output lengths across modes.
Raw JSON is checkpointed after every request. First-response logits are saved
outside the repository for a paired KL comparison; timing includes one GPU clone.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant

TREE = dict(top_level=16, node_bucket_size=4, node_cluster_block_size=4096,
            beam_width=2048, start_nodes=32768, node_centroid_precision="fp8")


def digest(ids):
    return hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def source_info(harness_path=None):
    root = Path(__file__).resolve().parents[1]
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "harness_sha256": hashlib.sha256(Path(harness_path or __file__).read_bytes()).hexdigest()}


def hardware_snapshot():
    selected = os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",")[0].replace("GPU-", "")
    records = []
    for card in Path("/sys/class/drm").glob("card[0-9]"):
        dev = card / "device"
        uid = dev / "unique_id"
        if not uid.exists() or (selected and uid.read_text().strip() != selected):
            continue
        record = {"pci": dev.resolve().name, "unique_id": uid.read_text().strip()}
        for name in ("power_dpm_force_performance_level", "gpu_busy_percent"):
            path = dev / name
            if path.exists():
                record[name] = path.read_text().strip()
        for hwmon in (dev / "hwmon").glob("hwmon*"):
            for name in ("temp1_input", "power1_average", "power1_cap", "freq1_input"):
                path = hwmon / name
                if path.exists():
                    record[name] = path.read_text().strip()
        records.append(record)
    return records


def bank(tokenizer, maximum):
    rng = random.Random(19011)
    weather = ["rain", "sunshine", "snow", "fog", "a strong wind"]
    places = ["the harbor", "the library", "the station", "the workshop", "the orchard"]
    records = []
    for i in range(maximum // 15 + 2048):
        records.append(f"Record R{i:06d}: access code {rng.randrange(100000, 1000000)}. "
                       f"The observer reported {rng.choice(weather)} near {rng.choice(places)}; "
                       f"the inventory contained {rng.randrange(10, 9999)} ordinary items.\n")
    tokens = tokenizer.encode("".join(records))
    if tokens.shape[1] < maximum:
        raise RuntimeError("Filler bank is shorter than the requested context")
    return tokens


def prompt(tokenizer, filler, length, trial):
    # The warmup is also a middle-depth retrieval sample; timed trials cover both ends.
    depths = [0.5, 0.1, 0.9, 0.3, 0.7]
    depth = depths[trial % len(depths)]
    expected = str(random.Random(73000 + trial).randrange(100000, 1000000))
    prefix = tokenizer.encode(f"<|im_start|>user\nArchive version {trial}. Read the archive, locate record "
                              "TARGET-ALPHA, and remember its six-digit access code.\nBEGIN ARCHIVE\n",
                              encode_special_tokens=True)
    needle = tokenizer.encode(f"\n\nRecord TARGET-ALPHA: the access code is {expected}.\n\n")
    suffix = tokenizer.encode("\nEND ARCHIVE\nWhat is the access code of record TARGET-ALPHA? "
                              "Reply with its six digits only.<|im_end|>\n<|im_start|>assistant\n"
                              "<think>\n</think>\n\n", encode_special_tokens=True)
    count = length - prefix.shape[1] - needle.shape[1] - suffix.shape[1]
    if count < 0:
        raise ValueError("Context is shorter than the retrieval instructions")
    insert = int(count * depth)
    ids = torch.cat((prefix, filler[:, :insert], needle, filler[:, insert:count], suffix), 1)
    assert ids.shape[1] == length
    return ids, expected, depth


def reset_prefix(generator):
    assert not generator.num_remaining_jobs()
    generator.pagetable.reset_page_table()
    if generator.recurrent_cache is not None:
        generator.recurrent_cache.clear()
        generator.recurrent_cache.current_size = 0


def kl_metrics(reference, candidate):
    p, q = reference.double().log_softmax(-1), candidate.double().log_softmax(-1)
    return {"first_response_kl_dense_to_mode": float((p.exp() * (p - q)).sum()),
            "first_response_top1_matches_dense": bool(reference.argmax() == candidate.argmax())}


def token_timing(observations):
    intervals = [b[1] - a[1] for a, b in zip(observations, observations[1:])]
    ordered = sorted(intervals)
    result = {"token_observations": observations,
              "token_interval_p50_seconds": statistics.median(intervals),
              "token_interval_p95_seconds": ordered[(len(ordered) * 95 + 99) // 100 - 1],
              "token_interval_max_seconds": max(intervals)}
    if len(observations) > 8:
        result["decode_after_8_tps"] = ((observations[-1][0] - observations[7][0]) /
                                         (observations[-1][1] - observations[7][1]))
    return result


def trial_run(generator, model, tokenizer, ids, new_tokens, capture):
    reset_prefix(generator)
    capture.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    job = Job(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, sampler=GreedySampler(),
              stop_conditions=model.config.eos_token_id_list)
    start = time.perf_counter()
    first = last = None
    observations = []
    completed = None
    generator.enqueue(job)
    progress = 0
    while generator.num_remaining_jobs():
        previous = job.new_tokens
        results = generator.iterate()
        if job.new_tokens > previous:
            last = time.perf_counter()
            observations.append([job.new_tokens, last - start])
            if first is None:
                first = last
        for result in results:
            if result["stage"] == "error":
                error = result.get("error")
                if isinstance(error, BaseException):
                    raise error
                raise RuntimeError(str(error))
            if result["stage"] == "prefill":
                current = result.get("curr_progress", 0)
                if current >= progress + 16384:
                    progress = current
                    print(json.dumps({"progress_tokens": current, "context": ids.shape[1]}), flush=True)
            if result["stage"] == "streaming" and result.get("eos"):
                completed = result
    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    if completed is None or completed["new_tokens"] != new_tokens or completed["cached_tokens"]:
        raise RuntimeError("Benchmark requires fixed output length and zero prefix cache hits")
    logits = capture["logits"].float().cpu().flatten()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite first-response logits")
    text = completed["full_completion"]
    row = {k: completed[k] for k in ("prompt_tokens", "new_tokens", "cached_tokens", "time_prefill", "time_generate")}
    row.update(wall_seconds=wall, observed_ttft_seconds=first - start,
               observed_decode_tps=(new_tokens - 1) / (last - first),
               engine_prefill_tps=ids.shape[1] / completed["time_prefill"],
               engine_decode_tps=new_tokens / completed["time_generate"],
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               response_prefix=text[:256], response_sha256=hashlib.sha256(text.encode()).hexdigest())
    row.update(token_timing(observations))
    return row, logits


def summarize(rows):
    measured = [r for r in rows if not r["warmup"]]
    result = {"timed_runs": len(measured), "retrieval_passes": sum(r["retrieval_correct"] for r in rows),
              "retrieval_cases": len(rows)}
    for key in ("engine_prefill_tps", "observed_decode_tps", "observed_ttft_seconds", "peak_allocated_bytes"):
        values = [r[key] for r in measured]
        median = statistics.median(values)
        result[key] = median
        result[key + "_spread"] = (max(values) - min(values)) / median if median else 0.
    if "first_response_kl_dense_to_mode" in measured[0]:
        result["first_response_kl_dense_to_mode"] = statistics.mean(r["first_response_kl_dense_to_mode"] for r in measured)
    for key in ("token_interval_p50_seconds", "token_interval_p95_seconds", "token_interval_max_seconds", "decode_after_8_tps"):
        if key in measured[0]:
            result[key] = statistics.median(r[key] for r in measured)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--modes", nargs="+", choices=["dense", "flat", "tree"], default=["dense", "flat", "tree"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 32768, 65536, 131072])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--cache-size", type=int)
    parser.add_argument("--cache-quant", default="5,4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Replace only the selected modes in an existing paired report")
    args = parser.parse_args()
    bits = [int(x) for x in args.cache_quant.split(",")]
    if len(bits) not in (1, 2):
        raise ValueError("Cache quantization takes one bit width or a K,V pair")
    if args.repeats < 1 or args.new_tokens < 2:
        raise ValueError("Benchmark requires a timed repeat and at least two generated tokens")
    if len(bits) == 1:
        bits *= 2
    maximum = max(args.lengths)
    capacity = args.cache_size or max(args.chunk_size, math_ceil256(maximum + args.new_tokens + 256))
    if capacity < maximum + args.new_tokens or capacity % 256:
        raise ValueError("Cache capacity must cover the longest request and be page-aligned")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(source=source_info(), model=args.model, projections=args.projections,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  arch=torch.cuda.get_device_properties(0).gcnArchName,
                  cache_quant=bits, cache_tokens=capacity, chunk_size=args.chunk_size,
                  new_tokens=args.new_tokens, timed_repeats=args.repeats,
                  warmups_per_length=1, complete=False, modes={})
    report["environment"] = {k: os.environ.get(k) for k in (
        "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_ENABLE_SDMA", "PYTHONPATH", "OMP_NUM_THREADS",
        "EXL3_BC_ATTN", "EXL3_QC_STAGING", "EXL3_ATTN_INT8FP8", "EXL3_R4D_ATTN")}
    report["model_config_sha256"] = hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest()
    report["projection_sha256"] = hashlib.sha256(Path(args.projections).read_bytes()).hexdigest()
    if args.resume:
        previous = json.loads(output.read_text())
        for key in ("model_config_sha256", "projection_sha256", "gpu", "torch", "cache_quant", "cache_tokens",
                    "chunk_size", "new_tokens", "timed_repeats"):
            if previous[key] != report[key]:
                raise ValueError(f"Resume changes the comparison setting {key}")
        expected = {(n, t) for n in args.lengths for t in range(args.repeats + 1)}
        for mode, block in previous["modes"].items():
            block.setdefault("source", previous["source"])
            if mode not in args.modes and {(r["length"], r["trial"]) for r in block["rows"]} != expected:
                raise ValueError(f"Cannot preserve incomplete mode {mode}")
            if mode not in args.modes:
                block["summaries"] = {str(n): summarize([r for r in block["rows"] if r["length"] == n])
                                      for n in args.lengths}
        previous.setdefault("resumes", []).append({"source": report["source"], "replaced_modes": args.modes})
        previous.pop("error", None)
        previous.pop("error_memory", None)
        previous["complete"] = False
        report = previous
    report["requested_lengths"] = args.lengths
    model = generator = cache = config = tokenizer = None
    try:
        for mode in args.modes:
            settings = None if mode == "dense" else {"projections": args.projections, **(TREE if mode == "tree" else {})}
            config = Config.from_directory(args.model, lsa=settings)
            model = Model.from_config(config)
            cache = Cache(model, capacity, layer_type=CacheLayer_quant, k_bits=bits[0], v_bits=bits[1], max_batch_size=1)
            model.load(device="cuda:0", max_chunk_size=args.chunk_size, progressbar=False)
            tokenizer = Tokenizer.from_config(config)
            filler = bank(tokenizer, maximum)
            generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=args.chunk_size,
                                  enable_defrag=False)
            capture = {}
            original = model.forward
            def observed_forward(*a, **kw):
                logits = original(*a, **kw)
                if "logits" not in capture:
                    capture["logits"] = logits[:, -1, :tokenizer.actual_vocab_size].detach().clone()
                return logits
            model.forward = observed_forward
            mode_report = dict(lsa=settings, source=source_info(), cache_layer_types=sorted({type(layer).__name__ for layer in cache.layers.values()}),
                               cache_storage_bytes=sum(int(layer.storage_size()) for layer in cache.layers.values()),
                               rows=[], summaries={})
            report["modes"][mode] = mode_report
            for length in args.lengths:
                for trial in range(args.repeats + 1):
                    ids, expected, depth = prompt(tokenizer, filler, length, trial)
                    sha = digest(ids)
                    print(json.dumps({"mode": mode, "length": length, "trial": trial, "warmup": trial == 0}), flush=True)
                    before = hardware_snapshot()
                    row, logits = trial_run(generator, model, tokenizer, ids, args.new_tokens, capture)
                    row["hardware_before"], row["hardware_after"] = before, hardware_snapshot()
                    first_code = re.search(r"\d{6}", row["response_prefix"])
                    row.update(length=length, trial=trial, warmup=trial == 0, depth=depth, expected_code=expected,
                               prompt_sha256=sha, retrieval_correct=first_code is not None and first_code.group() == expected)
                    reference_path = output.parent / (output.stem + f"-dense-{length}-{trial}.pt")
                    if mode == "dense":
                        torch.save({"prompt_sha256": sha, "logits": logits}, reference_path)
                        row.update(first_response_kl_dense_to_mode=0., first_response_top1_matches_dense=True)
                    elif reference_path.exists():
                        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
                        if reference["prompt_sha256"] != sha:
                            raise RuntimeError("Dense reference and candidate prompts differ")
                        row.update(kl_metrics(reference["logits"], logits))
                    mode_report["rows"].append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    write_json(output, report)
                group = [r for r in mode_report["rows"] if r["length"] == length]
                mode_report["summaries"][str(length)] = summarize(group)
                write_json(output, report)
            model.forward = original
            generator.clear_queue()
            model.unload()
            del original, observed_forward
            model = generator = cache = config = tokenizer = None
            capture.clear()
            gc.collect()
            torch.cuda.empty_cache()
        report["complete"] = True
    except BaseException as error:
        report["error"] = repr(error)
        try:
            report["error_memory"] = {"free_total": torch.cuda.mem_get_info(),
                                      "fraction": torch.cuda.get_per_process_memory_fraction(),
                                      "allocated": torch.cuda.memory_allocated(),
                                      "reserved": torch.cuda.memory_reserved()}
        except Exception:
            pass
        raise
    finally:
        write_json(output, report)
        if generator is not None:
            generator.clear_queue()
        if model is not None:
            model.unload()


def math_ceil256(value):
    return (value + 255) // 256 * 256


if __name__ == "__main__":
    main()
