#!/usr/bin/env python3
"""Greedy generation from explicit prompt token ids through Generator (paged cache).

  exl3_gen_ids.py --model DIR --ids-file FILE --new N [--kv fp16|q54]
Prints the generated ids on one line (comma separated). INT8 is selected by the
EXL3_INT8_* environment variables.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--kv", default="fp16", choices=["fp16", "q54"])
    a = ap.parse_args()

    import torch
    from exllamav3 import Config, Model, Cache, Generator, GreedySampler, Tokenizer, Job
    from exllamav3.cache.quant import CacheLayer_quant

    ids = [int(x) for x in open(a.ids_file).read().split()]
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    cache_len = max(4096, (len(ids) + a.new + 512) // 256 * 256)
    if a.kv == "q54":
        cache = Cache(model, max_num_tokens=cache_len, layer_type=CacheLayer_quant, k_bits=5, v_bits=4)
    else:
        cache = Cache(model, max_num_tokens=cache_len)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    job = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=a.new,
              sampler=GreedySampler(), stop_conditions=[])
    generator.enqueue(job)
    out = []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming" and "token_ids" in r and r["token_ids"].numel():
                out += [int(t) for t in r["token_ids"].flatten().tolist()]
    print("IDS=" + ",".join(map(str, out[:a.new])))


if __name__ == "__main__":
    main()
