#!/usr/bin/env python3
"""Probe layer-split load of Qwen3-30B BF16: print where the split lands."""
import sys

sys.path.insert(0, "/src")
import torch
from exllamav3.model import Config, Model

config = Config.from_directory("/work/models/qwen3-30b-a3b-bf16")
model = Model.from_config(config)
for i, n in model.load_gen(use_per_device=[32, 32], max_chunk_size=512,
                            progressbar=True, verbose=True):
    if i % 8 == 0:
        print(f"module {i}/{n}", flush=True)
print("LOAD OK")
for d in (0, 1):
    free, total = torch.cuda.mem_get_info(d)
    print(f"cuda:{d} free {free/2**30:.2f} total {total/2**30:.2f} GiB", flush=True)
