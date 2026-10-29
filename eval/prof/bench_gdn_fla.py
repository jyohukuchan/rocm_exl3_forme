#!/usr/bin/env python3
"""Our current GDN core cost (vendored fla chunk_gated_delta_rule) at 27B shapes.
B=1, q/k 16 heads x128, v 48 heads x128, bf16, prefill (no history/state save).
Interleaved rounds, median/min/max.
"""
import sys
sys.path.insert(0, "/src")
sys.path.insert(0, "/scratch/prebuilt")
import torch
torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} {props.gcnArchName}", flush=True)
# import package init first (ext), then the vendored kernel lazily like the wrapper does
import exllamav3.ext  # noqa: F401  (ensures prebuilt .so path, no rebuild)
from exllamav3.vendor.fla import chunk_gated_delta_rule
import statistics as st

Hk, Hv, D = 16, 48, 128
torch.manual_seed(0)

def build(T):
    q = torch.randn((1, T, Hk, D), device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn((1, T, Hk, D), device="cuda:0", dtype=torch.bfloat16)
    v = torch.randn((1, T, Hv, D), device="cuda:0", dtype=torch.bfloat16)
    g = torch.randn((1, T, Hv), device="cuda:0", dtype=torch.float32) * 0.5
    beta = torch.rand((1, T, Hv), device="cuda:0", dtype=torch.bfloat16)
    return q, k, v, g, beta

def time_fn(fn, iters=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2], min(ts), max(ts)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--t", nargs="+", type=int, default=[2048, 8192, 32768, 131072])
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
for T in a.t:
    q, k, v, g, beta = build(T)
    fn = lambda: chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                        initial_state=None,
                                        output_final_state=False,
                                        use_qk_l2norm_in_kernel=True)
    fn(); torch.cuda.synchronize()
    acc = []
    for _ in range(a.rounds):
        med, lo, hi = time_fn(fn, a.iters)
        acc.append(med)
    acc.sort()
    print(f"T={T}: fla-chunk-gdn={acc[len(acc)//2]:.2f}ms [min={min(acc):.2f} max={max(acc):.2f}]", flush=True)
    del q, k, v, g, beta
    torch.cuda.empty_cache()
