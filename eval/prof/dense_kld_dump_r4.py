#!/usr/bin/env python3
"""Dump full-vocab FP32 logits for KLD cases-v1.json with r4 prefill.

Same output protocol as dense_kld_dump.py (candidate dir + manifest.json),
but supported prefill attention runs through the Round 4 kernel via
r4_eval_hook (int8 mode, BM128+BN64). All other layers/shapes stay fp16,
so KLD vs the fp16 dump isolates the r4-covered layers (same methodology
as the round-1 EXL3_ATTN_INT8FP8 KLD).

Usage: dense_kld_dump_r4.py MODEL_DIR CASES_JSON OUT_DIR [PAD_W] [--mode int8|fp8]
Env: HIP_VISIBLE_DEVICES, EXL3_HGEMM_FP16_OUT=1 (fair fp16 GEMM baseline).
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, "/src")
sys.path.insert(0, "/src/eval/prof")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

import torch

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("cases_json")
ap.add_argument("out_dir")
ap.add_argument("pad_w", nargs="?", default=248320, type=int)
ap.add_argument("--mode", default="int8")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
torch.cuda.set_device(0)

from r4_eval_hook import install_r4_hook
info = install_r4_hook(mode=args.mode)

from exllamav3.model import Config, Model

config = Config.from_directory(args.model_dir)
model = Model.from_config(config)
model.load(device=torch.device("cuda:0"))
print(f"loaded {args.model_dir}", flush=True)

cases = json.loads(open(args.cases_json).read())["cases"]
manifest = {"schema": "exl3-kld-dump-v1", "engine": f"rocm-exl3-r4-{args.mode}",
            "model": args.model_dir,
            "int8_prefill": False,
            "cases": []}
with torch.inference_mode():
    for c in cases:
        ids = torch.tensor([c["token_ids"]], dtype=torch.long, device="cuda:0")
        logits = model.forward(ids, {}).float().cpu().numpy()[0]
        assert (logits != logits).sum() == 0, f"nonfinite in {c['id']}"
        n, w = logits.shape
        if w < args.pad_w:
            import numpy as np
            logits = np.pad(logits, ((0, 0), (0, args.pad_w - w)),
                            constant_values=-1e30)
        fn = c["id"] + ".f32"
        logits.astype("<f4").tofile(os.path.join(args.out_dir, fn))
        h = hashlib.sha256(
            json.dumps(c["token_ids"], separators=(",", ":")).encode()).hexdigest()
        manifest["cases"].append({
            "id": c["id"], "token_count": n,
            "input_token_ids_sha256": h,
            "positions": list(range(n)),
            "shape": [n, logits.shape[1]], "dtype": "f32",
            "layout": "row-major", "nonfinite_count": 0,
            "logits_file": fn, "raw_path": fn})
        print(f"{c['id']}: rows={n} w={logits.shape[1]}", flush=True)
with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote", args.out_dir, flush=True)
print(f"r4 hook: calls={info['calls']} layers={info['layers']}", flush=True)
