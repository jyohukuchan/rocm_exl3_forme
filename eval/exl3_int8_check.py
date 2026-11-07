#!/usr/bin/env python3
"""EXL3-INT8 end-to-end check on a dense model (Qwen3.8 27B) with a KV cache.

Two-chunk prefill so the second chunk attends to the (possibly quantized) KV
cache written by the first, then greedy decode. Saves last-chunk logits
(fp16) and decoded ids; compare two runs with --compare.

  run:     exl3_int8_check.py --model DIR --kv fp16|q54 --out OUT.npz [--device 0]
           (INT8 is selected with EXL3_INT8_PREFILL=1 EXL3_INT8_FIXED_GRID=1
            EXL3_INT8_MIN_ROWS=256 in the environment)
  compare: exl3_int8_check.py --compare A.npz B.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_gen(args):
    import torch
    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.device))
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler
    from exllamav3.cache.quant import CacheLayer_quant

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache_len = (args.chunk1 + args.chunk2 + args.decode + 512) // 256 * 256
    if args.kv == "q54":
        cache = Cache(model, max_num_tokens=cache_len, layer_type=CacheLayer_quant, k_bits=5, v_bits=4)
    else:
        cache = Cache(model, max_num_tokens=cache_len)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    text = open(args.text, encoding="utf-8").read()
    ids = tokenizer.encode(text)[:, :args.chunk1 + args.chunk2]
    prompt = tokenizer.decode(ids)[0]
    out = generator.generate(prompt=prompt, max_new_tokens=args.decode, sampler=GreedySampler(),
                             completion_only=True, add_bos=False)
    open(args.out, "w", encoding="utf-8").write(out)
    print(f"saved {args.out}: {out[:200]!r}")


def run(args):
    import torch
    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.device))
    from exllamav3 import Config, Model, Cache, Tokenizer, GreedySampler
    from exllamav3.cache.quant import CacheLayer_quant

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    total = args.chunk1 + args.chunk2 + args.decode + 1
    cache_len = (total + 255) // 256 * 256
    if args.kv == "q54":
        cache = Cache(model, max_num_tokens=cache_len, layer_type=CacheLayer_quant, k_bits=5, v_bits=4)
    else:
        cache = Cache(model, max_num_tokens=cache_len)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(config)

    text = open(args.text, encoding="utf-8").read()
    ids = tokenizer.encode(text)[:, :args.chunk1 + args.chunk2]
    assert ids.shape[-1] == args.chunk1 + args.chunk2, "text too short"

    block_table = torch.arange(cache_len // 256, dtype=torch.int32).unsqueeze(0)

    def mk(past, rs):
        return {"attn_mode": "flash_attn", "cache": cache, "block_table": block_table,
                "cache_seqlens": torch.tensor([past], dtype=torch.int32), "recurrent_states": rs}

    with torch.inference_mode():
        p = mk(0, None)
        model.forward(ids[:, :args.chunk1], p)
        rs = p.get("recurrent_states")
        logits = model.forward(ids[:, args.chunk1:], mk(args.chunk1, rs)).float()
        out = [int(torch.argmax(logits[0, -1]))]
        last = torch.tensor([[out[0]]], dtype=torch.long)
        pos = args.chunk1 + args.chunk2
        for _ in range(args.decode - 1):
            lg = model.forward(last, mk(pos, rs))
            nxt = int(torch.argmax(lg[0, -1]))
            out.append(nxt)
            last = torch.tensor([[nxt]], dtype=torch.long)
            pos += 1
    nf = int((~torch.isfinite(logits)).sum())
    np.savez(args.out, logits=logits[0].cpu().numpy().astype(np.float16), ids=np.array(out),
             prompt_last=ids[0, -1].item())
    print(f"saved {args.out} nonfinite={nf} decoded={tokenizer.decode(torch.tensor([out]))[0][:200]!r}")


def compare(a, b):
    import torch
    A, B = np.load(a), np.load(b)
    la = torch.from_numpy(A["logits"].astype(np.float32))
    lb = torch.from_numpy(B["logits"].astype(np.float32))
    pa = torch.log_softmax(la, -1)
    pb = torch.log_softmax(lb, -1)
    kl = (pa.exp() * (pa - pb)).sum(-1)
    agree = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    n = int((A["ids"] == B["ids"]).sum())
    first = next((i for i, (x, y) in enumerate(zip(A["ids"], B["ids"])) if x != y), None)
    print(f"KL(a||b) mean={kl.mean():.5f} max={kl.max():.5f}  top1 agree={agree:.4f}  "
          f"decode ids equal {n}/{len(A['ids'])} first_diff={first}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--kv", default="fp16", choices=["fp16", "q54"])
    ap.add_argument("--out")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--text", default=os.path.join(os.path.dirname(__file__), "eval_texts", "variable_man_mod_c1.txt"))
    ap.add_argument("--chunk1", type=int, default=1536)
    ap.add_argument("--chunk2", type=int, default=512)
    ap.add_argument("--decode", type=int, default=48)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--gen", action="store_true", help="greedy generation through Generator (paged cache)")
    a = ap.parse_args()
    compare(*a.compare) if a.compare else (run_gen(a) if a.gen else run(a))
