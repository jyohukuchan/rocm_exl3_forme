#!/usr/bin/env python3
"""Sweep every (K, codebook) quantizer shape with int8-grid s0.

Asserts decoded outputs land exactly on the s0 grid (fp16 round-trip
deviation 0.0) for every supported K (1..7) and codebook. Small tiles keep
temp buffers modest (K=1 needs 16 rows: 16*256*32768 int16 = 256 MiB).
"""
import sys

import torch

from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles


def main() -> int:
    K = int(sys.argv[1])
    cb = sys.argv[2]
    kw = {"mcg": True} if cb == "mcg" else ({"mul1": True} if cb == "mul1" else {})
    torch.manual_seed(7)
    rows = 16 if K == 1 else 64
    t = (torch.randn(rows, 256, device="cuda:0") * 3).contiguous()
    s0 = float(t.abs().amax().item()) / 127.0
    q1, _ = quantize_tiles(t, {"K": K, "int8_s0": s0, **kw})
    dev = ((torch.round(q1 / s0) * s0).half().float() - q1).abs().amax().item()
    flag = "OK" if dev == 0.0 else "OFFGRID"
    print(f"K{K} {cb}: {flag} dev={dev:.1e}", flush=True)
    return 0 if dev == 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
