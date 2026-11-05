#!/usr/bin/env python3
"""Random search of 24-bit multiplier constants for the mul-hash codebook x = (s * C) mod 2^32,
f = sum of the 4 bytes of x (decoded on the GPU with one full-rate v_mul_u32_u24 + one dot4)."""
import random
import sys
import torch
import viterbi_sim as vs

K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
torch.manual_seed(0)
w_fast = torch.randn(12, 192, device=vs.DEV)
w_ref = torch.randn(32, 512, device=vs.DEV)
alphas = [1.02, 1.06] if K >= 3 else [1.0, 1.08]
rd = 2.0 ** (-2 * K)


def cb(c):
    x = (vs.S * c) & 0xFFFFFFFF
    return vs.norm(sum(vs.bytes_of(x, 4)).float())


base = vs.best_mse(vs.norm(vs.cb_mul1()), K, w_ref, alphas)[0]
print(f"K={K}: mul1 (0x83DCD12D) reference MSE {base:.5f} ({base / rd:.3f} x bound)", flush=True)
random.seed(1)
cands = [0xDCD12D] + [random.randrange(1, 1 << 24) | 1 for _ in range(N)]
scored = []
for c in cands:
    scored.append((vs.best_mse(cb(c), K, w_fast, alphas)[0], c))
scored.sort()
print("top candidates (fast screen) re-evaluated on the reference set:")
for _, c in scored[:6]:
    m = vs.best_mse(cb(c), K, w_ref, alphas)[0]
    print(f"  C=0x{c:06X}  MSE {m:.5f}  ({m / rd:.3f} x bound, {m / base:.3f} x mul1)", flush=True)
m = vs.best_mse(cb(0xDCD12D), K, w_ref, alphas)[0]
print(f"lower 24 bits of mul1's constant (0xDCD12D): MSE {m:.5f} ({m / base:.3f} x mul1)")
spread = [s for s, _ in scored]
print(f"screen spread: best {spread[0]:.5f} median {spread[len(spread) // 2]:.5f} worst {spread[-1]:.5f}")
