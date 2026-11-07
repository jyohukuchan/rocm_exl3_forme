#!/usr/bin/env python3
"""Viterbi simulation of bitshift-trellis codebooks on i.i.d. Gaussian data.

State s (16 bit) -> weight f(s). Step: s' = ((s << K) | b) & 0xFFFF with K new bits b. Free start state
(sequences are long, so the missing tail-biting constraint is negligible). Reports MSE / variance at the best
scale, comparable with the T16 write-up (rate-distortion bound 2^(-2K); mul1 at K=2 ~0.064).
"""
import math
import sys
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
S = torch.arange(65536, dtype=torch.int64, device=DEV)


def bytes_of(x, n):
    return [((x >> (8 * i)) & 255) for i in range(n)]


def norm(f):
    f = f.double()
    return ((f - f.mean()) / f.std()).float()


def cb_mul1():
    x = (S * 0x83DCD12D) & 0xFFFFFFFF
    return sum(bytes_of(x, 4)).float()


def cb_mul1_int8():
    x = (S * 0x83DCD12D) & 0xFFFFFFFF
    b = sum(bytes_of(x, 4))
    return ((b - 509) >> 2).float()


def cb_t16(c=0x3A2F):
    x = (S * c) & 0xFFFF
    return sum(bytes_of(x, 2)).float()


def cb_t16_int8(c=0x3A2F):
    x = (S * c) & 0xFFFF
    return (((sum(bytes_of(x, 2))) - 255) >> 1).float()


def cb_mul16_4byte(c=0x3A2F):
    x = S * c  # < 2^32, one 24-bit multiply
    return sum(bytes_of(x, 4)).float()


def cb_mul16_3byte(c=0x3A2F):
    x = S * c
    return sum(bytes_of(x, 3)).float()


def cb_iid_table(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(65536, generator=g).to(DEV)


def cb_nibble_tables(scramble=True, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(4, 16, generator=g).to(DEV)
    s = ((S * 0x9E37) & 0xFFFF) if scramble else S
    return sum(t[i][(s >> (4 * i)) & 15] for i in range(4))


def cb_byte_tables(seed=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.randn(2, 256, generator=g).to(DEV)
    s = (S * 0x9E37) & 0xFFFF
    return t[0][s & 255] + t[1][s >> 8]


def cb_popcount(c=0x9E3779B1):
    x = (S * c) & 0xFFFFFFFF
    pc = torch.zeros_like(x)
    for i in range(32):
        pc += (x >> i) & 1
    return pc.float()


def cb_uniform_low8():
    return (S & 255).float()


CODEBOOKS = {
    "mul1 (32b mul, 4-byte sum)": cb_mul1,
    "mul1 int8 grid ((B-509)>>2)": cb_mul1_int8,
    "T16 (0x3A2F, 2-byte sum)": cb_t16,
    "T16 int8 grid ((lo+hi-255)>>1)": cb_t16_int8,
    "u24 mul 0x3A2F, 4-byte sum": cb_mul16_4byte,
    "u24 mul 0x3A2F, 3-byte sum": cb_mul16_3byte,
    "iid Gaussian table (reference)": cb_iid_table,
    "4 x 16-entry nibble tables (scrambled)": cb_nibble_tables,
    "2 x 256-entry byte tables (scrambled)": cb_byte_tables,
    "popcount of 32b product": cb_popcount,
    "low 8 bits of state (uniform)": cb_uniform_low8,
}


@torch.no_grad()
def viterbi_mse(f, K, alpha, w):
    """f: [65536] unit-variance table; w: [B, L] unit Gaussian. Returns mean squared error per weight."""
    B, L = w.shape
    tab = (alpha * f).float()
    nhi, nlo = 1 << K, 1 << (16 - K)
    cost = (w[:, 0:1] - tab[None, :]) ** 2
    for t in range(1, L):
        m, _ = cost.view(B, nhi, nlo).min(dim=1)
        cost = m.repeat_interleave(1 << K, dim=1) + (w[:, t:t + 1] - tab[None, :]) ** 2
    return cost.min(dim=1).values.mean().item() / L


def best_mse(f, K, w, alphas):
    return min((viterbi_mse(f, K, a, w), a) for a in alphas)


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    names = sys.argv[4].split("|") if len(sys.argv) > 4 else list(CODEBOOKS)
    torch.manual_seed(0)
    w = torch.randn(B, L, device=DEV)
    rd = 2.0 ** (-2 * K)
    print(f"K={K} L={L} B={B} device={DEV}  rate-distortion bound {rd:.5f}", flush=True)
    for name in names:
        f = norm(CODEBOOKS[name]())
        base = 1.0
        res = best_mse(f, K, w, [0.92, 0.96, 1.0, 1.04, 1.08, 1.12, 1.16] if K <= 2 else [0.96, 0.98, 1.0, 1.02, 1.04, 1.06])
        mse, alpha = res
        print(f"{name:42s} MSE {mse:.5f}  (x{mse / rd:.3f} of bound, alpha {alpha})", flush=True)


if __name__ == "__main__":
    main()
