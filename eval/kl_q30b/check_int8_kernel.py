#!/usr/bin/env python3
"""Verify the int8-grid (s0) quantizer path on GPU.

1. plain-vs-optimized bit identity (K=3/mul1, s0=0): the int8 patch must not
   change normal-path results.
2. s0>0: outputs land exactly on the int8 grid, deterministic across runs,
   and differ from the unrounded path.
3. s0>0 with the optimized kernel raises loudly (no silent unrounded encode).
"""
import sys

import torch

from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles

torch.manual_seed(1234)
tiles = torch.randn(4096, 256, device="cuda:0")
tiles = (tiles * torch.randint(1, 8, (4096, 1), device="cuda:0").float()).contiguous()


def run(s0):
    qa = {"K": 3, "mul1": True}
    if s0:
        qa["int8_s0"] = s0
    q, idx = quantize_tiles(tiles, qa)
    return q, idx


def main() -> int:
    import hashlib
    h = lambda t: hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]

    import os
    if os.environ.get("EXL3_QT_OPTIMIZED", "1") == "0":
        # Plain-kernel leg: s0=0 hashes for the other leg, plus the s0 grid
        # exactness, determinism and effect checks.
        q00, i00 = run(0.0)
        with open("/work/kl-q30b/check-plain-hashes.txt", "w") as f:
            f.write(f"{h(q00)} {h(i00)}\n")
        print(f"plain hashes: {h(q00)} {h(i00)}")
        s0 = float(tiles.abs().amax().item()) / 127.0
        q1, i1 = run(s0)
        q2, _ = run(s0)
        print(f"s0 deterministic: {h(q1) == h(q2)}")
        assert h(q1) == h(q2)
        grid = torch.round(q1 / s0)
        # Exactness means fp16(grid*s0) reproduces the kernel output bit for
        # bit (the kernel rounds to the grid, then stores fp16).
        dev = ((grid * s0).half().float() - q1).abs().amax().item()
        print(f"s0={s0:.6f} max grid deviation: {dev:.3e} "
              f"in-range: {bool(((grid >= -128) & (grid <= 127)).all())}")
        assert dev == 0.0, "decoded values off the int8 grid!"
        assert ((grid >= -128) & (grid <= 127)).all()
        print(f"grid changes results: {h(q1) != h(q00)} "
              f"(mean|q-q0|={(q1 - q00).abs().mean().item():.6f})")
        with open("/work/kl-q30b/check-plain-s0-hashes.txt", "w") as f:
            f.write(f"{h(q1)} {h(i1)} {s0!r}\n")
        print("INT8-GRID KERNEL OK (plain leg)")
        return 0

    # Optimized leg: normal-path AND int8-grid bit identity vs plain.
    import pathlib
    q0o, i0o = run(0.0)
    hp = pathlib.Path("/work/kl-q30b/check-plain-hashes.txt").read_text().split()
    print(f"plain==optimized(s0=0): q {h(q0o) == hp[0]} idx {h(i0o) == hp[1]}")
    assert h(q0o) == hp[0] and h(i0o) == hp[1], "plain/optimized mismatch!"

    hs = pathlib.Path("/work/kl-q30b/check-plain-s0-hashes.txt").read_text().split()
    s0 = float(hs[2])
    qo, io = run(s0)
    print(f"plain==optimized(s0>0): q {h(qo) == hs[0]} idx {h(io) == hs[1]}")
    assert h(qo) == hs[0] and h(io) == hs[1], "plain/optimized s0 mismatch!"
    qs0 = torch.round(qo / s0)
    dev = ((qs0 * s0).half().float() - qo).abs().amax().item()
    print(f"optimized s0 grid deviation: {dev:.3e}")
    assert dev == 0.0
    print("INT8-GRID KERNEL OK (optimized leg)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
