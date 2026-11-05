#!/usr/bin/env python3
"""EXL3-INT8 mul24 / mul1 format round trip on a GPU: quantize random tiles, pack the trellis, then check that
the NumPy reference (exl3_int8_ref.decode_q) and ext.reconstruct (+ re-tiling) both reproduce the quantizer's own
reconstruction. Runs against whichever multiplier the ext was built with (EXL3_MUL1_MULT).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from exllamav3.modules.quant.exl3_lib.quantize import (
    quantize_tiles, pack_trellis, tensor_core_perm, tensor_core_perm_i, codebook_mul1_mult, mul24_to_mul1_gather,
    tile_order_frag,
)
from exllamav3.modules.quant.exl3 import _frag_fix
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.exl3_int8_ref import decode_q, S_FIX


@pytest.mark.parametrize("K", [2, 4, 6])
def test_quantize_pack_decode_roundtrip(K):
    dev = "cuda"
    kt, nt = 4, 8
    torch.manual_seed(K)
    w = torch.randn(kt * 16, nt * 16, device=dev)
    tiles = w.reshape(kt, 16, nt, 16).permute(0, 2, 1, 3).reshape(kt * nt, 256)
    qw, qi = quantize_tiles(tiles[:, tensor_core_perm(dev)].contiguous(), {"K": K, "mul1": True})
    qw = qw[:, tensor_core_perm_i(dev)]
    ref = qw.reshape(kt, nt, 16, 16).permute(0, 2, 1, 3).reshape(kt * 16, nt * 16)

    trellis = pack_trellis(qi.reshape(kt, nt, 256), {"K": K})
    q = decode_q(trellis.cpu().numpy(), mult=codebook_mul1_mult)
    assert q.shape == (nt * 16, kt * 16)
    grid = np.rint(ref.double().cpu().numpy() / S_FIX).clip(-128, 127).astype(np.int8).T
    assert (q == grid).all()

    rec = torch.empty((kt * 16, nt * 16), dtype=torch.half, device=dev)
    ext.reconstruct(rec, trellis, K, False, True)
    if tile_order_frag:
        _frag_fix(rec)
    assert torch.equal(rec.float(), ref.half().float())
