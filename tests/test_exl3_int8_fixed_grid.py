#!/usr/bin/env python3
"""EXL3-INT8 fixed grid (doc/exl3_int8.md).

The math test runs anywhere. The decoder test needs a GPU and an EXL3 mul1
model directory in EXL3_INT8_TEST_MODEL, and skips otherwise:
  EXL3_INT8_TEST_MODEL=/work/models/q30b-exl3-305-i python3 -m pytest tests/test_exl3_int8_fixed_grid.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

S_FIX = 4.0 * 887.0 / 131072.0


def _mul1_value(b):
    return (887 * b - 452608) / 131072.0


def _q_shift(b):
    return (b - 509) >> 2


def test_shift_equals_round_for_every_byte_sum():
    b = np.arange(0, 1021, dtype=np.int64)
    exact = np.clip(np.rint(_mul1_value(b) / S_FIX), -128, 127).astype(np.int64)
    assert (_q_shift(b) == exact).all()
    assert _q_shift(b).min() == -128 and _q_shift(b).max() == 127


def test_fp16_decode_rounds_to_the_same_grid_point():
    b = np.arange(0, 1021, dtype=np.int64)
    vh = _mul1_value(b).astype(np.float16).astype(np.float64)
    assert (np.rint(vh / S_FIX).astype(np.int64) == _q_shift(b)).all()


def test_real_decoder_matches_shift_formula():
    model = os.environ.get("EXL3_INT8_TEST_MODEL")
    if not model:
        pytest.skip("EXL3_INT8_TEST_MODEL not set")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from safetensors import safe_open
    from exllamav3.ext import exllamav3_ext as ext

    wmap = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    names = sorted(k for k in wmap if k.endswith(".trellis"))
    names = [names[0], names[len(names) // 2], names[-1]]
    from exllamav3.util.exl3_int8_ref import decode_q
    for name in names:
        with safe_open(os.path.join(model, wmap[name]), "pt", device="cpu") as f:
            trellis = f.get_tensor(name)
        k = trellis.shape[-1] // 16
        w = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype=torch.half, device="cuda")
        ext.reconstruct(w, trellis.cuda(), k, False, True)
        q_dev = np.clip(np.rint(w.double().cpu().numpy() / S_FIX), -128, 127).astype(np.int8).T
        q_ref = decode_q(trellis.numpy())
        assert q_ref.shape == q_dev.shape
        assert (q_ref == q_dev).all(), name
