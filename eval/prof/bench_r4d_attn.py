#!/usr/bin/env python3
"""r4d prefill attention (pin b9e42ab + rx10) vs our fp16 Triton prefill.

27B geometry (1 seq, 24 q-heads / 4 kv-heads / head_dim 256, causal).
r4d side: bf16 Q, fp8-e4m3 or bf16 paged KV (block 16), called via ctypes
through the plain C ABI (R4DArgs*, stream). fp8 QK/PV legs selected by
R4D_ATTN_FP8={0,1,2,3} (read once per process by the library).
Our side: paged_attn_triton_prefill over a full-window fp16 PAGE=256 cache,
default knobs and retuned BM128/BN64/S1 (fp16-best). Interleaved rounds,
median/min/max, plus correctness vs a chunked fp32 manual reference.

Usage (one GPU, R9700):
  R4D_ATTN_FP8=0 python3 bench_r4d_attn.py [--smoke] [--tk ...] [--tq ...]
Env: R4D_SO (default /scratch/base-pin/r4d.so), HIP_VISIBLE_DEVICES.
"""
import argparse
import ctypes
import os
import statistics as st
import sys

sys.path.insert(0, "/src")
sys.path.insert(0, "/scratch/prebuilt")

import torch

torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print(f"device: {props.name} {props.gcnArchName} "
      f"{props.total_memory / 1e9:.1f}GB", flush=True)
print(f"R4D_ATTN_FP8={os.environ.get('R4D_ATTN_FP8', '(unset->0)')}", flush=True)

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill

Hq, Hkv, D, PAGE = 24, 4, 256, 256
SM = 1.0 / (D ** 0.5)
BS = 16  # r4d paged block size
torch.manual_seed(0)

R4D_SO = os.environ.get("R4D_SO", "/scratch/base-pin/r4d.so")
r4d = ctypes.CDLL(R4D_SO)


class R4DArgs(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p), ("kv", ctypes.c_void_p),
        ("block_table", ctypes.c_void_p), ("seqused_k", ctypes.c_void_p),
        ("out", ctypes.c_void_p), ("k_descale", ctypes.c_void_p),
        ("v_descale", ctypes.c_void_p), ("q_descale", ctypes.c_void_p),
        ("scratch", ctypes.c_void_p),
        ("num_seqs", ctypes.c_int32), ("q_len", ctypes.c_int32),
        ("q_heads", ctypes.c_int32), ("kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32), ("block_size", ctypes.c_int32),
        ("max_blocks", ctypes.c_int32),
        ("kv_block_stride", ctypes.c_int64), ("kv_head_stride", ctypes.c_int64),
        ("scale", ctypes.c_float), ("splits", ctypes.c_int32),
        ("max_ctx", ctypes.c_int32),
    ]


def _bind(name):
    fn = getattr(r4d, name)
    fn.argtypes = [ctypes.POINTER(R4DArgs), ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


R4D_FP8 = _bind("r4d_attn_prefill_h256_gqa6_fp8kv")
R4D_BF16 = _bind("r4d_attn_prefill_h256_gqa6_bf16kv")


def stream_ptr():
    return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def r4d_call(fn, q, kvp, bt, seq, out, kd, vd, Tk, Tq, nb):
    assert q.is_contiguous() and kvp.is_contiguous() and out.is_contiguous()
    args = R4DArgs(
        q.data_ptr(), kvp.data_ptr(), bt.data_ptr(), seq.data_ptr(),
        out.data_ptr(),
        kd.data_ptr() if kd is not None else None,
        vd.data_ptr() if vd is not None else None, None, None,
        1, Tq, Hq, Hkv, D, BS, nb,
        kvp.stride(0), kvp.stride(1),
        SM, 0, Tk,
    )
    rc = fn(ctypes.byref(args), stream_ptr())
    if rc != 0:
        raise RuntimeError(f"r4d entry returned {rc}")
    return out


def build(Tk, Tq):
    assert Tk % PAGE == 0 and Tk % BS == 0 and Tq <= Tk
    torch.manual_seed(Tk * 131 + Tq)
    q32 = torch.randn((1, Tq, Hq, D), device="cuda:0", dtype=torch.float32)
    k32 = (torch.randn((1, Tk, Hkv, D), device="cuda:0",
                       dtype=torch.float32) * 0.5).contiguous()
    v32 = (torch.randn((1, Tk, Hkv, D), device="cuda:0",
                       dtype=torch.float32) * 0.5).contiguous()
    torch.manual_seed(0)
    q = q32.half()
    k, v = k32.half(), v32.half()
    qb = q32.bfloat16()
    kb, vb = k32.bfloat16(), v32.bfloat16()
    past = Tk - Tq
    # our Triton: full-window fp16 PAGE=256 cache, append path
    pages = Tk // PAGE
    kc = torch.empty((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16)
    vc = torch.empty((pages, PAGE, Hkv, D), device="cuda:0", dtype=torch.float16)
    kc.view(1, Tk, Hkv, D).copy_(k32)
    vc.view(1, Tk, Hkv, D).copy_(v32)
    bt = torch.arange(pages, device="cuda:0", dtype=torch.int32).view(1, pages)
    cs = torch.tensor([past], device="cuda:0", dtype=torch.int32)
    k = k32[:, past:].half().contiguous()
    v = v32[:, past:].half().contiguous()
    # r4d paged: (nb, Hkv, 16, 2D), K then V per slot, bf16
    nb = Tk // BS
    k4 = kb.view(1, nb, BS, Hkv, D).permute(1, 3, 2, 0, 4).reshape(nb, Hkv, BS, D)
    v4 = vb.view(1, nb, BS, Hkv, D).permute(1, 3, 2, 0, 4).reshape(nb, Hkv, BS, D)
    kvp_f = torch.empty((nb, Hkv, BS, 2 * D), device="cuda:0",
                        dtype=torch.bfloat16)
    kvp_f[..., :D].copy_(k4)
    kvp_f[..., D:].copy_(v4)
    # r4d fp8 paged, per-tensor scales (vLLM production style)
    sk = kb.abs().amax().item() / 448.0
    sv = vb.abs().amax().item() / 448.0
    k8 = (kb / sk).to(torch.float8_e4m3fn)
    v8 = (vb / sv).to(torch.float8_e4m3fn)
    kvp8 = torch.empty((nb, Hkv, BS, 2 * D), device="cuda:0",
                       dtype=torch.float8_e4m3fn)
    kvp8[..., :D].copy_(k8.view(1, nb, BS, Hkv, D).permute(1, 3, 2, 0, 4)
                        .reshape(nb, Hkv, BS, D))
    kvp8[..., D:].copy_(v8.view(1, nb, BS, Hkv, D).permute(1, 3, 2, 0, 4)
                        .reshape(nb, Hkv, BS, D))
    kd = torch.full((1, Hkv), sk, device="cuda:0", dtype=torch.float32)
    vd = torch.full((1, Hkv), sv, device="cuda:0", dtype=torch.float32)
    rbt = torch.arange(nb, device="cuda:0", dtype=torch.int32).view(1, nb)
    rseq = torch.tensor([Tk], device="cuda:0", dtype=torch.int32)
    return (q, qb, k, v, kc, vc, bt, cs, kvp_f, kvp8, kd, vd, rbt, rseq,
            k32, v32)


def manual_ref(q, kf, vf):
    """Chunked fp32 causal SDPA. q (1,Tq,Hq,D), k/v (1,Tk,Hkv,D)."""
    Tq, Tk = q.shape[1], kf.shape[1]
    gqh = Hq // Hkv
    out = torch.empty((1, Tq, Hq, D), device="cuda:0", dtype=torch.float32)
    qf, kff, vff = q.float(), kf.float(), vf.float()
    for hv in range(Hkv):
        kv = kff[:, :, hv]          # (1,Tk,D)
        vv = vff[:, :, hv]          # (1,Tk,D)
        for i0 in range(0, Tq, 64):
            i1 = min(i0 + 64, Tq)
            qq = qf[:, i0:i1, hv * gqh:(hv + 1) * gqh]   # (1,c,g,D)
            sc = torch.einsum("cqgd,ktd->cqgkt", qq, kv) * SM                 if False else (qq.unsqueeze(3) * kv.unsqueeze(1).unsqueeze(2)).sum(-1) * SM
            g = torch.arange(Tk - Tq + i0, Tk - Tq + i1,
                             device="cuda:0").view(1, -1, 1, 1)
            kp = torch.arange(Tk, device="cuda:0").view(1, 1, 1, Tk)
            sc = sc.masked_fill(kp > g, float("-inf"))
            p = torch.softmax(sc, dim=-1)
            o = (p.unsqueeze(-1) * vv.view(1, 1, 1, Tk, D)).sum(-2)
            out[:, i0:i1, hv * gqh:(hv + 1) * gqh] = o
    return out.half()


def time_fn(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2], min(ts), max(ts)


def rel_err(got, base):
    d = (got.float() - base.float()).abs()
    return d.max().item(), (d.max() / base.float().abs().amax().clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tk", nargs="+", type=int, default=[8192, 32768, 131072])
    ap.add_argument("--tq", nargs="+", type=int, default=[2048, 8192])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.tk, args.tq, args.iters, args.rounds = [2048], [256], 3, 1

    if not args.skip_correctness:
        print("== correctness (Tk=2048 Tq=256) ==", flush=True)
        (q, qb, k, v, kc, vc, bt, cs, kvp_f, kvp8,
         kd, vd, rbt, rseq, k32, v32) = build(2048, 256)
        ref = manual_ref(q, k32, v32)
        outs = {}
        outs["fp16-def"] = paged_attn_triton_prefill(
            q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=SM,
            out=torch.empty_like(q))
        outs["r4d-bf16kv"] = r4d_call(
            R4D_BF16, qb.view(256, Hq, D), kvp_f, rbt, rseq,
            torch.empty((256, Hq, D), device="cuda:0", dtype=torch.bfloat16),
            None, None, 2048, 256, 128)
        outs["r4d-fp8kv"] = r4d_call(
            R4D_FP8, qb.view(256, Hq, D), kvp8, rbt, rseq,
            torch.empty((256, Hq, D), device="cuda:0", dtype=torch.bfloat16),
            kd, vd, 2048, 256, 128)
        for name, o in outs.items():
            oo = o.view(1, 256, Hq, D)
            ma, re = rel_err(oo, ref)
            print(f"  {name}: vs manual maxabs={ma:.2e} rel={re:.2e}",
                  flush=True)
        ma, re = rel_err(outs["r4d-bf16kv"].view(1, 256, Hq, D),
                         outs["fp16-def"])
        print(f"  r4d-bf16kv vs fp16-def: maxabs={ma:.2e} rel={re:.2e}",
              flush=True)
        del q, qb, k, v, kc, vc, bt, cs, kvp_f, kvp8, kd, vd, rbt, rseq, ref, outs, k32, v32
        torch.cuda.empty_cache()

    print("== speed (ms, med [min,max] over rounds; r4dx = ratio vs fp16-best) ==",
          flush=True)
    for Tk in args.tk:
        for Tq in args.tq:
            (q, qb, k, v, kc, vc, bt, cs, kvp_f, kvp8,
             kd, vd, rbt, rseq, k32, v32) = build(Tk, Tq)
            nb = Tk // BS
            o_r8 = torch.empty((Tq, Hq, D), device="cuda:0", dtype=torch.bfloat16)
            o_rf = torch.empty((Tq, Hq, D), device="cuda:0", dtype=torch.bfloat16)
            o_fd = torch.empty_like(q)
            o_fb = torch.empty_like(q)
            qv = qb.view(Tq, Hq, D)
            arms = {
                "r4d-fp8kv": lambda: r4d_call(R4D_FP8, qv, kvp8, rbt, rseq,
                                              o_r8, kd, vd, Tk, Tq, nb),
                "r4d-bf16kv": lambda: r4d_call(R4D_BF16, qv, kvp_f, rbt, rseq,
                                               o_rf, None, None, Tk, Tq, nb),
                "fp16-def": lambda: paged_attn_triton_prefill(
                    q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=SM,
                    out=o_fd),
                "fp16-best": lambda: paged_attn_triton_prefill(
                    q, k, v, kc, vc, bt, cs, causal=True, softmax_scale=SM,
                    out=o_fb, block_m=128, block_n=64, num_stages=1),
            }
            acc = {name: [] for name in arms}
            for _ in range(args.rounds):
                for name, fn in arms.items():
                    med, lo, hi = time_fn(fn, args.iters)
                    acc[name].append(med)
            base = sorted(acc["fp16-best"])[len(acc["fp16-best"]) // 2]
            row = f"Tk={Tk} Tq={Tq}:"
            for name in arms:
                m = sorted(acc[name])[len(acc[name]) // 2]
                row += f" {name}={m:.2f}({m / base:.2f}x)"
            print(row + f" [spread r4d-fp8={max(acc['r4d-fp8kv'])-min(acc['r4d-fp8kv']):.2f}]",
                  flush=True)
            del q, qb, k, v, kc, vc, bt, cs, kvp_f, kvp8, kd, vd, rbt, rseq, k32, v32
            del o_r8, o_rf, o_fd, o_fb
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
