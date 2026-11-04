#!/usr/bin/env python3
"""Limited LSA decode A/B: one 8K prompt and one synthetic 64K attention block.

The model benchmark reuses its prompt prefix and measures 64-token decode.
The block benchmark uses real weights and synthetic KV/index history, with
fixed-position teacher-forced inputs. It is not a 64K end-to-end speed claim.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.cache.lsa import CacheLayer_lsa_quant
from exllamav3.constants import PAGE_SIZE
from rocm_tools.lsa_benchmark import TREE, bank, prompt, source_info, write_json


def configure(model, mode):
    os.environ['EXL3_LSA_DECODE_GRAPH'] = '1' if mode == 'flat_graph' else '0'
    for module in model:
        idx = getattr(module, 'lsa_indexer', None)
        if idx is not None:
            idx.settings = replace(idx.settings, decode='descend' if mode == 'descend_eager' else 'flat')


def run_job(generator, ids, expected):
    job = Job(ids, min_new_tokens=64, max_new_tokens=64, sampler=GreedySampler())
    generator.enqueue(job)
    observed = []
    start = time.perf_counter()
    completion = None
    while generator.num_remaining_jobs():
        previous = job.new_tokens
        results = generator.iterate()
        if job.new_tokens > previous:
            observed.append((job.new_tokens, time.perf_counter()))
        for r in results:
            if r['stage'] == 'error':
                raise r['error']
            if r['stage'] == 'streaming' and r.get('eos'):
                completion = r
    torch.cuda.synchronize()
    assert completion and job.new_tokens == 64
    return dict(decode_tps=63/(observed[-1][1]-observed[0][1]),
                wall_seconds=time.perf_counter()-start,
                cached_tokens=completion['cached_tokens'],
                retrieval_correct=completion['full_completion'].lstrip().startswith(expected),
                response_prefix=completion['full_completion'][:64])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--projections', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=32)
    args = p.parse_args()
    torch.manual_seed(4101)
    config = Config.from_directory(args.model, lsa=dict(projections=args.projections, **TREE))
    model = Model.from_config(config)
    cache = Cache(model, 16640, layer_type=CacheLayer_quant, k_bits=5, v_bits=4, max_batch_size=1)
    model.load(device='cuda:0', max_chunk_size=4096, progressbar=False)
    tokenizer = Tokenizer.from_config(config)
    ids, expected, _ = prompt(tokenizer, bank(tokenizer, 8192), 8192, 1)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=4096, enable_defrag=False)
    report = dict(source=source_info(__file__), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                  model=args.model, cache_quant=[5,4], prompt_tokens=8192, output_tokens=64,
                  prompt_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                  model_decode={}, block_64k={}, complete=False)
    for mode in ['descend_eager', 'flat_eager', 'flat_graph']:
        configure(model, mode)
        warmup = run_job(generator, ids, expected)
        timed = run_job(generator, ids, expected)
        replays = sum(getattr(getattr(m.lsa_indexer, 'decode_graph', None), 'replays', 0)
                      for m in model if getattr(m, 'lsa_indexer', None) is not None)
        report['model_decode'][mode] = dict(warmup=warmup, timed=timed, graph_replays=replays)
        print(json.dumps(dict(stage='model_decode', mode=mode, **timed, graph_replays=replays)), flush=True)
        write_json(args.output, report)
    module = next(m for m in model if getattr(m, 'lsa_indexer', None) is not None)
    idx = module.lsa_indexer
    # This history is synthetic; only the layer's weights are the model's.
    capacity = 65536
    layer = CacheLayer_lsa_quant(None, module, 100, capacity, 5, 4)
    layer.alloc(torch.device('cuda:0'))
    bt = torch.arange(capacity//PAGE_SIZE, dtype=torch.int32).unsqueeze(0)
    bt_gpu = bt.cuda()
    seq0 = torch.zeros(1, device='cuda:0', dtype=torch.int32)
    k = torch.randn((1, capacity, module.num_kv_heads, module.head_dim), device='cuda:0', dtype=torch.float16)
    v = torch.randn_like(k)
    layer.update_kv_direct(seq0, bt_gpu, k, v, capacity)
    keys = F.normalize(torch.randn((1, capacity, idx.index_dim), device='cuda:0'), dim=-1)
    layer.append_index(keys, bt_gpu, seq0)
    del k, v, keys
    x = torch.randn((1, 1, module.hidden_size), device='cuda:0', dtype=torch.float16)
    reference = None
    for mode in ['descend_eager', 'flat_eager', 'flat_graph']:
        configure(model, mode)
        params = dict(cache=layer, block_table=bt, cache_seqlens=torch.tensor([capacity-1], dtype=torch.int32),
                      position=capacity-1, causal=True)
        for _ in range(3):
            out = module.decode_flash_attn(x, 1, 1, params)
        out = out.clone()
        if mode == 'flat_eager':
            reference = out
        if mode == 'flat_graph':
            torch.testing.assert_close(out, reference, atol=3e-3, rtol=3e-3)
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                module.decode_flash_attn(x, 1, 1, params)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-start)*1000/args.iterations)
        report['block_64k'][mode] = dict(milliseconds=statistics.median(times), samples_ms=times,
                                       iterations=args.iterations, synthetic_history=True,
                                       graph_matches_eager=(mode == 'flat_graph'))
        print(json.dumps(dict(stage='block_64k', mode=mode, **report['block_64k'][mode])), flush=True)
        write_json(args.output, report)
    assert all(v['timed']['retrieval_correct'] for v in report['model_decode'].values())
    assert report['model_decode']['flat_graph']['graph_replays'] > 0
    report['complete'] = True
    write_json(args.output, report)
    model.unload()

if __name__ == '__main__':
    main()
