#!/usr/bin/env python3
"""Allocate a split LSA model/cache and worst-case decode table workspace.

Runs a short retrieval request; does not prefill or assess quality at the full
cache capacity, and does not modify RoPE automatically.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from rocm_tools.lsa_benchmark import bank, prompt, source_info, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--projections',required=True)
    p.add_argument('--cache-size',type=int,default=2097664)
    p.add_argument('--kv',type=int,nargs=2,default=[5,4])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=dict(source=source_info(__file__),cache_tokens=a.cache_size,cache_quant=a.kv,
                split='layers',gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                full_context_prefill_tested=False,loaded=False,workspace_allocated=False,short_retrieval_correct=False)
    model=None
    workspace=[]
    try:
        c=Config.from_directory(a.model,lsa={'projections':a.projections})
        model=Model.from_config(c)
        cache=Cache(model,a.cache_size,layer_type=CacheLayer_quant,k_bits=a.kv[0],v_bits=a.kv[1],max_batch_size=1)
        report['cache_storage_gib']=sum(int(l.storage_size()) for l in cache.layers.values())/2**30
        # The autosplit loader consumes OOM exceptions while trying another
        # device. Preserve their messages without retaining traceback tensors.
        report['allocation_failures']=[]
        def allocation_trace(frame,event,arg):
            if not frame.f_code.co_filename.endswith('/model_ls.py'):
                return None
            if event=='exception':
                kind,error,_=arg
                if kind.__name__=='OutOfMemoryError' or 'out of memory' in str(error):
                    report['allocation_failures'].append(dict(module=getattr(frame.f_locals.get('module'),'key',None),error=str(error)))
            return allocation_trace
        sys.settrace(allocation_trace)
        try:
            model.load(max_chunk_size=4096,progressbar=False)
        finally:
            sys.settrace(None)
        report['loaded']=True
        write_json(a.output,report)
        print('Model and full-capacity KV/index loaded',flush=True)
        report['cache_layers_per_gpu']={str(i):sum(l.device==torch.device('cuda',i) for l in cache.layers.values()) for i in range(torch.cuda.device_count())}
        # Hold enough bytes for a fully populated logical read plus its largest
        # static graph table. No long-context attention work is required here.
        by_device={}
        for l in cache.layers.values():
            s=l.lsa_settings
            n=(a.cache_size-1)//s.window_size*(s.window_size//s.bucket_size)
            cap=1<<(max(1,n)-1).bit_length()
            size=(n+cap)*(l.index_dim+s.bucket_cap)*4
            workspace.append(torch.empty(size,dtype=torch.uint8,device=l.device))
            by_device[str(l.device)]=by_device.get(str(l.device),0)+size
        report['workspace_allocated']=True
        report['workspace_gib_per_gpu']={k:v/2**30 for k,v in by_device.items()}
        write_json(a.output,report)
        print('Full-context decode table workspace allocated',flush=True)
        tokenizer=Tokenizer.from_config(c)
        ids,expected,_=prompt(tokenizer,bank(tokenizer,8192),8192,1)
        gen=Generator(model,cache,tokenizer,max_batch_size=1,max_chunk_size=4096,enable_defrag=False)
        job=Job(ids,max_new_tokens=16,sampler=GreedySampler())
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r['stage']=='error': raise r['error']
                if r['stage']=='streaming' and r.get('eos'):
                    report['response']=r['full_completion']
                    report['short_retrieval_correct']=r['full_completion'].lstrip().startswith(expected)
        report['memory']=[dict(device=i,allocated_gib=torch.cuda.memory_allocated(i)/2**30,
                               reserved_gib=torch.cuda.memory_reserved(i)/2**30,
                               free_gib=torch.cuda.mem_get_info(i)[0]/2**30) for i in range(torch.cuda.device_count())]
        report['graph_replays']=sum(getattr(getattr(m.lsa_indexer,'decode_graph',None),'replays',0) for m in model if getattr(m,'lsa_indexer',None) is not None)
    except Exception as error:
        report['error']=repr(error)
        report['memory']=[dict(device=i,allocated_gib=torch.cuda.memory_allocated(i)/2**30,
                               reserved_gib=torch.cuda.memory_reserved(i)/2**30,
                               free_gib=torch.cuda.mem_get_info(i)[0]/2**30) for i in range(torch.cuda.device_count())]
        raise
    finally:
        write_json(a.output,report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
        workspace.clear()
        if model is not None: model.unload()
        gc.collect()

if __name__=='__main__': main()
