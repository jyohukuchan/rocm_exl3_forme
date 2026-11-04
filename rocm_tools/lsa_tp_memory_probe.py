#!/usr/bin/env python3
"""Stress real TP/MTP decode at the full window with synthetic cached history.

This avoids an hours-long prefill while checking the exact cache, table and
graph memory shapes. It does not measure retrieval quality. The natural full
prefill remains a separate validation.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from exllamav3 import model_init
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.lsa import mp_lsa_stats


@torch.inference_mode()
def seed_history(context,cache_id,table,past):
    pages=table[0].tolist()
    seeded=0
    for root in context['modules']:
        if root is None:continue
        for module in root:
            idx=getattr(module,'lsa_indexer',None)
            if idx is None:continue
            for layer in module.cache_layers:
                if layer.cache_id!=cache_id or not layer.owns_index:continue
                s=layer.lsa_settings
                layer.centroids.copy_(F.normalize(torch.randn_like(layer.centroids),dim=-1))
                layer.members.fill_(-1)
                members=torch.arange(s.window_size,device=layer.members.device,dtype=torch.int32).reshape(-1,s.bucket_size)
                layer.members[:,:,:s.bucket_size].copy_(members[None])
                layer.bucket_cache.clear()
                for block in range(past//s.window_size):
                    first=block*s.window_size//PAGE_SIZE
                    last=((block+1)*s.window_size+PAGE_SIZE-1)//PAGE_SIZE
                    signature=(block*s.window_size%PAGE_SIZE,tuple((int(p),0) for p in pages[first:last]))
                    layer.bucket_cache[signature]=block
                seeded+=1
    return seeded


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    model_init.add_args(p,default_cache_size=1835520,default_chunk_size=4096,add_draft_model_args=True)
    p.add_argument('--past-tokens',type=int,default=1835007)
    p.add_argument('--json-out',type=Path,required=True)
    a=p.parse_args()
    model,config,cache,_,draft,_,draft_cache=model_init.init(a,load_tokenizer=False,progress=False)
    if draft is not None:draft.attach_to(model)
    cols=(a.cache_size//PAGE_SIZE+15)//16*16
    table=torch.zeros((1,cols),dtype=torch.int32)
    table[0,:a.cache_size//PAGE_SIZE]=torch.arange(a.cache_size//PAGE_SIZE,dtype=torch.int32)
    report={'synthetic_history':True,'natural_full_prefill':False,'cache_tokens':a.cache_size,
            'initial_past_tokens':a.past_tokens,'physical_pages':a.cache_size//PAGE_SIZE,'padded_columns':cols,
            'runs':[],'complete':False}
    a.json_out.parent.mkdir(parents=True,exist_ok=True)
    def save():a.json_out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    try:
        report['seeded_index_layers']=model.tp_worker_dispatch_wait_multi(model.active_devices,seed_history,(id(cache),table,a.past_tokens))
        state=cache.get_test_state(a.past_tokens)
        report['loaded_stats']=model.tp_worker_dispatch_wait_multi(model.active_devices,mp_lsa_stats,())
        save()
        for n in [1,5,1,5]:
            past=state.position
            ids=torch.full((1,n),100,dtype=torch.long)
            params={'attn_mode':'flash_attn','cache':cache,'block_table':table,
                    'cache_seqlens':torch.tensor([past],dtype=torch.int32),'recurrent_states':[state],
                    'recurrent_history':n>1}
            if draft is not None:params.update(draft.draft_verifier_params)
            logits=model.forward(input_ids=ids,params=params)
            finite=bool(torch.isfinite(logits[:,:,:config.vocab_size]).all())
            row={'queries':n,'past':past,'target_logits_finite':finite}
            assert finite
            if draft is not None:
                hidden=params['export_states'][-1][:,-1:,:].contiguous()
                dp={'attn_mode':'flash_attn','cache':draft_cache,'block_table':table,
                    'cache_seqlens':torch.tensor([past+n-1],dtype=torch.int32),'target_hidden':hidden}
                ds=draft.forward(input_ids=ids[:,-1:],params=dp)
                row['draft_state_finite']=bool(torch.isfinite(ds).all())
                row['draft_token']=int(draft.sample_from_state(ds,{}).reshape(-1)[0])
                assert row['draft_state_finite'] and 0<=row['draft_token']<config.vocab_size
            if n>1:state.rewind(n-1)
            row['post_rewind_position']=state.position
            report['runs'].append(row)
            print(json.dumps(row),flush=True);save()
        report['final_stats']=model.tp_worker_dispatch_wait_multi(model.active_devices,mp_lsa_stats,())
        report['complete']=True;save()
    finally:
        model.unload()
        if draft is not None:draft.unload()

if __name__=='__main__':main()
