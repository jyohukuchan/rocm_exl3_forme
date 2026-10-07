"""Validate one routed MoE layer against FP32 and the per-expert reference."""
import argparse
import json
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from rocm_tools.moe_ref32 import fp32_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-m','--model',required=True)
    p.add_argument('--rows',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.manual_seed(811)
    model=Model.from_config(Config.from_directory(args.model))
    m=next(x for x in model if isinstance(x,BlockSparseMLP))
    with torch.inference_mode():
        m.load(torch.device('cuda:0'),max_chunk_size=1024)
        assert m.router_pre_norm is None and m.routed_pre_norm is None and m.routed_post_norm is None
        assert m.latent_in is None and m.latent_out is None
        m.shared_experts=None;m.shared_gate=None
        x=torch.randn(1,args.rows,m.hidden_size,device='cuda:0',dtype=torch.float16)*.125
        sel,wts=m.routing_fn(args.rows,m.routing_cfg,x.reshape(-1,m.hidden_size),{})
        oracle=fp32_reference(m,x,sel,wts).reshape_as(x)
        got=m.forward(x,{}).float()
        bc=m.bc
        try:
            m.bc=None
            separate=m.forward(x,{}).float()
        finally:m.bc=bc
        torch.cuda.synchronize()
        error=(got-oracle).abs();diff=got-separate
        r={'model':args.model,'module':m.key,'rows':args.rows,'full_model_loaded':False,
           'decode_bc_retained':bc is not None,'prefill_quant_path':m.support_quant_paths,
           'finite':bool(torch.isfinite(got).all()),'max_oracle_error':float(error.max()),
           'scaled_oracle_error':float((error/oracle.abs().clamp_min(1)).max()),
           'relative_l2_vs_separate':float(diff.norm()/separate.norm().clamp_min(1e-8)),
           'max_difference_vs_separate':float(diff.abs().max())}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
        if not r['finite'] or r['scaled_oracle_error']>.02:raise RuntimeError('MoE numerical check failed')


if __name__=='__main__':main()
