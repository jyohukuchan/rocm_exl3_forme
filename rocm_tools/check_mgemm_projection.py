"""Compare a real quantized K/V bundle with separate projections and FP32 matmul.

Loads only the two selected projection matrices, not another full model. Run
one row shape per process so a device fault cannot invalidate later cases.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from exllamav3 import Config,Model
from exllamav3.modules.multilinear import MultiLinear
from exllamav3.ext import exllamav3_ext as ext


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('-m','--model',required=True)
    p.add_argument('--rows',type=int,default=16)
    p.add_argument('--shape',type=int,default=-1)
    p.add_argument('--dispatch',action='store_true',help='Use the attention module dispatch instead of forcing the raw kernel')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.manual_seed(731)
    config=Config.from_directory(args.model);model=Model.from_config(config)
    attention=next(m for m in model if getattr(m,'k_proj',None) is not None and getattr(m,'v_proj',None) is not None)
    linears=[attention.k_proj,attention.v_proj]
    device=torch.device('cuda:0')
    with torch.inference_mode():
        if args.dispatch:attention.load(device)
        else:
            for linear in linears:linear.load(device)
        multi=MultiLinear(device,linears)
        x=torch.randn((1,args.rows,multi.in_features),device=device,dtype=torch.float16)*.125
        separate=torch.stack([linear.forward(x,{})[0] for linear in linears])
        oracle=torch.stack([(x.float() @ linear.inner.get_weight_tensor().float())[0] for linear in linears])
        torch.cuda.synchronize()
        result={'model':args.model,'module':attention.key,'rows':args.rows,'shape':args.shape,
                'in_features':multi.in_features,'out_features':multi.out_features,'bits':multi.K,
                'mcg':multi.mcg,'mul1':multi.mul1,'full_model_loaded':False,
                'module_dispatch':args.dispatch,
                'extension':ext.__file__,'separate_max_error':float((separate.float()-oracle).abs().max())}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        y=torch.full((2,args.rows,multi.out_features),float('nan'),device=device,dtype=torch.float16)
        had=torch.empty((2,args.rows,multi.in_features),device=device,dtype=torch.float16)
        start=time.monotonic()
        if args.dispatch:
            # Compare raw K/V projections, before the model-specific norm/RoPE.
            attention.finish_qkv=lambda q,k,v,g,*a:(q,k,v,g)
            _,k,v,_=attention.project_qkv(x,{})
            y=torch.stack([k[0],v[0]])
        else:
            ext.exl3_mgemm(x,multi.ptrs_trellis,y,multi.ptrs_suh,had,multi.ptrs_svh,
                          None,None,multi.K,args.shape,multi.mcg,multi.mul1,-1,-1,0,1,None,None)
        torch.cuda.synchronize()
        error=(y.float()-oracle).abs()
        result.update(seconds=time.monotonic()-start,finite=bool(torch.isfinite(y).all()),
                      max_error=float(error.max()),max_scaled_error=float((error/oracle.abs().clamp_min(1)).max()),
                      max_separate_difference=float((y-separate).abs().max()))
        args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        if not result['finite'] or result['max_scaled_error']>.02:raise RuntimeError('Projection numerical check failed')


if __name__=='__main__':main()
