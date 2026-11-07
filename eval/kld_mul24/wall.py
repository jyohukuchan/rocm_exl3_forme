import json, sys, torch
from safetensors import safe_open
from exllamav3.modules.quant.exl3 import LinearEXL3
SRC="/work/models/Qwen3.8-27B-BF16"
def idx(d): return json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
sw=idx(SRC); D=f"/work/models/{sys.argv[1]}"; wm=idx(D)
def get(d,m,k):
    with safe_open(f"{d}/{m[k]}","pt",device="cpu") as f: return f.get_tensor(k)
names=sorted(k[:-8] for k in wm if k.endswith(".trellis"))
bad=0
for n in names:
    tr=get(D,wm,n+".trellis").cuda()
    kw=dict(suh=get(D,wm,n+".suh").cuda(),svh=get(D,wm,n+".svh").cuda())
    lin=LinearEXL3(None,tr.shape[0]*16,tr.shape[1]*16,trellis=tr,mul1=get(D,wm,n+".mul1"),key=n,**kw)
    w=lin.get_weight_tensor().float()
    rk=n+".weight"
    if rk not in sw: print("NOREF",n,flush=True); continue
    ref=get(SRC,sw,rk).cuda().float().T
    if ref.shape!=w.shape: print("SHAPE",n,ref.shape,w.shape,flush=True); continue
    e=((w-ref).norm()/ref.norm()).item()
    flag = e>0.12
    if flag or "layers.0." in n or "mtp" in n or not "layers" in n: print(("BAD " if flag else "")+n,"K=",tr.shape[-1]//16,"%.4f"%e,flush=True)
print("done",len(names))
