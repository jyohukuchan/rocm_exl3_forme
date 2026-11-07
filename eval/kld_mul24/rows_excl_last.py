import json,sys,numpy as np
R="results/llama-bf16-fp16"
vocab=np.array(json.load(open("vocab-v1.json"))["valid_token_ids"])
rm={c["id"]:c for c in json.load(open(R+"/manifest.json"))["cases"]}
def lsm(x):
    x=x.astype(np.float64); m=x.max(1,keepdims=True); return x-m-np.log(np.exp(x-m).sum(1,keepdims=True))
res={}
for name,C in [("mul1",sys.argv[1]),("mul24",sys.argv[2]),("mul24int8",sys.argv[3])]:
    cm={c["id"]:c for c in json.load(open(C+"/manifest.json"))["cases"]}
    allk=[];allk_nl=[]
    for cid,r in rm.items():
        if cid.endswith("repeat"): continue
        shp=tuple(r["shape"]); a=np.memmap(R+"/"+r["logits_file"],dtype="<f4",mode="r",shape=shp)
        b=np.memmap(C+"/"+cm[cid]["logits_file"],dtype="<f4",mode="r",shape=tuple(cm[cid]["shape"]))
        k=[]
        for i in range(0,shp[0],16):
            la=lsm(a[i:i+16][:,vocab]); lb=lsm(b[i:i+16][:,vocab]); k+=list((np.exp(la)*(la-lb)).sum(1))
        k=np.array(k); allk+=list(k); allk_nl+=list(k[:-1])
        if name!="mul1": print(name,cid,"last row KL %.3f, others mean %.4f"%(k[-1],k[:-1].mean()))
    print(name,"mean all %.5f  mean excl last row %.5f"%(np.mean(allk),np.mean(allk_nl)))
