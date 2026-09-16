"""Does explicitly teaching query->key match improve recall through the
same compressed M/z memory? (review's next lever)

Training-only auxiliary loss: L = L_answer + lambda * CE(beta*cos(q,k_i); t),
t = queried entity. Documented matching supervision. The memory (additive
M/z), retention, precision, decoder, writes, budget are UNCHANGED; the aux
classifier is training-only and the eval uses the same linear M/z read.

Arms (matched init + batches, N=8, Q=1, 2 seeds):
  base   L_answer only (the v30 learned arm)
  disc   L_answer + lambda * discrimination CE

Reports per seed: recall, target rank/mass, memory-off recall,
clean-value rescue.
"""
import argparse, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from nonce_lm import make_batch, load_carrier, V
from oracle_addr import OracleAddrLM
from addr_diag import diagnose

def train(disc, N, carrier, steps, seed, dev, lam=0.5, B=16, lr=1e-3):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    m=OracleAddrLM('learned',N).to(dev)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    for step in range(steps):
        x,y,spans,tgt,_=make_batch(B,N,carrier,rng,dev)
        loss=F.cross_entropy(m(x,spans,tgt).reshape(-1,V),y.reshape(-1),ignore_index=-100)
        if disc:
            loss=loss+lam*F.cross_entropy(m.match_logits(x,spans),tgt)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
        if (step+1)%2000==0: print(f"    .. {'disc' if disc else 'base'} s{seed} step {step+1} loss {float(loss.detach()):.3f}",flush=True)
    return m

@torch.no_grad()
def evalm(m,N,carrier,dev,mode):
    r=np.random.default_rng(99); ok=n=0; m.eval()
    for _ in range(8):
        x,y,sp,t,_=make_batch(64,N,carrier,r,dev)
        rr=m.target_value(x,sp,t) if mode=='inject' else None
        lo=m(x,sp,t,mem_off=(mode=='off'),read_r=rr); s=y!=-100
        ok+=int((((lo.argmax(-1)==y)|~s).all(1)).sum()); n+=x.shape[0]
    return ok/n

ap=argparse.ArgumentParser(); ap.add_argument("--steps",type=int,default=6000)
ap.add_argument("--seeds",type=int,default=2); ap.add_argument("--lam",type=float,default=0.5)
ap.add_argument("--out",default="disc-train.json"); a=ap.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; carrier=load_carrier()
print(f"discrimination-training lever · N=8 Q=1 · lambda={a.lam} · {a.steps} steps · {a.seeds} seeds · {dev}\n")
res={}
for arm,disc in (("base",False),("disc",True)):
    for s in range(a.seeds):
        m=train(disc,8,carrier,a.steps,s,dev,lam=a.lam)
        d=diagnose(m,8,carrier,dev)
        on=evalm(m,8,carrier,dev,'on'); off=evalm(m,8,carrier,dev,'off'); inj=evalm(m,8,carrier,dev,'inject')
        torch.save({"model":m.state_dict(),"addr":"learned","N":8,"seed":s,"arm":arm},f"disc-{arm}-s{s}.pt")
        print(f"  {arm} s{s}: recall {on:.1%} | off {off:.1%} | clean-inject {inj:.1%} | "
              f"rank {d['target_rank']:.2f} mass {d['read_mass_target']:.2f} overlap {d['write_overlap']:.2f}")
        res[f"{arm}-s{s}"]=dict(recall=on,off=off,inject=inj,rank=d['target_rank'],
                                mass=d['read_mass_target'],overlap=d['write_overlap'])
json.dump(res,open(a.out,'w'),indent=1); print(f"\nwrote {a.out}")
