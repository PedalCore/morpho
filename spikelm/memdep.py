"""Held-out answer loss + recall, memory ON vs memory-read ZEROED.

Baseline (review): 64 uniform 10-char answers -> no-information per-char
loss = ln(64)/10 ~ 0.416. Held-out loss below that = real predictive
information. Memory-off (r zeroed) measures dependence on the memory
pathway: if OFF worsens loss/recall, the model uses memory; if barely
changes, something else supplies the information. Does NOT by itself
establish selective addressing.
"""
import argparse, json, math
import numpy as np, torch, torch.nn.functional as F
from nonce_lm import make_batch, load_carrier, V
from oracle_addr import OracleAddrLM

@torch.no_grad()
def evalu(m, N, carrier, dev, mem_off, batches=8):
    ev = np.random.default_rng(99); tot=ntok=ok=n=0
    m.eval()
    for _ in range(batches):
        x,y,spans,tgt,_ = make_batch(64,N,carrier,ev,dev)
        logits = m(x,spans,tgt,mem_off=mem_off)
        loss = F.cross_entropy(logits.reshape(-1,V), y.reshape(-1), ignore_index=-100, reduction='sum')
        sel = y!=-100
        tot += float(loss); ntok += int(sel.sum())
        pred = logits.argmax(-1)
        ok += int((((pred==y)|~sel).all(1)).sum()); n += x.shape[0]
    return tot/ntok, ok/n

ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--N",type=int,default=8); a=ap.parse_args()
dev="cuda" if torch.cuda.is_available() else "cpu"; carrier=load_carrier()
ck=torch.load(a.ckpt,map_location=dev); m=OracleAddrLM(ck["addr"],a.N).to(dev); m.load_state_dict(ck["model"])
base=math.log(64)/10
lon,ron = evalu(m,a.N,carrier,dev,False)
loff,roff = evalu(m,a.N,carrier,dev,True)
print(f"{a.ckpt}  addr={ck['addr']}  (no-info baseline loss {base:.3f})")
print(f"  memory ON :  held-out loss {lon:.3f}   exact recall {ron:.1%}")
print(f"  memory OFF:  held-out loss {loff:.3f}   exact recall {roff:.1%}")
print(f"  memory dependence:  dloss {loff-lon:+.3f}   drecall {ron-roff:+.1%}"
      f"   ({'uses memory' if (loff-lon>0.02 or ron-roff>0.02) else 'barely uses memory'})")
json.dump(dict(baseline=base, loss_on=lon, recall_on=ron, loss_off=loff,
               recall_off=roff), open(a.ckpt.replace('.pt','-memdep.json'),'w'), indent=1)

@torch.no_grad()
def eval_inject(m, N, carrier, dev, batches=8):
    """Inject the target's OWN value (perfect unmixed read) into the decoder."""
    ev = np.random.default_rng(99); ok=n=0; tot=ntok=0
    m.eval()
    for _ in range(batches):
        x,y,spans,tgt,_ = make_batch(64,N,carrier,ev,dev)
        r = m.target_value(x,spans,tgt)
        logits = m(x,spans,tgt,read_r=r)
        loss = F.cross_entropy(logits.reshape(-1,V), y.reshape(-1), ignore_index=-100, reduction='sum')
        sel=y!=-100; tot+=float(loss); ntok+=int(sel.sum())
        pred=logits.argmax(-1); ok+=int((((pred==y)|~sel).all(1)).sum()); n+=x.shape[0]
    return tot/ntok, ok/n
