"""v4.1 — can denser supervision rescue the fw2 synaptic store at N=8?

Three memory substrates now floor at N=8 (slot bank 1.4%, fw1 1.2/1.4/1.4,
fw2 1.4%), which localised the cliff to CREDIT DENSITY, not architecture:
one queried fact per sequence gives ~1/N training signal per stored
entity. The lever is denser supervision, and the user's refinement is to
pull it on the BEST substrate - fw2, the two-gate synaptic store
(dopamine write + octopamine retention) - not the slot bank.

So: fw2 memory, N=8 fixed, sweep how many stored entities are queried per
sequence. Each query reads the SAME fast-weight M with its own key and
injects its own read over its own span; M is built once by the writer.

ARMS (query count Q against N=8 stored):
  q1        one query - reproduces the fw2 N=8 floor (baseline)
  q2/q4/q8  Q DISTINCT entities queried - the density lever
  q8rep     8 repeats of ONE query - magnitude without coverage (control).
            Evaluated on first query only, so repeat-copyability cannot
            inflate the metric.
  curr      curriculum N=1->2->4->8, Q=1 - the basin/search-difficulty arm.

Uniform metric across all arms: first-query recall on fresh N=8/Q=1
sequences.

PRE-REGISTERED:
  q1 ~ floor (~1.4%, matching fw2). If recall RISES with Q -> credit
  density confirmed as the cliff's cause and denser supervision is the
  fix; the N=8 "limit" was a training protocol, not the synaptic store.
  q8rep stays near floor -> coverage, not gradient magnitude, is what
  matters. If q8 also floors -> denser supervision does not rescue it
  and the bottleneck is deeper (optimisation/addressing), redirecting to
  the architecture again.

    python fly_credit.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nonce_lm import (V, GAP, NAME_L, FACT_L, QUERY_L, ANS_L, PREGAP,
                      COLORS, ITEMS, load_carrier, enc)
from nonce_lm2 import Block, quantise_fixed
from nonce_credit import make_batch_multi


class FlyMQ(nn.Module):
    """fw2 synaptic store (write gate + retention), read Q times per seq."""

    def __init__(self, N, d=128, dk=16, dv=16, qbits=8, layers=3, heads=4,
                 window=128, gru=64, max_seq=2048):
        super().__init__()
        self.window, self.qbits, self.dk, self.dv = window, qbits, dk, dv
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.writer = nn.GRU(d, gru, batch_first=True)
        self.w_gate = nn.Linear(gru, 1)            # dopamine write gate
        self.w_ret = nn.Linear(gru, 1)             # octopamine retention
        self.w_key = nn.Linear(gru, dk)
        self.w_val = nn.Linear(gru, dv)
        self.q_gru = nn.GRU(d, gru, batch_first=True)
        self.q_key = nn.Linear(gru, dk)
        self.r_up = nn.Linear(dv, d, bias=False)
        nn.init.zeros_(self.r_up.weight)
        self.scale = dk ** -0.5

    def build_memory(self, x, cpre):
        B = x.shape[0]
        e = self.emb(x[:, :cpre]) + self.pos[:cpre]
        h, _ = self.writer(e)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)
        a = torch.sigmoid(self.w_ret(h)).squeeze(-1)           # fw2 retention
        k = torch.tanh(self.w_key(h))
        v = torch.tanh(self.w_val(h))
        M = x.new_zeros(B, self.dv, self.dk)
        z = x.new_zeros(B, self.dk)
        for t in range(cpre):
            at, gt = a[:, t].view(B, 1, 1), g[:, t].view(B, 1)
            M = at * M + gt.view(B, 1, 1) * torch.einsum(
                "bv,bk->bvk", v[:, t], k[:, t])
            z = a[:, t].view(B, 1) * z + gt * k[:, t]
        return quantise_fixed(torch.tanh(M), self.qbits), z

    def read(self, x, q0, M, z):
        qe = self.emb(x[:, q0:q0 + QUERY_L]) + self.pos[:QUERY_L]
        hq, _ = self.q_gru(qe)
        q = torch.tanh(self.q_key(hq[:, 1 + NAME_L]))
        num = torch.einsum("bvk,bk->bv", M, q)
        den = (z * q).sum(-1, keepdim=True) + 1e-3
        return num / den

    def forward(self, x, qstart, Q):
        B, Nx = x.shape
        M, z = self.build_memory(x, qstart)        # writer sees only pre-cue
        h = self.emb(x) + self.pos[:Nx]
        h = h.clone()
        for qi in range(Q):
            q0 = qstart + qi * (QUERY_L + ANS_L)
            r = self.read(x, q0, M, z)
            seg_end = min(q0 + QUERY_L + ANS_L, Nx)
            h[:, q0:seg_end] = h[:, q0:seg_end] + self.r_up(r).unsqueeze(1)
        mask = torch.ones(Nx, Nx, dtype=torch.bool, device=x.device)
        for i in range(Nx):
            lo = max(0, i - self.window + 1)
            mask[i, lo:i + 1] = False
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_out(h))


def train_arm(arm, carrier, steps, seed, device, B=16, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = FlyMQ(8).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        if arm == "curr":
            N = [1, 2, 4, 8][min(3, step // max(1, steps // 4))]; Q, distinct = 1, True
        else:
            N = 8
            Q = {"q1": 1, "q2": 2, "q4": 4, "q8": 8, "q8rep": 8}[arm]
            distinct = arm != "q8rep"
        x, y, spans, qt, qstart = make_batch_multi(
            B, N, Q, carrier, rng, device, distinct)
        logits = m(x, qstart, Q)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (step + 1) % 2000 == 0:
            print(f"    .. {arm} step {step+1}/{steps} loss "
                  f"{float(loss.detach()):.3f}", flush=True)
    ev = np.random.default_rng(99)
    ok = n = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, qt, qstart = make_batch_multi(
                64, 8, 1, carrier, ev, device, True)   # uniform: N=8, first query
            pred = m(x, qstart, 1).argmax(-1)
            sel = y != -100
            ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arms", nargs="+",
                    default=["q1", "q2", "q4", "q8", "q8rep", "curr"])
    ap.add_argument("--out", default="fly-credit.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()

    print(f"fw2 synaptic store + multi-query · N=8 · {a.steps} steps · "
          f"{a.seeds} seeds · {dev}")
    print(f"(fw2 N=8 single-query floor: 1.4%)\n")
    print(f"  {'arm':<8}{'recall med':>12}{'conv':>6}   per-seed")
    print("  " + "-" * 44)
    res = {}
    for arm in a.arms:
        accs = [train_arm(arm, carrier, a.steps, s, dev)
                for s in range(a.seeds)]
        med = float(np.median(accs))
        conv = sum(v >= 0.20 for v in accs)
        print(f"  {arm:<8}{med:>11.1%}{conv:>4}/{a.seeds}   "
              + " ".join(f"{v:.2f}" for v in accs), flush=True)
        res[arm] = dict(acc=accs, median=med, converged=conv)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
