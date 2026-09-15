"""v4 (fly) — a synaptic store vs our slot bank on the recall cliff.

Inspiration, not authority: the fan-shaped-body "synaptic store" (Wang,
fly-circuit-exploration) holds a running vector sum in CHANGING SYNAPSE
STRENGTHS, written under a dopamine gate and reset by octopamine - two
separate signals - and addressed by the active column rather than a
learned router. The mechanistic update rule is unpublished (the finding
is anatomical and flagged unproven at the minutes scale), so we transfer
the ARCHITECTURE, not an equation.

That architecture is a fast-weight associative memory:

    write (pre-cue positions):  M <- a_t * M + g_t * (v_t (x) k_t)
    read  (query):              r = (M q) / (z . q + eps),  z = sum a g k

  g_t = write gate     (dopamine analogue, sigmoid, learned)
  a_t = retention      (octopamine reset analogue: a_t -> 0 clears M)

WHY THIS IS THE RIGHT TEST OF OUR OPEN PROBLEM. Our slot bank routes each
write to one of K slots via a learned softmax router - which collapsed
(v25: 91% collisions) and hit a LEARNING CLIFF at N=8 (v28: recall 1.4%
while the router kept organising). A synaptic store has NO router to
collapse: writes superpose additively, addressed by the key direction
itself. The fly, linear-attention theory, and our own kvaddr result
(structured addressing beat free allocation 46.7 vs 8.4) all say the
same thing - let the key BE the address.

FAIR COMPARISON. M is d_v x d_k = 16 x 16 = 256 numbers, identical to the
slot ledger (K=8 x (16+16)). Same GRU writer, same query key, same reader
injection (zero-init residual over the query span), same trunk, same
data, same budget. Only the memory substrate differs.

ARMS, swept over N in {2,4,8,16}:
  slot   the v3.1 KV slot bank + load-balanced router  (the cliff)
  fw1    fast-weight, write gate only (a_t = 1: pure Hebbian accumulation)
  fw2    fast-weight, write gate + learned retention a_t (the two-gate
         fly hypothesis: does a separate reset/decay signal manage
         interference and lift the high-N cliff?)

PRE-REGISTERED:
  1. slot reproduces the v28 cliff: ~ceiling at N<=4, floor at N=8 (this
     harness's basin noise permitting).
  2. THE TEST: fw1/fw2 recall materially above floor at N=8 where slot
     floored -> the router, not capacity or credit, was the N=8 killer,
     and superposition fixes it.
  3. fw2 >= fw1 at high N -> the octopamine-style retention gate buys
     interference management (the two-signal principle earns its keep).
     fw2 ~ fw1 -> one gate suffices at these N; reset matters only later.
  Null (all three floor at N=8) -> the cliff is credit/signal density
  (v28's leading hypothesis), not routing, and the fly substrate does
  not rescue it. Either way the cliff's cause is localised.

    python fly_memory.py --entities 8
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nonce_lm import make_batch, load_carrier, V, QUERY_L, ANS_L, NAME_L
from nonce_lm2 import Block, quantise_fixed


class FlyNonceLM(nn.Module):
    """Trunk + writer identical to KVNonceLM; memory is a fast-weight
    matrix instead of a routed slot bank."""

    def __init__(self, mem, N, d=128, dk=16, dv=16, qbits=8, layers=3,
                 heads=4, window=128, gru=64, max_seq=2048):
        super().__init__()
        self.mem, self.window, self.qbits = mem, window, qbits
        self.dk, self.dv = dk, dv
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.writer = nn.GRU(d, gru, batch_first=True)
        self.w_gate = nn.Linear(gru, 1)            # dopamine: write gate
        self.w_ret = nn.Linear(gru, 1)             # octopamine: retention
        self.w_key = nn.Linear(gru, dk)
        self.w_val = nn.Linear(gru, dv)
        self.q_gru = nn.GRU(d, gru, batch_first=True)
        self.q_key = nn.Linear(gru, dk)
        self.r_up = nn.Linear(dv, d, bias=False)
        nn.init.zeros_(self.r_up.weight)           # exact no-op at step 0
        self.scale = dk ** -0.5

    def build_memory(self, x, cpre):
        """Fast-weight write over the pre-cue region. Returns M, z."""
        B = x.shape[0]
        e = self.emb(x[:, :cpre]) + self.pos[:cpre]
        h, _ = self.writer(e)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)          # (B,T) dopamine
        k = torch.tanh(self.w_key(h))                          # (B,T,dk)
        v = torch.tanh(self.w_val(h))                          # (B,T,dv)
        if self.mem == "fw2":
            a = torch.sigmoid(self.w_ret(h)).squeeze(-1)       # (B,T) octopamine
        else:
            a = torch.ones_like(g)                             # fw1: no decay
        M = x.new_zeros(B, self.dv, self.dk)
        z = x.new_zeros(B, self.dk)
        for t in range(cpre):                                  # sequential store
            at = a[:, t].view(B, 1, 1)
            gt = g[:, t].view(B, 1)
            M = at * M + gt.view(B, 1, 1) * torch.einsum(
                "bv,bk->bvk", v[:, t], k[:, t])
            z = a[:, t].view(B, 1) * z + gt * k[:, t]
        # quantise the persistent store to the same budget the slots used
        M = quantise_fixed(torch.tanh(M), self.qbits)
        return M, z

    def read(self, x, cpre, M, z):
        q0 = cpre                                              # query starts here
        qe = self.emb(x[:, q0:q0 + QUERY_L]) + self.pos[:QUERY_L]
        hq, _ = self.q_gru(qe)
        q = torch.tanh(self.q_key(hq[:, 1 + NAME_L]))          # (B,dk)
        num = torch.einsum("bvk,bk->bv", M, q)                 # M q
        den = (z * q).sum(-1, keepdim=True) + 1e-3
        return num / den                                       # (B,dv)

    def forward(self, x, cpre):
        B, Nx = x.shape
        M, z = self.build_memory(x, cpre)
        r = self.read(x, cpre, M, z)
        h = self.emb(x) + self.pos[:Nx]
        h = h.clone()
        h[:, cpre:] = h[:, cpre:] + self.r_up(r).unsqueeze(1)
        mask = torch.ones(Nx, Nx, dtype=torch.bool, device=x.device)
        for i in range(Nx):
            lo = max(0, i - self.window + 1)
            mask[i, lo:i + 1] = False
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_out(h))


def run(mem, N, carrier, steps, seed, device, B=16, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = FlyNonceLM(mem, N).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y, spans, tgt, SEQ = make_batch(B, N, carrier, rng, device)
        cpre = int(spans[:, -1, 1].max())          # write region ends at last fact
        # align cpre to the actual query start used by make_batch:
        cpre = x.shape[1] - (QUERY_L + ANS_L - 1)
        logits = m(x, cpre)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(99)
    ok = n = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, tgt, _ = make_batch(64, N, carrier, ev, device)
            cpre = x.shape[1] - (QUERY_L + ANS_L - 1)
            pred = m(x, cpre).argmax(-1)
            sel = y != -100
            ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--arms", nargs="+", default=["fw1", "fw2"])
    ap.add_argument("--out", default="fly-memory.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()

    print(f"fly synaptic store vs recall cliff · 256-number memory · "
          f"{a.steps} steps · {a.seeds} seeds · {dev}")
    print(f"(slot-bank cliff for reference from v28: N=4 49%, N=8 1.4%)\n")
    print(f"  {'arm':<6}{'N':>4}{'recall med':>12}{'conv':>6}   per-seed")
    print("  " + "-" * 48)
    res = {}
    for mem in a.arms:
        for N in a.entities:
            accs = [run(mem, N, carrier, a.steps, s, dev)
                    for s in range(a.seeds)]
            med = float(np.median(accs))
            conv = sum(v >= 0.20 for v in accs)
            print(f"  {mem:<6}{N:>4}{med:>11.1%}{conv:>4}/{a.seeds}   "
                  + " ".join(f"{v:.2f}" for v in accs), flush=True)
            res[f"{mem}-N{N}"] = dict(acc=accs, median=med, converged=conv)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
