"""v2b — how much writable memory does the learned cache actually need?

v2a established admission (84x gate ratio, ladder 5.4 -> 16.1 -> 79.9).
This measures the cost: recall against writable slot capacity, in two
nested phases per the frozen review:

  PHASE 1 (dimension): K=4 fixed, sweep d_slot in {128..1}, float32
      slots. This measures the DIMENSIONAL BOTTLENECK only - it is not
      called compression, because unrestricted floats can hide arbitrary
      information in analog precision.
  PHASE 2 (precision): at the knee widths, quantise the PERSISTENT SLOT
      STATE itself to q bits/scalar. Only then is the writable budget
      real:  B = K * d_slot * q bits,  comparable to the chunk's ~96 bits
      of entropy (16 tokens x log2(64)).

SIDE-CHANNEL LEDGER, audited by construction:
  * the persistent sequence-specific state is EXACTLY K x d_slot scalars:
    slots are stored at d_slot and up-projected to the transformer width
    by a learned matrix (weights are not sequence-specific);
  * the GRU writer's hidden state is transient and feeds only the slot
    accumulation; gate and address are consumed into the sum and do not
    persist; nothing else crosses the window boundary (sliding-window
    mask discards all K/V of the forgotten region);
  * chunks are random per sequence, so no positional cue carries content;
  * quantisation uses a FIXED grid on tanh-bounded slots - a per-sequence
    scale would itself be a ~32-bit side channel per slot. The tanh bound
    is applied in ALL v2b arms including float32, so phases differ only
    in q. (Sanity: d_slot=128 float should land near v2a's 79.9%.)

METRICS per configuration: token recall AND exact whole-chunk recall
(with random chunks, exact reconstruction must collapse once the budget
falls below source entropy - if it does not, there is an unaudited side
channel). Reported as median, full per-seed list, and convergence rate
(seed counts as converged at token recall >= 20%); v2a's bimodality
showed means over mixed basins describe no real run.

    python slot_binding3.py --widths 128 64 32 16 8 4 2 1
    python slot_binding3.py --widths 8 4 --qbits 16 8 4 2 1
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_binding2 import (VOCAB, C, NCH, PRE, SEQ, make_batch, Block)


def quantise_fixed(s, qbits):
    """Fixed-grid uniform quantisation on [-1, 1], straight-through.

    No data-dependent scale anywhere: the grid is the same for every
    sequence, so the quantised slot state carries exactly qbits/scalar.
    """
    if qbits is None:
        return s
    levels = 2 ** qbits
    q = torch.round((s + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    return s + (q - s).detach()


class SlotLM3(nn.Module):
    """The gated system from v2a, with slot width and precision as dials."""

    def __init__(self, K=4, d=128, d_slot=128, qbits=None, layers=3,
                 heads=4, window=32, gru=64):
        super().__init__()
        self.K, self.window, self.qbits = K, window, qbits
        from slot_binding2 import NTOK
        self.emb = nn.Embedding(NTOK, d)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.writer = nn.GRU(d, gru, batch_first=True)
        self.w_gate = nn.Linear(gru, 1)
        self.w_addr = nn.Linear(gru, K)
        self.w_val = nn.Linear(gru, d_slot)
        self.slot_up = nn.Linear(d_slot, d, bias=False)

    def build_slots(self, x):
        e = self.emb(x[:, :PRE]) + self.pos[:PRE]
        h, _ = self.writer(e)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)
        a = torch.softmax(self.w_addr(h), dim=-1)
        v = self.w_val(h)
        w = g.unsqueeze(-1) * a
        num = torch.einsum("btk,btd->bkd", w, v)
        den = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        s = torch.tanh(num / den)                # bounded: fixed-grid safe
        s = quantise_fixed(s, self.qbits)        # the PERSISTENT state
        return self.slot_up(s), g

    def forward(self, x):
        B, N = x.shape
        slots, g = self.build_slots(x)
        h = torch.cat([slots, self.emb(x) + self.pos[:N]], dim=1)
        K = self.K
        L = K + N
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        for i in range(N):
            lo = max(0, i - self.window + 1)
            mask[K + i, K + lo:K + i + 1] = False
            mask[K + i, :K] = False
        mask[:K, :K] = False
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_out(h[:, K:])), g


def run(d_slot, qbits, K, steps, seed, device, B=64, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = SlotLM3(K=K, d_slot=d_slot, qbits=qbits).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y, _, _ = make_batch(B, rng, device)
        logits, _ = m(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(99)
    tok_ok = tok_n = seq_ok = seq_n = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, _, _ = make_batch(64, ev, device)
            pred = m(x)[0].argmax(-1)
            sel = y != -100
            tok_ok += int((pred[sel] == y[sel]).sum()); tok_n += int(sel.sum())
            hit = ((pred == y) | ~sel).all(dim=1)
            seq_ok += int(hit.sum()); seq_n += y.shape[0]
    return tok_ok / tok_n, seq_ok / seq_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+",
                    default=[128, 64, 32, 16, 8, 4, 2, 1])
    ap.add_argument("--qbits", type=int, nargs="+", default=[0],
                    help="0 = float32 (phase 1); else bits/scalar (phase 2)")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="slot-capacity.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"K={a.K} · chunk entropy ~{C * 6} bits · {a.steps} steps · "
          f"{a.seeds} seeds · chance {1 / VOCAB:.1%} · {dev}\n")
    print(f"  {'d_slot':>7}{'q':>5}{'budget':>10}{'tok med':>9}{'exact med':>11}"
          f"{'conv':>6}   per-seed token recall")
    print("  " + "-" * 75)
    res = {}
    for q in a.qbits:
        qb = None if q == 0 else q
        for d_slot in a.widths:
            toks, exacts = [], []
            for s in range(a.seeds):
                t, e = run(d_slot, qb, a.K, a.steps, s, dev)
                toks.append(t); exacts.append(e)
            med_t = float(np.median(toks))
            med_e = float(np.median(exacts))
            conv = sum(t >= 0.20 for t in toks)
            budget = (f"{a.K * d_slot * q}b" if qb else
                      f"{a.K * d_slot}f32")
            seeds_s = " ".join(f"{t:.2f}" for t in toks)
            print(f"  {d_slot:>7}{q if qb else '-':>5}{budget:>10}"
                  f"{med_t:>8.1%}{med_e:>10.1%}{conv:>4}/{a.seeds}"
                  f"   {seeds_s}", flush=True)
            res[f"d{d_slot}-q{q}"] = dict(
                token=toks, exact=exacts, median_token=med_t,
                median_exact=med_e, converged=conv,
                budget_bits=(a.K * d_slot * q) if qb else None)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
