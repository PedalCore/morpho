"""v2c — POST-CUE associative retrieval: store first, find out what
mattered later.

v2a's admission result had the cue BEFORE the chunks: the writer knew
what mattered and stored only that. This reverses it. Four random chunks
pass, THEN the cue names which one is wanted. The writer must decide what
to preserve without knowing which candidate will be queried — the point
where a selective cache has to become an associative memory.

BUDGET, fixed across all arms at the v2b sweet spot: K=4 slots, d_slot=16,
q=8 -> 512 persistent bits for ~384 bits of candidate entropy (4 chunks x
96 bits). NOTE THE DIRECTION OF THIS INEQUALITY (review correction): 512
bits EXCEEDS the candidate entropy, so storing all four candidates
losslessly is information-theoretically possible in principle. If the
system fails to preserve all four, that is coding/optimisation
inefficiency - a useful measurement - and NOT forced forgetting. The
budget pressures the code (v2b showed ~52% token recall at 1.3x entropy
for ONE chunk), but "the budget forces forgetting" would be an
overclaim and is not the claim.

CAUSALITY GUARD. The writer consumes ONLY the region before the cue
(chunks + filler). Its GRU state at cue time could otherwise still carry
recent content forward and write it after peeking at the cue — a side
channel around "decide before knowing". The reader's window (32) covers
cue + recall, so reading is target-aware; writing provably is not.

ARMS (write machinery identical; only the selective channels differ):

  none      no slots — the floor.
  funiform  gate pinned to 1 AND uniform address: true write-everything.
  gated     free learned write (v2a's machinery). The gate can still be
            chunk-vs-filler selective, but NOT target-selective.
  kvaddr    address pinned to chunk identity (chunk i -> slot i; the ID
            marker drives a one-hot address; filler follows the last seen
            marker). Gate and value learned. A structural key-value store:
            does identity-addressing beat free-form allocation?

PRE-REGISTERED:
  1. none ~ 1.6%.
  2. Every slot arm lands WELL below the pre-cued equivalent at this
     budget (d16/q8 pre-cue: 76.0%) — the price of not knowing.
  3. kvaddr >= gated: when everything must be kept, structured addressing
     should beat learned allocation, which no longer has an admission
     signal to exploit.
  4. INTEGRITY: the gated arm's write-gate ratio target/other-chunks ~ 1.
     The writer cannot know the target; a ratio >> 1 means the causality
     guard leaks, and the run is invalid.
  5. Exact whole-chunk recall stays negligible - expected from v2b's
     coding inefficiency (exact recall collapsed even at generous
     budgets), NOT required by information theory: 512 > 384 bits, so a
     perfect code could in principle keep all four candidates.

    python slot_binding4.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_binding2 import VOCAB, C, NCH, IDS, CUE, RECALL, NTOK, Block
from slot_binding3 import quantise_fixed

FREE, FINAL_MIN = 60, 40
CPRE = NCH * (1 + C) + FREE + FINAL_MIN          # writer-visible region
SEQ = CPRE + 3 + C                               # + CUE, ID_t, RECALL, recall


def make_batch(B, rng, device):
    xs = np.zeros((B, SEQ), dtype=np.int64)
    spans = np.zeros((B, NCH, 2), dtype=np.int64)
    tgt = rng.integers(0, NCH, size=B)
    chunks = rng.integers(0, VOCAB, size=(B, NCH, C))
    for b in range(B):
        order = rng.permutation(NCH)
        gaps = rng.multinomial(FREE, [1 / NCH] * NCH)
        p = 0
        for gi, ci in enumerate(order):
            fill = rng.integers(0, VOCAB, size=gaps[gi])
            xs[b, p:p + gaps[gi]] = fill; p += gaps[gi]
            xs[b, p] = IDS[ci]; p += 1
            xs[b, p:p + C] = chunks[b, ci]
            spans[b, ci] = (p, p + C); p += C
        xs[b, p:CPRE] = rng.integers(0, VOCAB, size=CPRE - p)
        xs[b, CPRE] = CUE
        xs[b, CPRE + 1] = IDS[tgt[b]]
        xs[b, CPRE + 2] = RECALL
        xs[b, CPRE + 3:] = chunks[b, tgt[b]]
    x = torch.from_numpy(xs[:, :-1]).to(device)
    y = torch.full((B, SEQ - 1), -100, dtype=torch.long)
    y[:, CPRE + 2:] = torch.from_numpy(
        np.stack([chunks[b, tgt[b]] for b in range(B)]))
    return (x, y.to(device), torch.from_numpy(spans).to(device),
            torch.from_numpy(tgt).to(device))


class SlotLM4(nn.Module):
    def __init__(self, mode, K=4, d=128, d_slot=16, qbits=8, layers=3,
                 heads=4, window=32, gru=64):
        super().__init__()
        self.mode, self.K, self.window, self.qbits = mode, K, window, qbits
        self.emb = nn.Embedding(NTOK, d)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        if mode != "none":
            self.writer = nn.GRU(d, gru, batch_first=True)
            self.w_gate = nn.Linear(gru, 1)
            self.w_addr = nn.Linear(gru, K)
            self.w_val = nn.Linear(gru, d_slot)
            self.slot_up = nn.Linear(d_slot, d, bias=False)

    def build_slots(self, x):
        if self.mode == "none":
            return None, None
        pre = x[:, :CPRE]                        # writer NEVER sees the cue
        e = self.emb(pre) + self.pos[:CPRE]
        h, _ = self.writer(e)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)
        a = torch.softmax(self.w_addr(h), dim=-1)
        if self.mode == "funiform":
            g = torch.ones_like(g)
            a = torch.full_like(a, 1.0 / self.K)
        elif self.mode == "kvaddr":
            # identity addressing: each position follows the most recent ID
            # marker; chunk i -> slot i. Positions before any marker spread
            # uniformly. Gate stays learned.
            B, T = pre.shape
            a = torch.zeros(B, T, self.K, device=x.device)
            cur = torch.full((B,), -1, dtype=torch.long, device=x.device)
            for t in range(T):
                for ci in range(NCH):
                    cur = torch.where(pre[:, t] == IDS[ci],
                                      torch.full_like(cur, ci), cur)
                hot = F.one_hot(cur.clamp_min(0), self.K).float()
                none_yet = (cur < 0).float().unsqueeze(-1)
                a[:, t] = (1 - none_yet) * hot + none_yet / self.K
        w = g.unsqueeze(-1) * a
        num = torch.einsum("btk,btd->bkd", w, self.w_val(h))
        den = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        s = quantise_fixed(torch.tanh(num / den), self.qbits)
        return self.slot_up(s), g

    def forward(self, x):
        B, N = x.shape
        slots, g = self.build_slots(x)
        h = self.emb(x) + self.pos[:N]
        K = 0 if slots is None else slots.shape[1]
        if slots is not None:
            h = torch.cat([slots, h], dim=1)
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


def gate_ratio(g, spans, tgt):
    t_, o_ = [], []
    for b in range(g.shape[0]):
        for c in range(NCH):
            lo, hi = int(spans[b, c, 0]), int(spans[b, c, 1])
            (t_ if c == int(tgt[b]) else o_).append(float(g[b, lo:hi].mean()))
    return float(np.mean(t_)), float(np.mean(o_))


def run(mode, steps, seed, device, B=64, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = SlotLM4(mode).to(device)
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
    tok = tot = seq_ok = seq_n = 0
    gt = go = 0.0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, tgt = make_batch(64, ev, device)
            logits, g = m(x)
            pred = logits.argmax(-1)
            sel = y != -100
            tok += int((pred[sel] == y[sel]).sum()); tot += int(sel.sum())
            seq_ok += int((((pred == y) | ~sel).all(dim=1)).sum())
            seq_n += y.shape[0]
            if g is not None:
                a, b_ = gate_ratio(g.cpu(), spans.cpu(), tgt.cpu())
                gt += a / 8; go += b_ / 8
    return tok / tot, seq_ok / seq_n, (gt, go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--arms", nargs="+",
                    default=["none", "funiform", "gated", "kvaddr"])
    ap.add_argument("--out", default="slot-postcue.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"POST-CUE · {NCH}x{C}-token chunks (~{NCH * C * 6}b candidate "
          f"entropy) · budget 4x16x8 = 512b · window 32 · chance {1/VOCAB:.1%}")
    print(f"{a.steps} steps · {a.seeds} seeds · {dev}\n")
    print(f"  {'arm':<10}{'tok med':>9}{'exact med':>11}{'conv':>6}"
          f"{'g tgt':>8}{'g oth':>8}   per-seed")
    print("  " + "-" * 72)
    res = {}
    for mode in a.arms:
        toks, exacts, gs = [], [], []
        for s in range(a.seeds):
            t, e, g = run(mode, a.steps, s, dev)
            toks.append(t); exacts.append(e); gs.append(g)
        med = float(np.median(toks))
        mede = float(np.median(exacts))
        conv = sum(t >= 0.20 for t in toks)
        gt = float(np.mean([g[0] for g in gs]))
        go = float(np.mean([g[1] for g in gs]))
        seeds_s = " ".join(f"{t:.2f}" for t in toks)
        print(f"  {mode:<10}{med:>8.1%}{mede:>10.1%}{conv:>4}/{a.seeds}"
              f"{gt:>8.3f}{go:>8.3f}   {seeds_s}", flush=True)
        res[mode] = dict(token=toks, exact=exacts, median=med,
                         gate_tgt=gt, gate_other=go, converged=conv)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
