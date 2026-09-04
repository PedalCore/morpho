"""v2a — cache ADMISSION: can the model learn WHAT deserves a slot?

v1 established binding: a learned streaming write recalls a random chunk
at 99.8% after it leaves the attention window (floor and sink control at
chance). But v1's chunk sat at a fixed position, so the write could be
position-keyed. This asks the frozen v2a question:

    can the model learn what deserves a slot, rather than merely where
    to write?

TASK. A pre-cue names which of four chunk identities will matter. Four
random 16-token chunks then appear, each prefixed by its identity marker,
in RANDOM ORDER at RANDOM POSITIONS with random filler between. A final
filler of at least 40 tokens (window is 32) puts every chunk out of reach
before RECALL asks for the cued one. Chunk contents, order, positions,
filler lengths and the target identity are all randomised per sequence —
nothing is learnable in weights. Chance is 1/64.

The cue precedes the chunks, so the WRITER must remember it. v1's write
was per-token and could not. The writer here is a minimal streaming GRU
over the prefix — causal, cheap, and deliberately NOT the transformer's
own hidden states, so the write path remains a separate streaming
mechanism. Write at position t: scalar gate g_t = sigmoid(.), address
a_t = softmax over K slots, value v_t — all from the GRU state.

ARMS (three, as frozen in review):

  none     windowed, no slots — the floor.
  forced   g_t = 1 everywhere: every position is written, address/value
           still learned. Tests whether finite slots can retain
           everything; with 4 chunks + filler competing for K slots the
           budget should bind structurally.
  gated    the full learned write — the selective-memory system.

THE HEADLINE DIAGNOSTIC is not accuracy: it is the write gate logged
around each candidate chunk. g_target >> g_distractor is direct evidence
of learned cache admission — allocation you can see, not a task that
happened to be solved. One caveat pre-registered: selectivity could live
in the ADDRESS rather than the gate (route target to a read slot,
distractors to a junk slot), so per-chunk write mass is logged as well;
"admission by address" is a pass with a different mechanism, and the
instrument must not assume one mechanism (the four-losses lesson).

PRE-REGISTERED:
  1. none ~ 1.6%.
  2. gated > forced, with the gap widening as K shrinks (run at K=4 and
     K=2 so the budget provably binds somewhere).
  3. If gated succeeds: mean gate on the cued chunk's tokens > 2x the
     mean on non-target chunks. If instead gates are flat and the win is
     in the address, report admission-by-address explicitly.

    python slot_binding2.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, C, NCH = 64, 16, 4          # content alphabet, chunk len, n chunks
IDS = [VOCAB + i for i in range(NCH)]           # identity markers
CUE, RECALL = VOCAB + NCH, VOCAB + NCH + 1
NTOK = VOCAB + NCH + 2
FINAL_MIN, FREE = 40, 60           # final gap >= window; extra filler spread
PRE = 2 + NCH * (1 + C) + FINAL_MIN + FREE      # cue..end-of-filler
SEQ = PRE + 1 + C


def make_batch(B, rng, device):
    xs = np.zeros((B, SEQ), dtype=np.int64)
    spans = np.zeros((B, NCH, 2), dtype=np.int64)   # chunk token spans
    tgt = rng.integers(0, NCH, size=B)
    chunks = rng.integers(0, VOCAB, size=(B, NCH, C))
    for b in range(B):
        order = rng.permutation(NCH)
        gaps = rng.multinomial(FREE, [1 / (NCH + 1)] * (NCH + 1))
        p = 0
        xs[b, p] = CUE; xs[b, p + 1] = IDS[tgt[b]]; p += 2
        for gi, ci in enumerate(order):
            p += gaps[gi]                                # random filler
            xs[b, p] = IDS[ci]; p += 1
            xs[b, p:p + C] = chunks[b, ci]
            spans[b, ci] = (p, p + C); p += C
        # remaining filler (incl. the guaranteed >= FINAL_MIN tail)
        xs[b, p:PRE] = 0
        fill = rng.integers(0, VOCAB, size=PRE - p - gaps[NCH:].sum())
        xs[b, p:p + len(fill)] = fill
        p += len(fill)
        xs[b, p:PRE] = rng.integers(0, VOCAB, size=PRE - p)
        xs[b, PRE] = RECALL
        xs[b, PRE + 1:] = chunks[b, tgt[b]]
    # every non-chunk, non-marker prefix position is filler; overwrite the
    # zeros left where gaps were with random filler tokens
    x = torch.from_numpy(xs[:, :-1]).to(device)
    y = torch.full((B, SEQ - 1), -100, dtype=torch.long)
    y[:, PRE:] = torch.from_numpy(
        np.stack([chunks[b, tgt[b]] for b in range(B)]))
    return (x, y.to(device), torch.from_numpy(spans).to(device),
            torch.from_numpy(tgt).to(device))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


class SlotLM2(nn.Module):
    def __init__(self, mode, K=4, d=128, layers=3, heads=4, window=32,
                 gru=64):
        super().__init__()
        self.mode, self.K, self.window = mode, K, window
        self.emb = nn.Embedding(NTOK, d)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        if mode != "none":
            self.writer = nn.GRU(d, gru, batch_first=True)
            self.w_gate = nn.Linear(gru, 1)
            self.w_addr = nn.Linear(gru, K)
            self.w_val = nn.Linear(gru, d)

    def build_slots(self, x):
        if self.mode == "none":
            return None, None
        e = self.emb(x[:, :PRE]) + self.pos[:PRE]
        h, _ = self.writer(e)                            # (B, PRE, gru)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)    # (B, PRE)
        if self.mode in ("forced", "funiform"):
            g = torch.ones_like(g)
        a = torch.softmax(self.w_addr(h), dim=-1)        # (B, PRE, K)
        if self.mode == "funiform":
            # the TRUE write-everything control: the learned address gave
            # "forced" an admission channel (route cued chunk to one slot,
            # junk to another) - a forced-K2 seed reached 51.8% that way.
            # Uniform address removes every selective channel.
            a = torch.full_like(a, 1.0 / self.K)
        v = self.w_val(h)                                # (B, PRE, d)
        w = g.unsqueeze(-1) * a                          # (B, PRE, K)
        num = torch.einsum("btk,btd->bkd", w, v)
        den = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        return num / den, g

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


def gate_diagnostic(g, spans, tgt):
    """Mean gate on cued-chunk tokens vs other chunks vs filler."""
    B = g.shape[0]
    m_t, m_o, m_f = [], [], []
    for b in range(B):
        sel = torch.zeros(PRE, dtype=torch.bool)
        for c in range(NCH):
            lo, hi = int(spans[b, c, 0]), int(spans[b, c, 1])
            m = g[b, lo:hi].mean()
            (m_t if c == int(tgt[b]) else m_o).append(float(m))
            sel[lo:hi] = True
        m_f.append(float(g[b, ~sel].mean()))
    return (float(np.mean(m_t)), float(np.mean(m_o)), float(np.mean(m_f)))


def run(mode, K, steps, seed, device, B=64, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = SlotLM2(mode, K=K).to(device)
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
    ok = tot = 0
    gt = go = gf = 0.0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, tgt = make_batch(64, ev, device)
            logits, g = m(x)
            sel = y != -100
            ok += int((logits.argmax(-1)[sel] == y[sel]).sum())
            tot += int(sel.sum())
            if g is not None:
                a, b_, c = gate_diagnostic(g.cpu(), spans.cpu(), tgt.cpu())
                gt += a; go += b_; gf += c
    n = sum(p.numel() for p in m.parameters())
    return ok / tot, (gt / 8, go / 8, gf / 8) if mode != "none" else None, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--slots", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--out", default="slot-admission.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{NCH} chunks x {C} tokens · random order/positions · window 32 · "
          f"final gap >= {FINAL_MIN} · chance {1 / VOCAB:.1%}")
    print(f"{a.steps} steps · {a.seeds} seeds · {dev}\n")
    print(f"  {'arm':<8}{'K':>3}{'recall':>9}{'spread':>8}"
          f"{'gate tgt':>10}{'gate other':>11}{'gate fill':>10}")
    print("  " + "-" * 60)
    res = {}
    for K in a.slots:
        for mode in ("none", "funiform", "forced", "gated"):
            if mode == "none" and K != a.slots[0]:
                continue                       # floor doesn't depend on K
            accs, gates = [], []
            for s in range(a.seeds):
                acc, gd, n = run(mode, K, a.steps, s, dev)
                accs.append(acc)
                if gd: gates.append(gd)
            mu, sd = float(np.mean(accs)), float(np.std(accs))
            if gates:
                g = np.mean(gates, axis=0)
                gs = f"{g[0]:>10.3f}{g[1]:>11.3f}{g[2]:>10.3f}"
            else:
                g, gs = None, f"{'—':>10}{'—':>11}{'—':>10}"
            print(f"  {mode:<8}{K:>3}{mu:>8.1%}{sd:>8.3f}{gs}", flush=True)
            res[f"{mode}-K{K}"] = {"acc": accs, "mean": mu, "std": sd,
                                   "gate_tgt_other_fill":
                                       None if g is None else list(g)}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
