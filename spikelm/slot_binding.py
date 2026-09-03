"""Can a transformer learn to BIND a chunk to a slot and recall it after
the chunk has left its attention window?

The idea under test (an "ephemeral token dictionary"): give the model a
few mutable slots that mean nothing until, within a sequence, it chooses
to write something into them — a per-sequence symbol table. The claimed
wins are (a) repeated/needed content becomes one compressed object plus
cheap references instead of resident K/V for every original token, and
(b) a new internal axis the model can organise however it likes.

WHY THE TASK IS BUILT THE WAY IT IS. A full-attention transformer is
already a pointer machine — induction heads copy repeated spans natively —
so on task loss alone every arm would tie and the experiment would measure
nothing. The constraint has to bite: attention here is SLIDING-WINDOW
(W=32) and the chunk sits G=64 tokens behind the recall site. At recall
time the chunk is unreachable except through a slot. Chance is 1/64.

Task: [chunk: C=16 uniform-random tokens] [filler: G random tokens]
[CUE] [recall the chunk, teacher-forced]. Loss only on the recall region.
Chunks are random per sequence, so nothing can be memorised in weights.

ARMS — each isolates one link in the chain:

  full     no slots, full attention.      Can the task be learned at all?
  none     no slots, windowed.            The floor: recall via nothing.
  static   K learned constant slots.      Attention-sink control: always-
                                          visible KV entries help attention
                                          numerically even with no content.
  oracle   slots written BY US from the true chunk (role-tagged sums, so
           order is recoverable).         Is the READ path learnable when
                                          the write is perfect?
  dynamic  slots written by a learned, streaming, causal write over the
           pre-cue prefix.                THE TEST: does the model learn
                                          to bind?

PRE-REGISTERED, before the first run:
  1. full  > 95%   (else the task is broken, and nothing else is readable)
  2. none  ~ 1.6%  (chance; if higher, the window leaks)
  3. static ~ none (if static helps materially, sinks confound everything)
  4. oracle > 90%  (read path learnable from role-tagged sums)
  5. THE TEST — dynamic lands near oracle: binding is learnable end to
     end. Dynamic near none with oracle high: the WRITE is the hard part,
     which is exactly where the SyncLM co-adaptation failure predicts
     trouble. Either way the chain localises the failure.

Scope, stated up front: v1 fixes the chunk at positions 0..C-1, so the
dynamic write may learn position-keyed copying rather than content
detection. That demonstrates the mechanism, not the discovery of WHAT to
cache; moving/multiple chunks and a memory price are v2, and only after
this chain passes.

    python slot_binding.py
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

C, G, VOCAB = 16, 64, 64          # chunk length, gap, token alphabet
CUE = VOCAB                        # one extra token id
SEQ = C + G + 1 + C                # chunk · filler · cue · recall


def make_batch(B, rng, device):
    chunk = rng.integers(0, VOCAB, size=(B, C))
    filler = rng.integers(0, VOCAB, size=(B, G))
    cue = np.full((B, 1), CUE)
    s = np.concatenate([chunk, filler, cue, chunk], axis=1)
    x = torch.from_numpy(s[:, :-1]).long().to(device)
    y = torch.full((B, SEQ - 1), -100, dtype=torch.long)
    y[:, C + G:] = torch.from_numpy(chunk).long()   # predict recall region only
    return x, y.to(device), torch.from_numpy(chunk).long().to(device)


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


class SlotLM(nn.Module):
    """Tiny causal transformer; K slot entries prepended as extra positions.

    Slots are computed BEFORE the transformer from the pre-cue prefix, so
    the recall region reads them but cannot write them, and the write path
    is streaming/causal by construction (a cumulative gated sum).
    """

    def __init__(self, mode, K=4, d=128, layers=3, heads=4, window=32):
        super().__init__()
        self.mode, self.K, self.window = mode, K, window
        self.emb = nn.Embedding(VOCAB + 1, d)
        self.pos = nn.Parameter(torch.randn(SEQ, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        if mode == "static":
            self.slots = nn.Parameter(torch.randn(K, d) * 0.02)
        if mode == "dynamic":
            self.w_gate = nn.Linear(2 * d, K)      # per-slot write strength
            self.w_val = nn.Linear(2 * d, d)       # what gets written
        if mode == "oracle":
            # fixed random role matrices tag within-group order, so the sum
            # is invertible in principle and the READ path is what's tested
            P = torch.randn(C // K, d, d) / math.sqrt(d)
            self.register_buffer("roles", P)

    def build_slots(self, x, chunk):
        B, d = x.shape[0], self.emb.embedding_dim
        if self.mode in ("full", "none"):
            return None
        if self.mode == "static":
            return self.slots.unsqueeze(0).expand(B, -1, -1)
        if self.mode == "oracle":
            e = self.emb(chunk)                              # (B, C, d)
            grp = e.view(B, C // self.K, self.K, d)          # rank, slot, d
            tagged = torch.einsum("brkd,rde->brke", grp, self.roles)
            return tagged.sum(dim=1)                         # (B, K, d)
        # dynamic: gated cumulative write over the pre-cue prefix
        pre = x[:, :C + G]
        e = self.emb(pre) + self.pos[:C + G]
        feat = torch.cat([e, self.pos[:C + G].expand(B, -1, -1)], dim=-1)
        gate = torch.sigmoid(self.w_gate(feat))              # (B, T, K)
        val = self.w_val(feat)                               # (B, T, d)
        num = torch.einsum("btk,btd->bkd", gate, val)
        den = gate.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        return num / den

    def forward(self, x, chunk):
        B, N = x.shape
        h = self.emb(x) + self.pos[:N]
        slots = self.build_slots(x, chunk)
        K = 0 if slots is None else slots.shape[1]
        if slots is not None:
            h = torch.cat([slots, h], dim=1)
        L = K + N
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        for i in range(N):                       # token i sees window + slots
            lo = max(0, i - self.window + 1) if self.mode != "full" else 0
            mask[K + i, K + lo:K + i + 1] = False
            mask[K + i, :K] = False
        mask[:K, :K] = False                     # slots see themselves
        for b in self.blocks:
            h = b(h, mask)
        return self.head(self.ln_out(h[:, K:]))


def run(mode, steps, seed, device, B=64, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = SlotLM(mode).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x, y, chunk = make_batch(B, rng, device)
        loss = F.cross_entropy(m(x, chunk).reshape(-1, VOCAB),
                               y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(99)
    ok = tot = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, chunk = make_batch(64, ev, device)
            pred = m(x, chunk).argmax(-1)
            sel = y != -100
            ok += int((pred[sel] == y[sel]).sum()); tot += int(sel.sum())
    n = sum(p.numel() for p in m.parameters())
    return ok / tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arms", nargs="+",
                    default=["full", "none", "static", "oracle", "dynamic"])
    ap.add_argument("--out", default="slot-binding.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"chunk {C} · gap {G} · window 32 · vocab {VOCAB} · "
          f"{a.steps} steps · {a.seeds} seeds · {dev}")
    print(f"chance {1 / VOCAB:.1%} · KV retained per token: windowed 32+4 "
          f"slots vs full {SEQ}\n")
    print(f"  {'arm':<10}{'recall acc':>12}{'spread':>9}{'params':>10}")
    print("  " + "-" * 43)
    res = {}
    for mode in a.arms:
        accs, n = [], 0
        for s in range(a.seeds):
            acc, n = run(mode, a.steps, s, dev)
            accs.append(acc)
        mu, sd = float(np.mean(accs)), float(np.std(accs))
        print(f"  {mode:<10}{mu:>11.1%}{sd:>9.3f}{n:>10,}", flush=True)
        res[mode] = {"acc": accs, "mean": mu, "std": sd, "params": n}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
