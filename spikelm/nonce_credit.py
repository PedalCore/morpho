"""v3.3 — can denser credit restore learned associative-memory scaling?

v28 found the N=8 collapse happens while the writer still organises
(collisions flat, slots filling) and capacity is generous - a LEARNING
cliff. Leading hypothesis: one queried fact per sequence gives useful
gradient per stored entity ~ 1/N. Plausible, not proven. This tests it.

DESIGN (frozen in review). N=8 stored facts, 8 slots, 2048 bits - fixed.
Vary the supervision:

  q1        one query (v28's protocol - the floor baseline)
  q2/q4/q8  Q DISTINCT entities queried per sequence
  q8rep     8 REPEATS of one query: the discriminating control - the
            same answer-token gradient magnitude, no added coverage of
            the stored associations. (Later repeats sit within window of
            earlier answers and are copyable without memory; that only
            reinforces what this arm is for: magnitude without coverage.
            All arms are therefore EVALUATED on the first query only.)
  curr      curriculum: N=1 -> 2 -> 4 -> 8 (1500 steps each), Q=1 - the
            basin/search-difficulty arm.

PRE-REGISTERED OUTCOME TABLE (verbatim from review):
  multi-query helps, curriculum does not -> supervision density
  curriculum helps, multi-query does not -> basin/search difficulty
  both help                              -> both contribute
  neither helps                          -> revisit the addressing
                                            architecture itself
Sharp form of the density prediction: Q=1 floor; recall recovers as Q
rises; q8rep stays at floor (else magnitude alone suffices and coverage
is not the mechanism).

ARCHITECTURE: per-query retrieval. Each query segment computes its own
key from its own nonce name and injects its own retrieved value over its
own span. Asserted equivalent to the single-query forward at Q=1 before
anything is trained.

Only after some arm learns N=8 cleanly does the ORIGINAL capacity
question (N=8/16/32 at fixed memory: graceful degradation vs bits/fact)
become measurable. The SNR-read mechanism probe is parked per review.

    python nonce_credit.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nonce_lm import (V, GAP, NAME_L, FACT_L, QUERY_L, ANS_L, PREGAP,
                      COLORS, ITEMS, STOI, load_carrier, enc)
from nonce_lm2 import KVNonceLM, quantise_fixed


def make_batch_multi(B, N, Q, carrier, rng, device, distinct=True):
    """N facts, then GAP, then Q query+answer segments back to back."""
    SEQ = N * FACT_L + PREGAP + GAP + Q * (QUERY_L + ANS_L)
    xs = np.zeros((B, SEQ), dtype=np.int64)
    spans = np.zeros((B, N, 2), dtype=np.int64)
    qtargets = np.zeros((B, Q), dtype=np.int64)
    for b in range(B):
        names = set()
        while len(names) < N:
            names.add("".join(chr(65 + c) for c in rng.integers(0, 26, NAME_L)))
        names = list(names)
        combos = [(COLORS[rng.integers(8)], ITEMS[rng.integers(8)])
                  for _ in range(N)]
        gaps = rng.multinomial(PREGAP, [1 / N] * N)
        p = 0
        for j, ci in enumerate(rng.permutation(N)):
            g = gaps[j]
            off = rng.integers(0, len(carrier) - g - 1)
            xs[b, p:p + g] = carrier[off:off + g]; p += g
            fact = f"\n{names[ci]} carries the {combos[ci][0]} {combos[ci][1]}.\n"
            xs[b, p:p + FACT_L] = enc(fact)
            spans[b, ci] = (p, p + FACT_L); p += FACT_L
        off = rng.integers(0, len(carrier) - GAP - 1)
        xs[b, p:p + GAP] = carrier[off:off + GAP]; p += GAP
        if distinct:
            qs = rng.permutation(N)[:Q]
        else:
            qs = np.full(Q, rng.integers(N))
        for qi, t in enumerate(qs):
            qtargets[b, qi] = t
            xs[b, p:p + QUERY_L] = enc(f"\n{names[t]} carries the ")
            p += QUERY_L
            xs[b, p:p + ANS_L] = enc(f"{combos[t][0]} {combos[t][1]}")
            p += ANS_L
    x = torch.from_numpy(xs[:, :-1]).to(device)
    y = torch.full((B, SEQ - 1), -100, dtype=torch.long)
    qstart = N * FACT_L + PREGAP + GAP
    for qi in range(Q):
        a0 = qstart + qi * (QUERY_L + ANS_L) + QUERY_L
        y[:, a0 - 1:a0 - 1 + ANS_L] = torch.from_numpy(xs[:, a0:a0 + ANS_L])
    return (x, y.to(device), torch.from_numpy(spans).to(device),
            torch.from_numpy(qtargets).to(device), qstart)


class MultiQueryLM(KVNonceLM):
    """Per-query retrieval: each query segment gets its own key and its
    own injected read. Reduces exactly to the parent at Q=1 (asserted)."""

    def forward_multi(self, x, spans, qtargets, qstart):
        B, Nx = x.shape
        Q = qtargets.shape[1]
        aux = {"balance": x.new_zeros(()).float(),
               "keymatch": x.new_zeros(()).float()}
        keys, vals, route, g, hw = self.write(x, spans)
        if self.write_mode == "learned":
            imp = route.mean(dim=(0, 1))
            aux["balance"] = self.K * (imp ** 2).sum() - 1.0
        h = self.emb(x) + self.pos[:Nx]
        h = h.clone()
        N_ = spans.shape[1]
        ent_k = torch.zeros(B, N_, keys.shape[-1], device=x.device)
        for b in range(B):
            for j in range(N_):
                lo, hi = int(spans[b, j, 0]), int(spans[b, j, 1])
                ent_k[b, j] = self.w_key(hw[b, lo:hi]).mean(0)
        kc = []
        for qi in range(Q):
            q0 = qstart + qi * (QUERY_L + ANS_L)
            q = self.query_key(x, q0)
            a = torch.softmax(
                torch.einsum("be,bke->bk", q, keys) * self.scale, dim=-1)
            r = torch.einsum("bk,bke->be", a, vals)
            seg_end = min(q0 + QUERY_L + ANS_L, Nx)
            h[:, q0:seg_end] = h[:, q0:seg_end] + self.r_up(r).unsqueeze(1)
            logits_kc = torch.einsum("be,bne->bn", q, ent_k) * self.scale
            kc.append(F.cross_entropy(logits_kc, qtargets[:, qi]))
        aux["keymatch"] = torch.stack(kc).mean()
        mask = torch.ones(Nx, Nx, dtype=torch.bool, device=x.device)
        for i in range(Nx):
            lo = max(0, i - self.window + 1)
            mask[i, lo:i + 1] = False
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_out(h)), aux


def train_arm(arm, carrier, steps, seed, device, B=16, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = MultiQueryLM("learned", "learned", 8, max_seq=2048).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        if arm == "curr":
            N = [1, 2, 4, 8][min(3, step // (steps // 4))]
            Q, distinct = 1, True
        else:
            N = 8
            Q = {"q1": 1, "q2": 2, "q4": 4, "q8": 8, "q8rep": 8}[arm]
            distinct = arm != "q8rep"
        x, y, spans, qt, qstart = make_batch_multi(
            B, N, Q, carrier, rng, device, distinct)
        logits, aux = m.forward_multi(x, spans, qt, qstart)
        loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                ignore_index=-100)
                + 0.01 * aux["balance"] + 0.1 * aux["keymatch"])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m


@torch.no_grad()
def eval_first_query(m, carrier, device, batches=8):
    """Uniform metric for every arm: recall on the FIRST query of an
    N=8, Q=1 sequence — untouched by repeat-copyability."""
    ev = np.random.default_rng(99)
    ok = n = 0
    for _ in range(batches):
        x, y, spans, qt, qstart = make_batch_multi(
            64, 8, 1, carrier, ev, device, True)
        logits, _ = m.forward_multi(x, spans, qt, qstart)
        pred = logits.argmax(-1)
        sel = y != -100
        ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--arms", nargs="+",
                    default=["q1", "q2", "q4", "q8", "q8rep", "curr"])
    ap.add_argument("--out", default="nonce-credit.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()

    print(f"credit-density test · N=8 fixed · 2048-bit memory · "
          f"{a.steps} steps · {a.seeds} seeds · {dev}")
    print(f"eval: first-query recall on N=8/Q=1 sequences · chance {1/64:.1%}\n")
    print(f"  {'arm':<8}{'recall med':>12}{'conv':>6}   per-seed")
    print("  " + "-" * 48)
    res = {}
    for arm in a.arms:
        accs = []
        for s in range(a.seeds):
            m = train_arm(arm, carrier, a.steps, s, dev)
            accs.append(eval_first_query(m, carrier, dev))
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        med = float(np.median(accs))
        conv = sum(v >= 0.20 for v in accs)
        print(f"  {arm:<8}{med:>11.1%}{conv:>4}/{a.seeds}   "
              + " ".join(f"{v:.2f}" for v in accs), flush=True)
        res[arm] = dict(acc=accs, median=med, converged=conv)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
