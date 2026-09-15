"""Does K=V tying (Kayyam et al. 2026) break associative RECALL?

The paper reports Q!=K=V (unify key and value projections) costs only 3.1%
LM perplexity for 50% KV-cache savings. A commenter reconstructed the
mechanism: under K=V "the query is a guess as to what the value is, and
the head returns the value closest to the guess."

That framing predicts K=V must FAIL on recall, for a reason LM perplexity
never exposes: recall needs a distinct KEY to address by (the entity name)
and a separate VALUE to fetch (the fact). Query-as-guess-at-the-value
only works when you can guess the content - but in recall the content is
the unknown you are retrieving. Collapse key and value and there is
nothing left to address with.

PRE-REGISTERED (the commenter's mechanism, made falsifiable):
  Independent K/V recalls (our v3.1 result); K=V-tied recall collapses
  toward floor - and the gap is LARGE, not the paper's ~3%, because the
  cost lands on addressing, which LM perplexity does not stress.
  Null (tied ~ independent) -> address and content are not separable jobs
  here after all, and the paper's cheap-tying result extends to recall.

Faithful to the paper: tie the PROJECTIONS (w_val := w_key), so K and V
are the same projection of the writer state - not v:=k as vectors. N=1
(no interference, no credit dilution - the cleanest possible test) and
N=4. Everything else is the v3.1 learned/learned KV memory unchanged.

    python kv_tie.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from nonce_lm import make_batch, load_carrier, V
from nonce_lm2 import KVNonceLM


def run(tie, N, carrier, steps, seed, device, B=16, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = KVNonceLM("learned", "learned", N, max_seq=2048).to(device)
    if tie:
        m.w_val = m.w_key                      # K=V: one projection for both
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y, spans, tgt, _ = make_batch(B, N, carrier, rng, device)
        logits, aux, _ = m(x, spans, tgt)
        loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                ignore_index=-100)
                + 0.01 * aux["balance"] + 0.1 * aux["keymatch"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(99)
    ok = n = 0
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, tgt, _ = make_batch(64, N, carrier, ev, device)
            pred = m(x, spans, tgt)[0].argmax(-1)
            sel = y != -100
            ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="kv-tie.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()

    print(f"K=V tying vs recall · {a.steps} steps · {a.seeds} seeds · {dev}")
    print(f"paper's K=V cost on LM perplexity: ~3.1%. prediction: recall cost >> that\n")
    print(f"  {'projection':<16}{'N':>4}{'recall med':>12}{'conv':>6}   per-seed")
    print("  " + "-" * 52)
    res = {}
    for tie in (False, True):
        for N in a.entities:
            accs = [run(tie, N, carrier, a.steps, s, dev)
                    for s in range(a.seeds)]
            med = float(np.median(accs))
            conv = sum(v >= 0.20 for v in accs)
            lab = "K=V tied" if tie else "K,V independent"
            print(f"  {lab:<16}{N:>4}{med:>11.1%}{conv:>4}/{a.seeds}   "
                  + " ".join(f"{v:.2f}" for v in accs), flush=True)
            res[f"{'tied' if tie else 'indep'}-N{N}"] = dict(
                acc=accs, median=med, converged=conv)
    print("\n  the drop from independent -> tied is the address/content-"
          "separability cost that LM perplexity hides")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
