"""Does the internal tick loop actually do anything? Parity says yes or no.

Before wiring a CTM into a language model we need to know our
implementation works, and language is a terrible place to find that out -
everything there is dominated by scale and precision, and a subtly broken
model just looks like a slightly worse model.

Parity is the right test. The parity of L bits cannot be computed by a
single attention lookup; it needs accumulation over the sequence, and it
is the task the CTM paper uses to demonstrate sequential internal
computation. If the tick loop works, more ticks should mean more accuracy,
because the extra ticks are the only thing that changes.

THE ABLATION THAT MATTERS. Identical model, identical parameter count in
everything except the loop, only `ticks` varying. T=1 removes the loop
entirely: one query, one look at the data, one prediction. If T=8 does not
beat T=1, the loop is not doing the work and neither our implementation
nor the idea survives.

PRE-REGISTERED, before the first run:
  1. T=1 is at or near chance (50%). One glance cannot count bits.
  2. Accuracy rises monotonically with T.
  3. T>=8 is well above chance.
  Failure of 2 or 3 stops the language work.

    python ctm_parity.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from ctm import CTM, ctm_loss, mean_tick_loss, last_tick_loss, tail_mean_loss


def batch(B, L, rng, device):
    bits = torch.from_numpy(rng.integers(0, 2, size=(B, L))).long()
    return bits.to(device), (bits.sum(1) % 2).to(device)


class ParityCTM(nn.Module):
    """Embed the bits, let the CTM look at them however it likes."""

    def __init__(self, L, d=64, ticks=8, D=128, M=16):
        super().__init__()
        self.emb = nn.Embedding(2, d)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.ctm = CTM(d_input=d, n_out=2, D=D, M=M, ticks=ticks,
                       pairs_out=256, pairs_action=256, d_head=64)

    def forward(self, bits):
        return self.ctm(self.emb(bits) + self.pos)


def run(ticks, L, steps, device, seed=0, d=64, D=128, lr=2e-3, B=128,
        loss_fn=mean_tick_loss):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ParityCTM(L, d=d, ticks=ticks, D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for _ in range(steps):
        x, y = batch(B, L, rng, device)
        logits = model(x)
        loss, _, _ = loss_fn(logits, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    # evaluate on the FINAL tick, matching what last_tick_loss trains
    model.eval()
    ev = np.random.default_rng(12345)
    correct = tot = 0
    per_tick = torch.zeros(ticks)
    with torch.no_grad():
        for _ in range(20):
            x, y = batch(256, L, ev, device)
            lg = model(x)                                  # (B, T, 2)
            p = lg.softmax(-1)
            ent = -(p * p.clamp_min(1e-9).log()).sum(-1)
            del ent, p
            correct += int((lg[:, -1].argmax(-1) == y).sum())   # final tick
            tot += len(y)
            per_tick += (lg.argmax(-1) == y.unsqueeze(1)).float().sum(0).cpu()
    return correct / tot, (per_tick / tot).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--ticks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--loss", choices=["tail", "last", "mean", "two_tick"], default="tail")
    ap.add_argument("--out", default="ctm-parity.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = {"tail": tail_mean_loss, "last": last_tick_loss,
               "mean": mean_tick_loss, "two_tick": ctm_loss}[a.loss]

    print(f"parity of {a.length} bits · {a.steps} steps · {a.seeds} seeds "
          f"· {dev} · {a.loss} loss\n")
    print(f"  {'ticks':>6}{'accuracy':>12}{'spread':>10}   per-tick accuracy")
    print("  " + "-" * 62)
    res = {"length": a.length, "steps": a.steps, "arms": {}}
    for T in a.ticks:
        accs, pt = [], None
        for s in range(a.seeds):
            acc, per_tick = run(T, a.length, a.steps, dev, seed=s, loss_fn=loss_fn)
            accs.append(acc)
            pt = per_tick
        m, sd = float(np.mean(accs)), float(np.std(accs))
        shown = " ".join(f"{v:.2f}" for v in pt[:8])
        print(f"  {T:>6}{m:>11.1%}{sd:>10.3f}   {shown}")
        res["arms"][str(T)] = {"acc": accs, "mean": m, "std": sd,
                               "per_tick": pt}

    best = max(res["arms"].items(), key=lambda kv: kv[1]["mean"])
    one = res["arms"].get("1", {}).get("mean")
    print(f"\n  T=1 (loop removed): {one:.1%}" if one else "")
    print(f"  best: T={best[0]} at {best[1]['mean']:.1%}")
    if one is not None:
        print(f"  the loop is worth {best[1]['mean'] - one:+.1%}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
