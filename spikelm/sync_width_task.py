"""Is synchronisation's width USABLE, or only present?

We measured that P sampled pairs span 3-9x more directions than the D
neurons they come from. That is capacity, not use. On parity a trained
model showed ~1.7 usable directions no matter how many pairs it had,
because parity's output is one bit and nothing pressured it to be wide.

So: a task that cannot be done narrowly. Given L bits, produce K outputs,
each the XOR of a DIFFERENT fixed pair of input bits. Every output is an
independent function of the input, so solving all K genuinely requires K
directions in the representation. Turning K up turns the width requirement
up, directly — and because each individual output is a 2-bit XOR rather
than a wide parity, the task stays learnable, so the bottleneck is width
rather than computation. A K-wide version of hard parity would sit at
chance in every arm and measure nothing.

THE COMPARISON. The readout is the only thing that varies:

  sync   D=32   K outputs read from P sampled PAIRS of 32 neurons
  act    D=32   K outputs read from the 32 activations themselves
  act    D=128  the same, from 128 activations - a bigger state, as a
                reference for what the sync arm is trying to match

Everything else - the tick loop, the synapse, the neuron-level models, and
the query path, which stays sync-driven in all arms - is identical. Only
where the OUTPUT is read from changes.

PRE-REGISTERED, before running:
  1. At K=1 all three arms are similar; width is not needed.
  2. As K rises, act/D=32 saturates first - 32 activations cannot carry
     many more than 32 independent outputs.
  3. THE TEST: sync/D=32 keeps up past that point, tracking act/D=128.
     If instead sync/D=32 saturates alongside act/D=32, the extra
     directions are real but unusable, and the case for a narrow state
     driving a wide representation fails.

Readout parameter counts are reported, because the sync arm has more of
them (P x K against D x K) and that has to be visible when reading the
result. The claim being tested is about the size of the STATE, which is
what the O(D^2) synapse scales with - not about total parameters.

    python sync_width_task.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctm import CTM


def make_task(L, K, subset=2, seed=0):
    """K outputs, each the XOR of a different fixed PAIR of input bits.

    Deliberately easy per output. Parity over large subsets is hard to
    LEARN, so a K-wide version of it would sit at chance for every arm and
    measure computation rather than width. A 2-bit XOR is trivial; having
    K independent ones is not. That puts the only bottleneck on how many
    directions the representation can carry, which is the question.
    """
    rng = np.random.default_rng(seed)
    masks = np.zeros((K, L), dtype=np.int64)
    seen = set()
    for k in range(K):
        while True:
            idx = tuple(sorted(rng.choice(L, subset, replace=False)))
            if idx not in seen or len(seen) >= L * (L - 1) // 2:
                break
        seen.add(idx)
        masks[k, list(idx)] = 1
    return torch.from_numpy(masks)


def batch(B, L, masks, rng, device):
    bits = torch.from_numpy(rng.integers(0, 2, size=(B, L))).long()
    y = (bits.float() @ masks.t().float()) % 2           # (B, K)
    return bits.to(device), y.to(device)


class WidthModel(nn.Module):
    """One CTM, with the output read either from sync or from activations."""

    def __init__(self, L, K, D, readout, pairs=512, d=64, ticks=8, M=16):
        super().__init__()
        self.emb = nn.Embedding(2, d)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.readout = readout
        self.ctm = CTM(d_input=d, n_out=K, D=D, M=M, ticks=ticks,
                       pairs_out=pairs, pairs_action=pairs, d_head=64)
        if readout == "act":
            # read the K outputs straight off the activations instead
            self.ctm.w_out = nn.Linear(D, K, bias=False)

    def forward(self, bits):
        kv = self.emb(bits) + self.pos
        ctm = self.ctm
        B = kv.shape[0]
        z = ctm.z0.expand(B, ctm.D).contiguous()
        A = ctm.a0.t().expand(B, ctm.D, ctm.M).contiguous()
        no, do = ctm.sync_out.reset(B, kv.device, kv.dtype)
        na, da = ctm.sync_act.reset(B, kv.device, kv.dtype)
        o = torch.zeros(B, ctm.attn.q.in_features, device=kv.device,
                        dtype=kv.dtype)
        ys = []
        for _ in range(ctm.ticks):
            a = ctm.synapse(torch.cat([z, o], dim=-1))
            A = torch.cat([A[:, :, 1:], a.unsqueeze(-1)], dim=-1)
            z = ctm.ln(ctm.nlm(A))
            s_out, no, do = ctm.sync_out.step(z, no, do)
            s_act, na, da = ctm.sync_act.step(z, na, da)
            o = ctm.attn(ctm.w_in(s_act), kv)            # query path: always sync
            ys.append(ctm.w_out(s_out if self.readout == "sync" else z))
        return torch.stack(ys, 1)                        # (B, T, K) logits

    def readout_params(self):
        return self.ctm.w_out.weight.numel()


def tail_bce(logits, y, frac=0.5):
    """Mean over the final half of ticks, as established for tick ablations."""
    B, T, K = logits.shape
    start = max(0, T - max(1, int(round(T * frac))))
    tail = logits[:, start:]
    tgt = y.unsqueeze(1).expand(B, tail.shape[1], K)
    return F.binary_cross_entropy_with_logits(tail, tgt)


def run(L, K, D, readout, steps, device, seed=0, pairs=512, B=128, lr=2e-3,
        subset=2):
    torch.manual_seed(seed)
    masks = make_task(L, K, subset=subset, seed=1234)    # task fixed across arms
    rng = np.random.default_rng(seed)
    m = WidthModel(L, K, D, readout, pairs=pairs).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y = batch(B, L, masks, rng, device)
        loss = tail_bce(m(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(4321)
    ok = tot = 0
    with torch.no_grad():
        for _ in range(8):
            x, y = batch(256, L, masks, ev, device)
            pred = (m(x)[:, -1] > 0).float()
            ok += float((pred == y).float().sum()); tot += y.numel()
    return ok / tot, m.readout_params()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--subset", type=int, default=2)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=512)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of sync32/act32/act128/act512")
    ap.add_argument("--out", default="sync-width-task.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    all_arms = [("sync32", "sync  D=32", 32, "sync"),
                ("act32", "act   D=32", 32, "act"),
                ("act128", "act   D=128", 128, "act"),
                # readout-parameter-matched control: sync/D=32 reads K outputs
                # off 512 pairs; 512 activations match that readout width with
                # a 16x larger state. If act/512 matches sync/32, the width
                # result was about readout parameters, not about pairs.
                ("act512", "act   D=512", 512, "act")]
    arms = [x[1:] for x in all_arms if x[0] in (a.arms or ["sync32", "act32", "act128"])]

    print(f"K x {a.subset}-bit XOR · L={a.length} bits · {a.steps} steps · {a.seeds} seeds "
          f"· P={a.pairs} · {dev}")
    print("  per-bit accuracy; chance is 50%\n")
    hdr = f"  {'readout':<14}" + "".join(f"{('K='+str(k)):>10}" for k in a.ks)
    print(hdr + f"{'readout params':>16}")
    print("  " + "-" * (len(hdr) + 14))
    res = {}
    for label, D, ro in arms:
        row, params = [], None
        print(f"  {label:<14}", end="", flush=True)
        for K in a.ks:
            out = [run(a.length, K, D, ro, a.steps, dev, seed=s,
                       pairs=a.pairs, subset=a.subset) for s in range(a.seeds)]
            acc = float(np.mean([o[0] for o in out]))
            params = out[0][1]
            row.append(acc)
            print(f"{acc:>9.1%}", end="", flush=True)
        print(f"{params:>16,}")
        res[label] = {"acc": row, "ks": a.ks, "readout_params": params}

    print("\n  does sync width get used?")
    for K, i in zip(a.ks, range(len(a.ks))):
        s32 = res["sync  D=32"]["acc"][i]
        a32 = res["act   D=32"]["acc"][i]
        a128 = res["act   D=128"]["acc"][i]
        verdict = ("sync beats its own state size" if s32 > a32 + 0.02
                   else "no gain over plain activations")
        print(f"    K={K:<4} sync/32 {s32:.1%}  act/32 {a32:.1%}  "
              f"act/128 {a128:.1%}   {verdict}")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
