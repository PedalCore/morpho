"""Can the synchronisation representation be read from SPIKES?

Synchronisation is the one part of the CTM that gets CHEAPER when
amplitude is removed, which is the reverse of everything else we have
measured. In the recursive form

    num_ij <- exp(-r_ij) num_ij + z_i z_j
    den_ij <- exp(-r_ij) den_ij + 1
    S_ij    = num_ij / sqrt(den_ij)

the only arithmetic on the data is the product z_i z_j. For real-valued z
that is a multiply, which we measured at 305 gates. For binary z it is an
AND — one gate. And den carries no data at all: it converges to
1/(1 - exp(-r)), so it is a per-pair constant that can be precomputed and
folded into the readout.

So a spiking synchronisation unit is one AND gate, one leaky counter, and
a fixed scale. That is the whole thing.

Two questions this measures, both of which have to hold for that to be
worth anything:

  1. Does the representation survive being read from spikes at all?
     Post-activations are thresholded to {0,1} before entering sync,
     everything else unchanged.
  2. How wide does the counter have to be? Our capacity experiment found
     that sharp selection tolerates coarse WEIGHTS but not a coarse COUNT
     - it needed about 6 bits - while the bit-budget audit found recurrent
     state to be the most precision-hungry tensor in a language model at
     12. The sync accumulator is both a count and a recurrent state, so it
     is not obvious which of those it behaves like. That is the open
     question, and it decides whether this is buildable.

Measured on the parity task where the tick loop demonstrably works.

    python ctm_spiking_sync.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctm import tail_mean_loss
from ctm_parity import ParityCTM, batch

CFG = {"spike": False, "bits": None}


def spike(z):
    """Threshold to {0,1} with a straight-through estimator.

    Surrogate gradient rather than the real one: the forward pass is a hard
    threshold, the backward pass pretends it was the identity, which is the
    standard trick and the same one spikelm/spiking.py uses.
    """
    hard = (z > 0).to(z.dtype)
    return z + (hard - z).detach()


def quantise(x, bits, den):
    """The accumulator as a real counter: fixed width, correct full scale.

    THREE earlier versions of this measured the quantiser rather than the
    model, and all three read as chance:

      per-tensor scale     one scale across all 256 pairs flushes the quiet
                           pairs to zero (6 bits -> 51.5%)
      per-batch max        scale moves with batch composition, so the
                           counter trained against is not the one evaluated
                           (8 bits -> 51.6%)
      asymptotic bound     1/(1-decay) is the limit over INFINITE ticks; at
                           T=8 the real maximum is ~100x smaller, so the LSB
                           was ~100x too coarse (8 bits -> 51.4%)

    The correct full scale was already being computed: den. Since prod is
    in {0,1} for binary spikes, num <- decay*num + prod is bounded term by
    term by den <- decay*den + 1. So num/den is in [0,1] by construction and
    den is exactly the counter's full scale at every tick.
    """
    if bits is None:
        return x
    # den is EXACTLY the largest num can be: prod is in {0,1} for binary
    # spikes, so num <- decay*num + prod is bounded above by
    # den <- decay*den + 1 term by term. Using the asymptotic bound
    # 1/(1-decay) instead overestimates the range by ~100x at T=8 and makes
    # the LSB far too coarse — that was our 8-bit "51.4%", not the model.
    s = den / (2 ** bits - 1)
    q = torch.round(x / s).clamp(0, 2 ** bits - 1) * s   # unsigned: num >= 0
    return x + (q - x).detach()


def patched_step(self, z, num, den):
    """Synchronisation.step, with the two hardware questions inserted."""
    decay = torch.exp(-F.softplus(self.r_raw))
    zz = spike(z) if CFG["spike"] else z
    prod = zz[:, self.ia] * zz[:, self.ib]        # an AND when zz is binary
    den_next = decay * den + 1.0
    num = quantise(decay * num + prod, CFG["bits"], den_next)
    return num / den_next.sqrt().clamp_min(1e-6), num, den_next


def run(seed, L, T, steps, device, B=128, lr=2e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = ParityCTM(L, ticks=T).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y = batch(B, L, rng, device)
        loss, _, _ = tail_mean_loss(m(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(12345)
    ok = tot = 0
    with torch.no_grad():
        for _ in range(12):
            x, y = batch(256, L, ev, device)
            ok += int((m(x)[:, -1].argmax(-1) == y).sum()); tot += len(y)
    return ok / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="ctm-spiking-sync.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from ctm import Synchronisation
    Synchronisation.step = patched_step

    arms = [("real-valued sync (reference)", False, None),
            ("SPIKING sync, exact counter", True, None),
            ("SPIKING sync, 8-bit counter", True, 8),
            ("SPIKING sync, 6-bit counter", True, 6),
            ("SPIKING sync, 4-bit counter", True, 4),
            ("SPIKING sync, 3-bit counter", True, 3),
            ("SPIKING sync, 2-bit counter", True, 2)]

    print(f"parity of {a.length} bits · T={a.ticks} · {a.steps} steps · "
          f"{a.seeds} seeds · {dev}\n")
    print(f"  {'sync input':<32}{'accuracy':>11}{'spread':>9}{'gates/pair':>12}")
    print("  " + "-" * 66)
    res = {}
    for label, sp, bits in arms:
        CFG.update(spike=sp, bits=bits)
        accs = [run(s, a.length, a.ticks, a.steps, dev) for s in range(a.seeds)]
        m, sd = float(np.mean(accs)), float(np.std(accs))
        cost = "~305" if not sp else "1 + counter"
        print(f"  {label:<32}{m:>10.1%}{sd:>9.3f}{cost:>12}")
        res[label] = {"acc": accs, "mean": m, "std": sd}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
