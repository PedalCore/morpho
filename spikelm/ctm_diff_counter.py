"""A differential counter for synchronisation, and what it buys.

Five uniform-quantisation attempts read as chance. The diagnostic says
why: num/den has a per-pair mean of 0.044 and a per-input spread of 0.038,
so the accumulator idles near the bottom of its range and the informative
variation covers about 4% of it. Uniform quantisation over [0, den] spends
96% of its levels where the signal never goes — at 6 bits the whole signal
fits inside 2.4 levels.

The fix is what an instrument designer would do: null the baseline and
amplify the deviation. Instead of counting coincidences,

    num <- decay*num + prod                       (idles at 0.044*den)

count EXCESS coincidences against the pair's own expected rate,

    num <- decay*num + (prod - rate_ij)           (idles at 0)

which centres the accumulator so the quantiser's levels land where the
signal actually lives. rate_ij is one calibrated constant per pair, and
subtracting a constant each tick is free in hardware — it is a bias on a
signed counter, not an operation.

Calibration uses running statistics during training, frozen for
evaluation, which is the same discipline as batch-norm and the same thing
a calibration pass does on real hardware. The scale likewise: a per-pair
constant, not a per-batch quantity, so the counter the model trains
against is the one it is evaluated with.

PREDICTED, before running: centring gains roughly 2.5-3 bits, so 8-bit
differential should work where 8-bit uniform (9.8 levels) failed, and
should land near the exact-counter result of 100%.
"""
import argparse, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from ctm import tail_mean_loss, Synchronisation
from ctm_parity import ParityCTM, batch

CFG = {"bits": None, "diff": True, "k": 3.0}
STATS = {}

def spike(z):
    h = (z > 0).to(z.dtype)
    return z + (h - z).detach()

def diff_step(self, z, num, den):
    decay = torch.exp(-F.softplus(self.r_raw))
    zz = spike(z)
    prod = zz[:, self.ia] * zz[:, self.ib]
    st = STATS.setdefault(id(self), {
        "rate": torch.zeros(self.n_pairs), "scale": torch.ones(self.n_pairs)})
    if self.training:
        with torch.no_grad():                     # calibration, frozen at eval
            st["rate"].mul_(0.99).add_(0.01 * prod.mean(0))
    if CFG["diff"]:
        prod = prod - st["rate"]
    num = decay * num + prod
    den = decay * den + 1.0
    if self.training:
        with torch.no_grad():
            st["scale"].mul_(0.99).add_(0.01 * num.std(0).clamp_min(1e-6))
    if CFG["bits"] is not None:
        n = 2 ** (CFG["bits"] - 1)
        s = (CFG["k"] * st["scale"]) / max(n - 1, 1)
        q = torch.round(num / s).clamp(-n, n - 1) * s
        num = num + (q - num).detach()
    return num / den.sqrt().clamp_min(1e-6), num, den

def run(seed, L, T, steps, dev, B=128, lr=2e-3):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    STATS.clear()
    m = ParityCTM(L, ticks=T).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    m.train()
    for _ in range(steps):
        x, y = batch(B, L, rng, dev)
        loss, _, _ = tail_mean_loss(m(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
    m.eval()
    ev = np.random.default_rng(12345); ok = tot = 0
    with torch.no_grad():
        for _ in range(12):
            x, y = batch(256, L, ev, dev)
            ok += int((m(x)[:, -1].argmax(-1) == y).sum()); tot += len(y)
    return ok / tot

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()
dev = "cpu"
Synchronisation.step = diff_step

# The control comes FIRST this time. An earlier version put centring in
# every arm including the "exact" one, so its 46.8% could not be attributed
# to centring or to the harness until this control existed.
arms = [("CONTROL: exact, no centring", None, False),
        ("exact, WITH centring", None, True),
        ("8-bit, no centring", 8, False),
        ("8-bit, WITH centring", 8, True)]
print(f"parity of 6 bits · T=8 · {a.steps} steps · {a.seeds} seeds\n")
print(f"  {'counter':<32}{'accuracy':>11}{'spread':>9}")
print("  " + "-" * 52)
res = {}
for label, bits, diff in arms:
    CFG.update(bits=bits, diff=diff)
    accs = [run(s, 6, 8, a.steps, dev) for s in range(a.seeds)]
    print(f"  {label:<32}{np.mean(accs):>10.1%}{np.std(accs):>9.3f}")
    res[label] = accs
json.dump(res, open("ctm-diff-counter.json", "w"), indent=1)
print("\n  wrote ctm-diff-counter.json")
