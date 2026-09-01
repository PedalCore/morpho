"""Intrinsic width of a synchronisation readout, independent of any task.

sync_rank.py measured a TRAINED model and found ~1-2 usable directions.
That is not a fact about synchronisation; it is a fact about parity, whose
output is one bit, so nothing pressures the representation to be wide. To
ask what the readout CAN carry, feed it generic activations and count.

The theory says the pessimistic bound is the wrong one. rank(S) <= min(D,t)
constrains S for a SINGLE input, but the feature vector across inputs is
different: distinct pairs (i,j) are distinct monomials z_i z_j, and
distinct functions stay independent across samples. So P pairs should give
close to P directions up to D(D+1)/2, and per-pair decay should add more by
reading the same pair through different temporal filters.

If that holds, a narrow state really can drive a wide representation and
the O(D^2) synapse shrinks. If P directions collapse to ~D, it cannot.
"""
import json
import numpy as np
from sync_rank import effective_rank, sync_features

def measure(D, T, P, n, decay_mode, rng):
    Z = rng.standard_normal((n, T, D))
    ia, ib = rng.integers(0, D, P), rng.integers(0, D, P)
    r = {"learned-like": rng.uniform(0.3, 1.0, P),
         "uniform": np.full(P, 0.6),
         "none": np.zeros(P)}[decay_mode]
    return effective_rank(sync_features(Z, ia, ib, r))

rng = np.random.default_rng(0)
T, n = 8, 4096
print(f"generic activations, T={T} ticks, {n} samples\n")
print(f"  {'D':>4}{'pairs':>7}{'decays':>14}{'participation':>15}{'n95':>7}"
      f"{'vs D':>8}{'vs pairs':>10}")
print("  " + "-" * 66)
res = {}
for D in (16, 32, 64):
    base = effective_rank(rng.standard_normal((n, D)))
    for P in (64, 256, 1024):
        if P > D * (D + 1) // 2:
            continue
        for mode in ("learned-like", "uniform", "none"):
            er = measure(D, T, P, n, mode, rng)
            pr = er["participation"]
            print(f"  {D:>4}{P:>7}{mode:>14}{pr:>15.1f}{er['n95']:>7}"
                  f"{pr/D:>8.1f}x{pr/P:>9.2f}x")
            res[f"D{D}-P{P}-{mode}"] = er
    print()
print("  vs D  > 1 means the readout is WIDER than its neuron count")
print("  vs pairs ~ 1 means the pairs are close to fully independent")
json.dump(res, open("sync-capacity.json", "w"), indent=1)
print("\n  wrote sync-capacity.json")
