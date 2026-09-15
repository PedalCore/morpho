"""Can the fast-weight store hold N associations AT ALL? (isolated, no LM)

Review verdict on the fw floors: they cannot be read as credit-density
evidence until we know the store even retains N facts. Two bugs remained
in the "fixed" memory: (1) retention decays the whole matrix at EVERY
step, so earlier facts fade through later ones (first-fact survival
~1e-77 in an init probe - the store may hold mostly the LAST fact); (2)
separately squashing M and z breaks the linear-attention normaliser
(repeated writes of 0.2 read as 0.26 -> 0.92 -> 1.0).

This strips the memory OUT of the language model and measures it directly,
so no learning/addressing confound is in the way. N random (key, value)
pairs written into a d_k x d_v fast-weight store, then each key queried;
we measure how faithfully each value comes back, by fact POSITION.

Write rules compared (reviewer step B):
  additive   M <- a*M + (v (x) k)            our current rule
  delta      M <- a*M + b*(v - (a*M) k) k^T  gated delta-rule, |k|=1:
                                             writes the ERROR, so an
                                             existing association is not
                                             clobbered by a new one

Read (proper linear-attention normaliser, one consistent map): with unit
keys and the delta rule, M k = v exactly at write time; for additive we
normalise by z = sum k so read = (M q)/(z.q). No separate tanh on M,z.

Key geometry swept: orthogonal (N <= d_k, no interference - the mechanism
ceiling) and random unit (overlap grows with N - the realistic case).
Retention a swept: 1.0 (no decay - isolates capacity from forgetting) and
0.99 (mild decay - shows the whole-matrix-decay bug's cost by position).

Diagnostics:
  * per-position recall: cosine(read(k_i), v_i), averaged over trials,
    reported for the FIRST and LAST written fact separately - if only the
    last survives, first<<last;
  * causal check: change one fact's value, confirm its read moves and the
    others do not (isolates addressing from crosstalk).

    python mem_probe.py
"""

import argparse
import json

import numpy as np


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def make_keys(N, dk, mode, rng):
    if mode == "orthogonal":
        A = rng.standard_normal((max(N, dk), dk))
        Q, _ = np.linalg.qr(A)                    # orthonormal rows
        return Q[:N]
    return unit(rng.standard_normal((N, dk)))     # random unit, overlapping


def write(keys, vals, rule, a, beta=1.0):
    """Sequentially write N pairs. Returns M (dv,dk) and z (dk)."""
    dv, dk = vals.shape[1], keys.shape[1]
    M = np.zeros((dv, dk)); z = np.zeros(dk)
    for k, v in zip(keys, vals):
        if rule == "additive":
            M = a * M + np.outer(v, k)
            z = a * z + k
        else:                                     # gated delta-rule
            Md = a * M
            M = Md + beta * np.outer(v - Md @ k, k)
            z = a * z + k
    return M, z


def read(M, z, q, rule):
    if rule == "additive":
        den = z @ q
        den = den if abs(den) > 1e-6 else (1e-6 if den >= 0 else -1e-6)
        return (M @ q) / den
    return M @ q                                  # delta w/ unit keys: M k = v


def cosine(a, b):
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def probe(N, dk, dv, rule, a, key_mode, trials, rng):
    first, last, others_stable, target_moves = [], [], [], []
    for _ in range(trials):
        keys = make_keys(N, dk, key_mode, rng)
        vals = unit(rng.standard_normal((N, dv)))
        M, z = write(keys, vals, rule, a)
        recalls = [cosine(read(M, z, keys[i], rule), vals[i]) for i in range(N)]
        first.append(recalls[0]); last.append(recalls[-1])
        # causal: change fact j's value, rewrite, check its read moves and
        # a bystander's does not
        if N >= 2:
            j, b = 0, N - 1
            v2 = vals.copy(); v2[j] = unit(rng.standard_normal((1, dv)))[0]
            M2, z2 = write(keys, v2, rule, a)
            target_moves.append(
                cosine(read(M2, z2, keys[j], rule), v2[j]))     # new value recalled?
            others_stable.append(
                cosine(read(M2, z2, keys[b], rule), vals[b]))   # bystander intact?
    return dict(first=float(np.mean(first)), last=float(np.mean(last)),
                target_moves=float(np.mean(target_moves)) if target_moves else None,
                bystander=float(np.mean(others_stable)) if others_stable else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk", type=int, default=16)
    ap.add_argument("--dv", type=int, default=16)
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="mem-probe.json")
    a_ = ap.parse_args()
    rng = np.random.default_rng(0)

    print(f"isolated fast-weight store · d_k={a_.dk} d_v={a_.dv} · "
          f"{a_.trials} trials · recall = cosine(read, true value)\n")
    res = {}
    for key_mode in ("orthogonal", "random"):
        for a in (1.0, 0.99):
            print(f"=== keys: {key_mode} · retention a={a} ===")
            print(f"  {'rule':<10}{'N':>4}{'recall[first]':>15}"
                  f"{'recall[last]':>14}{'newval':>9}{'bystander':>11}")
            for rule in ("additive", "delta"):
                for N in a_.ns:
                    if key_mode == "orthogonal" and N > a_.dk:
                        continue
                    r = probe(N, a_.dk, a_.dv, rule, a, key_mode, a_.trials, rng)
                    nv = f"{r['target_moves']:.2f}" if r['target_moves'] is not None else "  -"
                    bs = f"{r['bystander']:.2f}" if r['bystander'] is not None else "  -"
                    print(f"  {rule:<10}{N:>4}{r['first']:>15.2f}"
                          f"{r['last']:>14.2f}{nv:>9}{bs:>11}")
                    res[f"{key_mode}-a{a}-{rule}-N{N}"] = r
            print()
    json.dump(res, open(a_.out, "w"), indent=1)
    print(f"wrote {a_.out}")


if __name__ == "__main__":
    main()
