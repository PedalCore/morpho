"""Does interaction ORDER buy associative capacity without amplitude?

This is the load-bearing assumption behind building a spiking architecture
around synchronisation, tested on its own before anything is built on it.

The argument being checked. Classical Hopfield stores ~0.14N patterns and
fails because its energy is quadratic: the target's overlap and the
crosstalk from other patterns grow at similar rates. Krotov and Hopfield
showed that a rapidly growing energy F(overlap) crushes the crosstalk and
lifts capacity to ~N^(n-1) for F(x)=x^n, exponential for F=exp. Their
result needs no continuous-valued NEURONS - the capacity comes from the
order of the interaction, not from amplitude.

If that is right, an amplitude-free machine should be able to buy the same
capacity, because the overlap it needs is a COUNT (spikes coinciding) and
the nonlinearity it needs can be a THRESHOLD (a neuron firing). Neither
requires a high-precision multiply. That is the whole appeal for spiking
hardware, where a k-fold coincidence is one gate and a k-th order weight
tensor is combinatorially impossible.

So the arms are:

  poly2         classical Hopfield, the quadratic baseline
  poly3, poly5  Krotov's higher-order energies, amplitude-rich
  exp           modern Hopfield / attention's separation, amplitude-rich
  topk          AMPLITUDE-FREE: rank the overlaps, let the top k contribute
                equally. Comparison and counting only - no multiply, no
                exponential. This is what lateral inhibition does.
  step          AMPLITUDE-FREE: every pattern whose overlap clears a
                threshold contributes equally. This is what a neuron does.

PRE-REGISTERED PREDICTIONS, written before the first run:

  1. poly2 recovers the textbook ~0.14N and is worst.
  2. poly3/poly5/exp are far above it, rising with order.
  3. THE TEST: topk and step land with the high-order arms, not with
     poly2. If they land with poly2, amplitude was doing the work and the
     spiking-synchronisation story is dead as stated.
  4. Quantising the overlap COUNT to a few bits costs little, because the
     count only has to order the patterns correctly - the same reason the
     receptance gate survived 3 bits in the bit-budget audit.

Failure of 3 is the outcome that matters most and would stop the
architecture work, so it is stated first and plainly.

    python coincidence_capacity.py
"""

import argparse
import json

import numpy as np


# ------------------------------------------------------------ energy shapes

def pattern_weights(o, arm, param, bits=None):
    """How much each stored pattern gets to vote, given its match count.

    o: (P,) integer overlaps in [-N, N]. This single function is the ONLY
    difference between the arms — the retrieval loop below is shared, so
    nothing but the shape of the weighting varies across the comparison.
    """
    if bits is not None:                     # quantise the COUNT itself
        m = np.abs(o).max()
        if m > 0:
            n = 2 ** (bits - 1)
            s = m / max(n - 1, 1)
            o = np.round(o / s).clip(-n, n - 1) * s

    if arm == "linear":                      # classical Hopfield: w = overlap
        return o
    if arm == "poly":                        # Krotov: F(x)=relu(x)^n
        return np.maximum(o, 0.0) ** (param - 1)
    if arm == "exp":                         # modern Hopfield / softmax
        z = param * o
        return np.exp(z - z.max())
    if arm == "topk":                        # AMPLITUDE-FREE: rank only
        k = int(param)
        if k >= o.size:
            return np.ones_like(o, dtype=np.float64)
        cut = np.partition(o, -k)[-k]
        return (o >= cut).astype(np.float64)
    if arm == "step":                        # AMPLITUDE-FREE: threshold only
        return (o >= param * o.size ** 0 * 1.0).astype(np.float64)
    raise ValueError(arm)


def retrieve(X, probe, arm, param, bits=None, sweeps=4):
    """One associative recall, in the modern-Hopfield/attention form:

        s <- sign( sum_mu  w(<xi^mu, s>) * xi^mu )

    i.e. score every stored pattern against the probe, then rebuild the
    state as a weighted superposition of the patterns. Classical Hopfield
    is the case w(x)=x; attention is w=softmax. The amplitude-free arms
    use a rank or a threshold, which need comparison and counting only.
    """
    s = probe.copy()
    for _ in range(sweeps):
        w = pattern_weights(X @ s, arm, param, bits)
        new = np.sign(w @ X)
        new[new == 0] = s[new == 0]
        if np.array_equal(new, s):
            break
        s = new
    return s


# ------------------------------------------------------------- the sweep

def capacity(N, arm, param, bits=None, flip=0.1, trials=24, seed=0,
             p_grid=None, success=0.95):
    """Largest P at which corrupted probes are recovered exactly."""
    rng = np.random.default_rng(seed)
    best = 0
    for P in p_grid:
        ok = 0
        for t in range(trials):
            X = rng.choice([-1.0, 1.0], size=(P, N))
            mu = rng.integers(P)
            probe = X[mu].copy()
            idx = rng.choice(N, size=max(1, int(flip * N)), replace=False)
            probe[idx] *= -1
            out = retrieve(X, probe, arm, param, bits)
            ok += int(np.array_equal(out, X[mu]))
        if ok / trials >= success:
            best = P
        else:
            break                      # capacity curves are monotone; stop early
    return best


ARMS = [
    ("linear (classical Hopfield)", "linear", 0),
    ("poly3  (Krotov order 3)", "poly", 3),
    ("poly5  (Krotov order 5)", "poly", 5),
    ("poly9  (Krotov order 9)", "poly", 9),
    ("exp    (modern Hopfield)", "exp", 1.0),
    ("topk1  (AMPLITUDE-FREE, WTA)", "topk", 1),
    ("topk3  (AMPLITUDE-FREE, k=3)", "topk", 3),
]

# Corruption is the axis that actually separates these. At 10% of bits
# flipped every sharply-selecting rule finds the right pattern trivially,
# so capacity saturates and the arms look identical. The interesting
# question is how far the probe can be pushed before selection breaks.
FLIPS = [0.1, 0.2, 0.3, 0.4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--flip", type=float, default=0.1)
    ap.add_argument("--out", default="coincidence-capacity.json")
    a = ap.parse_args()

    print("Does interaction order buy capacity without amplitude?\n")
    print(f"  {a.trials} trials, exact recovery required in >=95%")
    CEIL = 64
    grid = lambda N: sorted(set(
        [max(1, int(N * f)) for f in (0.02, 0.05, 0.1, 0.2, 0.5)] +
        [N * m for m in (1, 2, 4, 8, 16, 32, CEIL)]))

    N = a.sizes[-1]
    res = {"sizes": a.sizes, "flips": FLIPS, "ceiling": CEIL, "by_flip": {}}
    print(f"\n  capacity at N={N}, as the probe gets more corrupted "
          f"(ceiling {CEIL}N):\n")
    hdr = f"  {'rule':<30}" + "".join(f"{int(f*100):>9}%" for f in FLIPS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for label, arm, param in ARMS:
        print(f"  {label:<30}", end="", flush=True)
        row = []
        for f in FLIPS:
            c = capacity(N, arm, param, flip=f, trials=a.trials, p_grid=grid(N))
            row.append(c)
            print(f"{c:>10}", end="", flush=True)
        print("")
        res["by_flip"][label] = row

    # does quantising the COUNT hurt? topk1 is the amplitude-free arm that
    # works, so it is the one worth pushing
    print(f"\n  quantising the overlap count itself (topk1, N={N}, 30% flipped):")
    print(f"    {'bits':>6}{'capacity':>10}")
    res["count_bits"] = {}
    for b in (1, 2, 3, 4, 6, None):
        c = capacity(N, "topk", 1, bits=b, flip=0.3, trials=a.trials,
                     p_grid=grid(N))
        res["count_bits"][str(b)] = c
        print(f"    {str(b) if b else 'exact':>6}{c:>10}")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
