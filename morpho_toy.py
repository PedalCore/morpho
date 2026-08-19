"""A trained language model, compiled to gates.

Takes the 84-parameter recurrence trained by `spikelm/train_toy.py` on
delayed recall and builds its ENTIRE forward pass as a MorphoHDL circuit:
embedding lookup, the diagonal recurrence held in registers, and the linear
readout. Then checks the netlist against the model it came from, token by
token, and reports what it costs in gates.

    python morpho_toy.py

The model solves its task exactly (1.000 accuracy in Q6.6 fixed point), so
"the circuit is correct" is a checkable statement rather than a judgement
call — which is why this task was chosen over character prediction, where a
model this small produces only plausible-looking noise.
"""

import json
import os
import time

import numpy as np

from tiny_morpho import morpho, Not, Xor, CAT, ZERO, ONE, LSLICE, VOID, brent_kung_adder
from tiny_morpho_seq import REG, DRIVE, compile_seq
from morpho_lm import qmul_s, cond_neg

W = json.load(open(os.path.join(os.path.dirname(__file__),
                                "spikelm", "toy-export", "weights.json")))
BITS, FRAC = W["bits"], W["frac"]
D = len(W["a"])
V = len(W["emb"])


def const_bus(value, bits=BITS):
    """A literal constant as a Morpho bus — wires tied high or low."""
    v = int(value) & ((1 << bits) - 1)
    return np.array([(v >> i) & 1 for i in range(bits)], dtype=np.int32)


def const_lane(value, bits=BITS):
    return const_bus(value, bits)[:, None]


# ------------------------------------------------------------------ cells

@morpho
def select4(sel, c0, c1, c2, c3):
    """4-way select on a 2-bit index, built from AND/OR over the sel bits.

    The embedding table is only four entries wide, so the lookup is this
    rather than a memory: every entry is a constant, and the mux is pure
    logic that the compiler folds hard.
    """
    s0, s1 = sel[0:1], sel[1:2]
    n0, n1 = Not(s0), Not(s1)
    def pick(c, a, b):
        from tiny_morpho import And
        return And(c, And(a, b))
    from tiny_morpho import Or, And
    out = Or(Or(And(c0, And(n0, n1)), And(c1, And(s0, n1))),
             Or(And(c2, And(n0, s1)), And(c3, And(s0, s1))))
    return out


def make_toy():
    """The whole trained model as one sequential circuit."""
    a_c = [const_bus(v) for v in W["a"]]
    b_c = [const_bus(v) for v in W["b"]]
    emb = [[const_bus(W["emb"][t][c]) for t in range(V)] for c in range(D)]
    wout = [[const_bus(W["w_out"][o][c]) for c in range(D)] for o in range(V)]
    bout = [const_bus(W["b_out"][o]) for o in range(V)]

    @morpho
    def toy(token):                       # token: 2 bits/step -> logits: V*BITS
        logits = []
        hs = []
        for c in range(D):
            u = select4(token, emb[c][0], emb[c][1], emb[c][2], emb[c][3])
            h = REG(np.zeros(BITS, np.int32))
            drive = qmul_s(a_c[c], h, FRAC)
            drive2 = qmul_s(b_c[c], u, FRAC)
            nxt, _ = brent_kung_adder(drive, drive2, ZERO)
            DRIVE(h, nxt)
            hs.append(h)
        for o in range(V):
            acc = bout[o]
            for c in range(D):
                p = qmul_s(wout[o][c], hs[c], FRAC)
                acc, _ = brent_kung_adder(acc, p, ZERO)
            logits.append(acc)
        return CAT(*logits)
    return toy


# ----------------------------------------------------------- verification

def reference(tokens):
    """The fixed-point model — identical arithmetic to spikelm/train_toy.py."""
    mask = (1 << BITS) - 1
    half = 1 << (BITS - 1)
    a = np.array(W["a"]); b = np.array(W["b"])
    emb = np.array(W["emb"]); Wo = np.array(W["w_out"]); bo = np.array(W["b_out"])
    def smul(x, y):
        return np.sign(x) * np.sign(y) * ((np.abs(x) * np.abs(y)) >> FRAC)
    n = tokens.shape[1]
    h = np.zeros((D, n), dtype=np.int64)
    outs = []
    for t in range(tokens.shape[0]):
        u = emb[tokens[t]].T                                  # (D, lanes)
        h = smul(a[:, None], h) + smul(b[:, None], u)
        h = ((h + half) & mask) - half
        lg = np.stack([bo[o] + sum(smul(Wo[o][c], h[c]) for c in range(D))
                       for o in range(V)])
        outs.append(lg.copy())
    return np.array(outs)                                     # (T, V, lanes)


def main():
    print(f"toy model: d={D}, vocab={V}, Q{FRAC}.{FRAC} ({BITS}-bit), "
          f"delay={W['config']['delay']}")
    print(f"trained accuracy: float {W['accuracy_float']:.3f}  "
          f"fixed point {W['accuracy_fixed']:.3f}\n")

    toy = make_toy()
    t0 = time.time()
    sim = compile_seq(toy, (2,))
    tc = time.time() - t0
    gates = len(sim.c.ops)
    print(f"compiled: {gates:,} gates in {tc:.2f}s  "
          f"(logic depth {int(sim.c.depths.max())})")

    steps, lanes = 32, 64
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, V, size=(steps, lanes))
    tok_bus = ((tokens[None] >> np.arange(2)[:, None, None]) & 1).astype(np.int32)

    t0 = time.time()
    trace = sim.run(steps, tok_bus)
    ts = time.time() - t0

    got = np.stack([
        ((trace[o * BITS:(o + 1) * BITS].astype(np.int64)
          << np.arange(BITS)[:, None, None]).sum(0) + (1 << (BITS - 1)))
        % (1 << BITS) - (1 << (BITS - 1))
        for o in range(V)], axis=1)                            # (T, V, lanes)
    want = reference(tokens)

    same = np.array_equal(got, want)
    lag = 0
    if not same and np.array_equal(got[1:], want[:-1]):
        lag, same = 1, True
    print(f"netlist vs model: {'BIT-EXACT' if same else 'MISMATCH'}"
          f"  over {steps} ticks × {lanes} lanes"
          + ("  [registers read one tick behind]" if lag else ""))
    if not same:
        d = np.flatnonzero((got != want).any((1, 2)))
        print(f"  first differing tick: {d[:3]}")
        raise SystemExit(1)

    pred = got[lag:].argmax(1)
    truth = tokens[:len(pred)]                                 # delay-1 target
    acc = float((pred[1:] == truth[:-1]).mean())
    print(f"circuit accuracy on delayed recall: {acc:.3f}")
    print(f"\nsimulation: {steps} ticks × {lanes} lanes in {ts:.2f}s "
          f"= {gates * steps * lanes / ts / 1e6:.1f}M gate-evals/s")
    print(f"per token, one lane: {gates} gates  "
          f"({gates / 1000:.1f}k — a Tiny Tapeout tile is ~1k cells)")


if __name__ == "__main__":
    main()
