"""morpho_stochastic — continuous-valued computation on boolean gates.

The memristive crossbar's appeal is that one device performs a multiply. The
digital crossbar in morpho_crossbar.py needs ~302 gates for the same thing,
exactly. Stochastic computing sits between them: encode a value as the
probability of a 1 in a bit stream, and

    multiply  =  ONE GATE          (XNOR, in the bipolar encoding)
    add       =  ONE MUX           (giving the mean, not the sum)

with accuracy that improves as the stream lengthens rather than being fixed
by a word width. That is analog's character — cheap operators, error traded
against time, composition that degrades — on ordinary logic, describable in
MorphoHDL exactly, and verifiable.

Encoding (bipolar): a value v in [-1, 1] is a stream whose bits are 1 with
probability (v+1)/2. Then XNOR(a, b) is 1 with probability corresponding to
the product, and a MUX driven by a fair random select averages its inputs.
A dot product therefore comes out SCALED by 1/n — the standard stochastic
trade, and the reason the select bits are part of the circuit.

    python morpho_stochastic.py
"""

import numpy as np

from tiny_morpho import morpho, compile, LUT, LSLICE, CAT, VOID, mux

Xnor = LUT(2, 0b1001)          # bipolar multiply: one gate
ONEBIT = np.zeros((1, 1), np.int32)


# ---------------------------------------------------------------- the cells

@morpho
def sc_mul(a, b):
    """A multiply. One gate — the whole point of the exercise."""
    return Xnor(a, b)


def _prod_done(w, x):
    return VOID


@morpho(fallback=_prod_done)
def sc_products(w, x):
    """One crosspoint per input: the entire multiply half of a crossbar row."""
    w0, wr = LSLICE(w, ONEBIT)
    x0, xr = LSLICE(x, ONEBIT)
    return CAT(sc_mul(w0, x0), sc_products(wr, xr))


@morpho
def sc_dot(w, x, sel):
    """A dot product: n gates and a mux tree, scaled by 1/n.

    The mux is the adder. Driven by uniformly random select bits it emits
    each product with equal probability, so the output stream's value is the
    MEAN of the products. Scaling is intrinsic to the encoding, not a
    normalisation step bolted on afterwards.
    """
    return mux(sc_products(w, x), sel)


# -------------------------------------------------------------- encode/decode

def encode(values, length, rng):
    """value in [-1,1] -> bit stream, 1 with probability (v+1)/2."""
    p = (np.asarray(values, dtype=np.float64) + 1.0) / 2.0
    return (rng.random((len(p), length)) < p[:, None]).astype(np.int32)


def decode(stream):
    """bit stream -> value in [-1,1]."""
    return 2.0 * np.asarray(stream, dtype=np.float64).mean(-1) - 1.0


# ------------------------------------------------------------ verification

def run_dot(w_vals, x_vals, length, rng):
    """Evaluate the Morpho circuit with stream bits laid along the lane axis."""
    n = len(w_vals)
    sbits = int(np.log2(n))
    assert 1 << sbits == n, "mux tree needs a power-of-two fan-in"
    # every stream independent: separate draws, no shared randomness
    w = encode(w_vals, length, rng)
    x = encode(x_vals, length, rng)
    sel = rng.integers(0, 2, size=(sbits, length)).astype(np.int32)
    # encode() already returns (n_values, stream_length) — which IS a Morpho
    # bus: n one-bit values across `length` lanes. No reshaping needed.
    out = sc_dot(w, x, sel)
    return decode(out[0])


def check_multiply(trials=6, length=8192, seed=0):
    rng = np.random.default_rng(seed)
    print("  one XNOR gate as a multiplier:")
    worst = 0.0
    for _ in range(trials):
        a, b = rng.uniform(-1, 1, size=2)
        sa, sb = encode([a], length, rng), encode([b], length, rng)
        got = decode(sc_mul(sa, sb)[0])
        err = abs(got - a * b)
        worst = max(worst, err)
        print(f"    {a:+.3f} × {b:+.3f} = {a*b:+.3f}   circuit {got:+.3f}   "
              f"err {err:.4f}")
    return worst


def check_dot(n=4, length=8192, trials=4, seed=1):
    rng = np.random.default_rng(seed)
    print(f"\n  a {n}-input dot product (scaled by 1/{n}):")
    errs = []
    for _ in range(trials):
        w = rng.uniform(-1, 1, size=n)
        x = rng.uniform(-1, 1, size=n)
        want = float(np.dot(w, x) / n)
        got = run_dot(w, x, length, rng)
        errs.append(abs(got - want))
        print(f"    want {want:+.4f}   circuit {got:+.4f}   err {errs[-1]:.4f}")
    return float(np.mean(errs))


def accuracy_curve(n=4, seed=2):
    print("\n  accuracy against stream length — error falls as 1/sqrt(L):")
    print(f"    {'bits':>7}  {'mean |err|':>10}  {'1/sqrt(L)':>10}")
    rng = np.random.default_rng(seed)
    for length in (64, 256, 1024, 4096, 16384):
        errs = []
        for _ in range(12):
            w = rng.uniform(-1, 1, size=n)
            x = rng.uniform(-1, 1, size=n)
            errs.append(abs(run_dot(w, x, length, rng) - np.dot(w, x) / n))
        print(f"    {length:>7}  {np.mean(errs):>10.4f}  {1/np.sqrt(length):>10.4f}")


def correlation_warning(n=4, length=8192, seed=3):
    """Independence is load-bearing: reuse one stream and the answer breaks."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, size=n)
    sbits = int(np.log2(n))
    ws = encode(w, length, rng)
    sel = rng.integers(0, 2, size=(sbits, length)).astype(np.int32)
    good = decode(sc_dot(ws, encode(w, length, rng), sel)[0])
    bad = decode(sc_dot(ws, ws, sel)[0])
    want = float(np.dot(w, w) / n)
    print(f"\n  independence is load-bearing (w·w/{n} = {want:+.4f}):")
    print(f"    independent streams   {good:+.4f}   err {abs(good-want):.4f}")
    print(f"    same stream reused    {bad:+.4f}   err {abs(bad-want):.4f}"
          "   <- correlation, not noise")


def cost(n):
    sbits = int(np.log2(n))
    c = compile(sc_dot, [n, n, sbits])
    return len(c.ops), int(c.depths.max())


if __name__ == "__main__":
    print("morpho_stochastic — a multiply that costs one gate\n")
    worst = check_multiply()
    mean_err = check_dot()
    accuracy_curve()
    correlation_warning()

    print("\ngates, against the exact digital crossbar:")
    print(f"  {'inputs':>7}  {'gates':>7}  {'depth':>6}  {'gates/multiply':>15}")
    for n in (2, 4, 8, 16, 32):
        g, d = cost(n)
        print(f"  {n:>7}  {g:>7}  {d:>6}  {g / n:>15.1f}")
    g32, _ = cost(32)
    print(f"\n  stochastic  {g32/32:.1f} gates per multiply, ~8 bits of accuracy "
          f"after ~16k stream bits")
    print( "  exact       302 gates per multiply, exact after 1 cycle")
    print(f"  ratio       {302/(g32/32):.0f}x fewer gates, paid for in time")
    print( "  memristive  1 device per multiply, ~128 ns, 1.83-5.61 dB error")
