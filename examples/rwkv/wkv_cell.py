# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""The complete RWKV wkv channel as ONE compiled Morpho circuit.

This is the cell the wkv-cell page simulates, made of gates. The spec is
the page's `Q.step` — Q8.8 saturating fixed point, base-2 exponentials via
the exhaustively-verified exp2 unit, floor-semantics multiplies, a `den+1`
zero guard, 32-step restoring division, pp initialized to -32768 — and the
numpy reference below is a line-by-line transcription of that JavaScript.
The circuit is verified BIT-EXACT against it, single steps and streams.

One structural liberty, provably identity-preserving: after the max
subtraction one of each exponential pair is exp2(0) = 65536 exactly, and
mulQ16(z, 65536) = z for every z — so instead of four exp2 units and six
multipliers the circuit muxes on the comparison and instantiates two
exp2 units and four multipliers. Bit-identical, ~40% smaller.

State: aa, bb, pp — three 16-bit registers per channel (the model's entire
recurrent memory). Inputs per tick: kq, vq (the token), uq, wq (the
channel's trained constants, base-2, Q8.8). Output: wkv, Q8.8.

Invariant relied on for the division (maintained by the update from the
zero init, as in the page): bb >= 0, hence den >= 1.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, REPEAT, Not, And, Or, Xor, LUT,
                         ripple_adder, wallace_multiplier, ZERO, ONE,
                         unpack, pack, compile)
from tiny_morpho_seq import REG, DRIVE, compile_seq
from examples.arithmetic.exp2 import exp2neg, reference as exp2_ref
from examples.arithmetic.divider import div_rec

Mux2 = LUT(3, 0b1100_1010)          # (x0, x1, sel) -> sel ? x1 : x0


def _or_tree(bus):
    out = bus[:1]
    for i in range(1, len(bus)):
        out = Or(out, bus[i:i + 1])
    return out


@morpho
def neg16(z):                       # two's complement negate
    s, _ = ripple_adder(Not(z), REPEAT(ZERO, z), ONE)
    return s


@morpho
def sat_add(a, b):                  # 16-bit signed saturating add
    s, _ = ripple_adder(a, b, ZERO)
    a15, b15, s15 = a[15:16], b[15:16], s[15:16]
    ovf = And(Not(Xor(a15, b15)), Xor(s15, a15))
    fill = Not(a15)                 # 0x7FFF on +ovf, 0x8000 on -ovf
    lo = Mux2(s[:15], REPEAT(fill, s[:15]), ovf)
    return CAT(lo, Mux2(s15, a15, ovf))


@morpho
def ge_absdiff(a, b):               # signed a>=b, and |a-b| (fits 16 bits)
    d, _ = ripple_adder(a, Not(b), ONE)
    a15, b15, d15 = a[15:16], b[15:16], d[15:16]
    ge = Mux2(Not(a15), Not(d15), Not(Xor(a15, b15)))
    return ge, Mux2(neg16(d), d, ge)


@morpho
def mul_q16(z, e):                  # floor(z*e / 2^16): z signed Q8.8, e Q0.16
    prod = wallace_multiplier(e, z)              # e * (z as unsigned), 33 bits
    corr = CAT(REPEAT(ZERO, z), And(e, REPEAT(z[15:16], e)))
    d, _ = ripple_adder(prod, Not(corr), ONE)    # subtract (e<<16)*sign(z)
    return d[16:32]


@morpho
def sdiv_sat(num, den):             # SAT((num<<8)/den): num signed, den>=1
    s = num[15:16]
    mag = Mux2(num, neg16(num), s)               # |num| (0x8000 -> 32768)
    q, _ = div_rec(CAT(REPEAT(ZERO, num[:8]), mag), den, REPEAT(ZERO, den))
    hi = _or_tree(q[15:])                        # q > 32767
    pos = CAT(Mux2(q[:15], REPEAT(ONE, q[:15]), hi), REPEAT(ZERO, s))
    big = Or(_or_tree(q[16:]), And(q[15:16], _or_tree(q[:15])))   # q > 32768
    minpat = CAT(REPEAT(ZERO, q[:15]), ONE)      # -32768
    neg = Mux2(neg16(q[:16]), minpat, big)
    return Mux2(pos, neg, s)


@morpho
def wkv_step(kq, vq, uq, wq, aa, bb, pp):
    c256 = CAT(REPEAT(ZERO, kq[:8]), ONE, REPEAT(ZERO, kq[:7]))
    # ---- output: wkv = (e1*aa + e2*v) / (e1*bb + e2), winner's e == 1.0
    ww = sat_add(uq, kq)
    sel, t = ge_absdiff(pp, ww)                  # sel: pp is the max
    e = exp2neg(t)                               # the loser's factor, Q0.16
    e88 = CAT(e[8:17], REPEAT(ZERO, e[:7]))      # e as Q8.8
    num = sat_add(Mux2(vq, aa, sel), mul_q16(Mux2(aa, vq, sel), e))
    den_pre = Mux2(sat_add(mul_q16(bb, e), c256), sat_add(bb, e88), sel)
    den, _ = ripple_adder(den_pre, REPEAT(ZERO, den_pre), ONE)
    out = sdiv_sat(num, den)
    # ---- update: aa' = f1*aa + f2*v, bb' = f1*bb + f2, pp' = max
    ww2 = sat_add(pp, wq)
    sel2, t2 = ge_absdiff(ww2, kq)               # sel2: decay path is the max
    g = exp2neg(t2)
    g88 = CAT(g[8:17], REPEAT(ZERO, g[:7]))
    aa2 = sat_add(Mux2(vq, aa, sel2), mul_q16(Mux2(aa, vq, sel2), g))
    bb2 = Mux2(sat_add(mul_q16(bb, g), c256), sat_add(bb, g88), sel2)
    pp2 = Mux2(kq, ww2, sel2)
    return out, aa2, bb2, pp2


@morpho
def wkv_cell(kq, vq, uq, wq):       # the sequential cell: 48 bits of state
    aa = REG(np.zeros(16, dtype=np.int32))
    bb = REG(np.zeros(16, dtype=np.int32))
    pp = REG(np.array([0] * 15 + [1], dtype=np.int32))     # -32768
    out, aa2, bb2, pp2 = wkv_step(kq, vq, uq, wq, aa, bb, pp)
    DRIVE(aa, aa2)
    DRIVE(bb, bb2)
    DRIVE(pp, pp2)
    return out


# ------------------------------------------------------------- the reference
# line-by-line transcription of the wkv-cell page's Q.step (site/wkv-cell.html)

def ref_step(aa, bb, pp, kq, vq, uq, wq):
    SAT = lambda x: np.clip(x, -32768, 32767).astype(np.int64)
    mul = lambda a, e: (a.astype(np.int64) * e.astype(np.int64)) >> 16
    ww = SAT(uq + kq)
    p = np.maximum(pp, ww)
    e1, e2 = exp2_ref(p - pp), exp2_ref(p - ww)
    num = SAT(mul(aa, e1) + mul(vq, e2))
    den = SAT(mul(bb, e1) + (e2 >> 8)) + 1
    q = (np.abs(num) << 8) // den
    out = SAT(np.where(num < 0, -q, q))
    ww2 = SAT(pp + wq)
    q2 = np.maximum(ww2, kq)
    f1, f2 = exp2_ref(q2 - ww2), exp2_ref(q2 - kq)
    aa2 = SAT(mul(aa, f1) + mul(vq, f2))
    bb2 = SAT(mul(bb, f1) + (f2 >> 8))
    return out, aa2, bb2, q2


def ref_run(kq, vq, uq, wq):        # streams (T, S) -> outputs (T, S)
    T, S = kq.shape
    aa = np.zeros(S, dtype=np.int64)
    bb = np.zeros(S, dtype=np.int64)
    pp = np.full(S, -32768, dtype=np.int64)
    outs = np.zeros((T, S), dtype=np.int64)
    for t in range(T):
        outs[t], aa, bb, pp = ref_step(aa, bb, pp, kq[t], vq[t], uq[t], wq[t])
    return outs


def float_run(k2, v, u2, w2):       # float64 base-2 recurrence, same shapes
    T, S = k2.shape
    aa = np.zeros(S)
    bb = np.zeros(S)
    pp = np.full(S, -1e30)
    outs = np.zeros((T, S))
    for t in range(T):
        ww = u2[t] + k2[t]
        p = np.maximum(pp, ww)
        e1, e2 = 2.0 ** (pp - p), 2.0 ** (ww - p)
        outs[t] = (e1 * aa + e2 * v[t]) / (e1 * bb + e2)
        ww2 = pp + w2[t]
        p2 = np.maximum(ww2, k2[t])
        f1, f2 = 2.0 ** (ww2 - p2), 2.0 ** (k2[t] - p2)
        aa = f1 * aa + f2 * v[t]
        bb = f1 * bb + f2
        pp = p2
    return outs


# ------------------------------------------------------------------- helpers

def to_raw(x):                      # signed int -> 16-bit two's complement
    return np.asarray(x).astype(np.int64) & 0xFFFF


def from_raw(p):                    # 16-bit unsigned -> signed
    p = np.asarray(p).astype(np.int64)
    return np.where(p >= 32768, p - 65536, p)


# (w2_raw, u2_raw) from the trained model's atlas export, Q8.8:
# block 0 median & slowest, block 3 median, block 5 median & slowest
PRESETS = [(-302, 164), (-3, 187), (-55, 62), (-23, 134), (-3, 132)]


def make_streams(T, per_preset, rng):
    """Realistic token streams over the trained presets. (T, S) int64."""
    S = len(PRESETS) * per_preset
    k = np.clip(rng.normal(0, 1.1, (T, S)), -6, 6)
    v = np.clip(rng.normal(0, 0.9, (T, S)), -6, 6)
    kq = np.round(k * 256).astype(np.int64)
    vq = np.round(v * 256).astype(np.int64)
    wq = np.repeat([p[0] for p in PRESETS], per_preset)[None].repeat(T, 0)
    uq = np.repeat([p[1] for p in PRESETS], per_preset)[None].repeat(T, 0)
    return kq, vq, uq.astype(np.int64), wq.astype(np.int64)


# --------------------------------------------------------------------- tests

def test_step(cases=20000):
    rng = np.random.default_rng(11)
    aa = rng.integers(-32768, 32768, cases)
    bb = rng.integers(0, 32768, cases)           # invariant: bb >= 0
    pp = rng.integers(-32768, 32768, cases)
    kq = rng.integers(-32768, 32768, cases)
    vq = rng.integers(-32768, 32768, cases)
    uq = rng.integers(-512, 512, cases)
    wq = rng.integers(-8192, 1, cases)
    # edge battery: init state, rails, den=1 path, divider clamps
    edges = np.array([
        [0, 0, -32768, 0, 0, 164, -302],
        [0, 0, -32768, -32768, 32767, 187, -3],
        [32767, 0, 0, 32767, -32768, 511, -8192],
        [-32768, 1, 100, -300, 300, 0, -1],
        [-32768, 0, 32767, 32767, 32767, 511, 0],
        [32767, 32767, -100, 200, -200, 187, -3],
    ], dtype=np.int64).T
    aa = np.concatenate([edges[0], aa]); bb = np.concatenate([edges[1], bb])
    pp = np.concatenate([edges[2], pp]); kq = np.concatenate([edges[3], kq])
    vq = np.concatenate([edges[4], vq]); uq = np.concatenate([edges[5], uq])
    wq = np.concatenate([edges[6], wq])

    want = ref_step(aa, bb, pp, kq, vq, uq, wq)
    args = [unpack(to_raw(x), 16) for x in (kq, vq, uq, wq, aa, bb, pp)]
    got_dyn = wkv_step(*args)
    circuit = compile(wkv_step, (16,) * 7)
    got_cmp = circuit(*args)
    for got in (got_dyn, got_cmp):
        for g, w in zip(got, want):
            assert (from_raw(pack(g)) == w).all()
    gates = sum(1 for op in circuit.ops if op.type == 'GATE')
    print(f'wkv_step: {len(aa):,} vectors bit-exact vs the page reference '
          f'(dynamic + compiled), {gates} gates')
    return circuit


def test_stream(T=96, per_preset=64):
    rng = np.random.default_rng(23)
    kq, vq, uq, wq = make_streams(T, per_preset, rng)
    want = ref_run(kq, vq, uq, wq)
    sim = compile_seq(wkv_cell, (16,) * 4)
    S = kq.shape[1]
    streams = [unpack(to_raw(x).ravel(), 16).reshape(16, T, S)
               for x in (kq, vq, uq, wq)]
    raw = sim.run(T, *streams)                    # (16, T, S)
    got = from_raw(pack(raw.reshape(16, -1)).reshape(T, S))
    assert (got == want).all(), 'stream mismatch'
    regs = sum(1 for op in sim.c.ops if op.type == 'REG')
    print(f'wkv_cell: {T} ticks x {kq.shape[1]} channels bit-exact vs the '
          f'page reference through compile_seq ({regs} registers)')
    return sim


def test_float_tracking(T=256, per_preset=40):
    rng = np.random.default_rng(31)
    kq, vq, uq, wq = make_streams(T, per_preset, rng)
    got = ref_run(kq, vq, uq, wq) / 256.0
    ref = float_run(kq / 256.0, vq / 256.0, uq / 256.0, wq / 256.0)
    err = got - ref
    rms = float(np.sqrt((err ** 2).mean()))
    worst = float(np.abs(err).max())
    sig = float(np.sqrt((ref ** 2).mean()))
    print(f'circuit vs float64 recurrence over {T} steps, trained presets: '
          f'rms {rms:.2e} (signal rms {sig:.2f}), worst {worst:.2e}')
    assert rms < 2e-2, 'fixed-point tracking degraded'


if __name__ == '__main__':
    test_step()
    test_stream()
    test_float_tracking()
    print('the wkv channel is a compiled Morpho circuit, '
          'bit-exact against the page that designed it')
