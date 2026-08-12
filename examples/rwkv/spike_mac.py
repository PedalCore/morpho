# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Does spiking reduce logic gates? Measured, not argued.

Spiking changes hardware in exactly two places:

  PRODUCER — the channel-mix nonlinearity:
    float baseline: y = relu(x)^2      -> a 16x16 multiplier
    spiking:        n = clamp(floor(x/0.5), 0, levels); y = n*0.5
                    with the uniform 0.5 threshold in Q8.8 this is
                    n = clamp(x >> 7, 0, 4): pure wiring + a comparator.

  CONSUMER — one MAC lane of the matmul that eats the spike vector (W_v):
    dense:   acc += w8 * x8            -> signed 8x8 multiplier + 24-bit add
    spike-4: acc += w8 * n,  n in 0..4 -> signed 8x3 multiplier + 24-bit add
    binary:  acc += n ? w8 : 0         -> a masked 24-bit add, no multiplier

Every unit below is a real compiled Morpho circuit, verified bit-exact
against numpy on exhaustive or dense-random inputs, then counted (Morpho
gates) and synthesized (yosys -> iCE40 LUT4s) so the answer is a table.
"""

import shutil
import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, REPEAT, Not, And, Or, Xor, LUT,
                         ripple_adder, wallace_multiplier, ZERO, ONE,
                         unpack, pack, compile)

Mux2 = LUT(3, 0b1100_1010)


@morpho
def mul_s8x8(a, b):                 # signed int8 x signed int8 -> signed [16]
    prod = wallace_multiplier(a, b)                        # 16 bits, unsigned
    # correction: subtract (b<<8)*sign(a) and (a<<8)*sign(b), mod 2^16
    ca = CAT(REPEAT(ZERO, a), And(b, REPEAT(a[7:8], b)))   # (b*sa)<<8
    cb = CAT(REPEAT(ZERO, b), And(a, REPEAT(b[7:8], a)))   # (a*sb)<<8
    s1, _ = ripple_adder(prod, Not(ca), ONE)
    s2, _ = ripple_adder(s1, Not(cb), ONE)
    return s2


@morpho
def mul_s8x3(a, n):                 # signed int8 x unsigned 3-bit -> signed [11]
    prod = wallace_multiplier(a, n)                        # 11 bits
    corr = CAT(REPEAT(ZERO, a), And(n, REPEAT(a[7:8], n))) # (n*sa)<<8
    s, _ = ripple_adder(prod, Not(corr), ONE)
    return s


def _sext(x, ref):
    return CAT(x, REPEAT(x[len(x) - 1:len(x)], ref[len(x):]))


@morpho
def mac_dense(w, x, acc):           # acc[24] += w8 * x8
    p = mul_s8x8(w, x)
    s, _ = ripple_adder(acc, _sext(p, acc), ZERO)
    return s


@morpho
def mac_spike4(w, n, acc):          # acc[24] += w8 * n(0..4)
    p = mul_s8x3(w, n)
    s, _ = ripple_adder(acc, _sext(p, acc), ZERO)
    return s


@morpho
def mac_binary(w, n, acc):          # acc[24] += n ? w8 : 0
    wm = And(w, REPEAT(n, w))
    s, _ = ripple_adder(acc, _sext(wm, acc), ZERO)
    return s


@morpho
def act_relusq(x):                  # Q8.8: y = relu(x)^2 >> 8, saturated [16]
    t = And(x, REPEAT(Not(x[15:16]), x))                   # relu: mask if neg
    prod = wallace_multiplier(t, t)                        # 32 bits (t>=0)
    y = prod[8:24]                                         # >> 8, Q8.8
    hi = prod[23:]                                         # y >= 2^15 -> saturate
    ov = hi[0:1]
    for i in range(1, len(hi)):
        ov = Or(ov, hi[i:i + 1])
    return Mux2(y, CAT(REPEAT(ONE, y[:15]), ZERO), ov)     # sat to 0x7FFF


@morpho
def act_spike4(x):                  # Q8.8: n = clamp(x>>7, 0, 4); y = n<<7
    neg = x[15:16]
    ge4 = Or(Or(x[9:10], x[10:11]), Or(x[11:12],           # x>=2.0 (any bit>=9)
             Or(x[12:13], Or(x[13:14], x[14:15]))))
    n0 = And(x[7:8], Not(neg))                             # raw count bits
    n1 = And(x[8:9], Not(neg))
    sat = And(ge4, Not(neg))
    # n = sat ? 4 : (n1 n0);  y = n * 0.5 in Q8.8 = n << 7
    b0 = And(n0, Not(sat))
    b1 = And(n1, Not(sat))
    y = CAT(REPEAT(ZERO, x[:7]), b0, b1, sat, REPEAT(ZERO, x[:6]))
    return y


# ------------------------------------------------------------------ checks

def ref_mac_dense(w, x, acc):
    return (acc + w * x) & 0xFFFFFF


def ref_mac_spike4(w, n, acc):
    return (acc + w * n) & 0xFFFFFF


def ref_mac_binary(w, n, acc):
    return (acc + w * n) & 0xFFFFFF


def ref_act_relusq(x):
    t = np.maximum(x, 0).astype(np.int64)
    y = (t * t) >> 8
    return np.minimum(y, 0x7FFF)


def ref_act_spike4(x):
    n = np.clip(x >> 7, 0, 4)
    return n << 7


def toraw(v, bits):
    return np.asarray(v).astype(np.int64) & ((1 << bits) - 1)


if __name__ == '__main__':
    rng = np.random.default_rng(9)
    N = 200000
    w = rng.integers(-128, 128, N)
    x = rng.integers(-128, 128, N)
    n3 = rng.integers(0, 5, N)
    n1 = rng.integers(0, 2, N)
    acc = rng.integers(0, 1 << 24, N)
    xq = rng.integers(-32768, 32768, N)

    units = []
    for name, cell, widths, ins, ref in [
        ('MAC dense  (w8 x x8 + acc24)', mac_dense, (8, 8, 24),
         (w, x, acc), ref_mac_dense(w, x, acc)),
        ('MAC spike-4 (w8 x n3 + acc24)', mac_spike4, (8, 3, 24),
         (w, n3, acc), ref_mac_spike4(w, n3, acc)),
        ('MAC binary  (w8 gated + acc24)', mac_binary, (8, 1, 24),
         (w, n1, acc), ref_mac_binary(w, n1, acc)),
        ('ACT relu^2  (Q8.8, saturated)', act_relusq, (16,),
         (xq,), ref_act_relusq(xq)),
        ('ACT spike-4 (Q8.8 threshold)', act_spike4, (16,),
         (xq,), ref_act_spike4(xq)),
    ]:
        c = compile(cell, widths)
        got = pack(c(*[unpack(toraw(a, wd), wd) for a, wd in zip(ins, widths)]))
        ow = len(c.outputs) if not isinstance(c.outputs, tuple) else None
        want = np.asarray(ref).astype(np.int64) & ((1 << ow) - 1)
        assert (got == want).all(), f'{name}: mismatch'
        gates = sum(1 for op in c.ops if op.type == 'GATE')
        depth = int(max(c.depths))
        units.append((name, cell.__name__, gates, depth))
        print(f'{name:32s} {gates:5d} gates  depth {depth:2d}  '
              f'({N:,} random cases bit-exact)')

    if shutil.which('yosys'):
        print()
        from tiny_morpho_hw import to_blif
        out = pathlib.Path(__file__).parent / 'netlists'
        out.mkdir(exist_ok=True)
        for name, fn, gates, depth in units:
            widths = {'mac_dense': (8, 8, 24), 'mac_spike4': (8, 3, 24),
                      'mac_binary': (8, 1, 24), 'act_relusq': (16,),
                      'act_spike4': (16,)}[fn]
            cell = {'mac_dense': mac_dense, 'mac_spike4': mac_spike4,
                    'mac_binary': mac_binary, 'act_relusq': act_relusq,
                    'act_spike4': act_spike4}[fn]
            blif = out / f'{fn}.blif'
            blif.write_text(to_blif(compile(cell, widths), fn))
            r = subprocess.run(['yosys', '-p',
                                f'read_blif {blif}; synth_ice40 -device u; stat'],
                               capture_output=True, text=True)
            luts = 0
            for ln in r.stdout.splitlines():
                parts = ln.split()
                if (len(parts) == 2 and parts[1].startswith('SB_LUT')
                        and parts[0].isdigit()):
                    luts = max(luts, int(parts[0]))
            print(f'{name:32s} {luts:5d} iCE40 LUT4s')
