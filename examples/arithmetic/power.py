# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Exponentiation in MorphoHDL: square-and-multiply as bus recursion.

a^e mod 2^N by binary exponentiation. The combinational version is a
Morpho recursion over the EXPONENT bus — peel one exponent bit per
level, conditionally multiply the accumulator, square the base, recurse
on the rest; the fallback returns the accumulator when the exponent bus
is exhausted. The multiplier is the article's own Wallace tree,
truncated to N bits (mod 2^N for free in binary).

The sequential version shows the space/time rotation at its most
extreme: MSB-first square-and-multiply needs ONLY the accumulator as
state —

    acc' = e_bit ? acc^2 * a : acc^2

so an entire exponentiator is one register bank and two multipliers,
taking one tick per exponent bit, for any exponent length."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, LSLICE, REPEAT, LUT, ONE, ZERO,
                         wallace_multiplier, unpack, pack, compile)
from tiny_morpho_seq import REG, DRIVE, compile_seq

Mux2 = LUT(3, 0b1100_1010)          # (x0, x1, sel) -> sel ? x1 : x0

@morpho
def mul_mod(a, b):                  # a:[N], b:[N] -> a*b mod 2^N
    return wallace_multiplier(a, b)[:len(a)]

@morpho(fallback=2)                 # exponent exhausted -> return acc
def power_rec(base, e, acc):        # LSB-first square-and-multiply
    e0, rest = LSLICE(e, ONE)
    acc2 = Mux2(acc, mul_mod(acc, base), e0)
    return power_rec(mul_mod(base, base), rest, acc2)

@morpho
def power(a, e):                    # a:[N], e:[E] -> a^e mod 2^N
    one_bus = CAT(ONE, REPEAT(ZERO, a[1:]))
    return power_rec(a, e, one_bus)

@morpho
def serial_power(a, e_bit):         # e streamed MSB-first, one tick/bit
    acc = REG(np.concatenate([[1], np.zeros(7)]).astype(np.int32))
    sq = mul_mod(acc, acc)
    DRIVE(acc, Mux2(sq, mul_mod(sq, a), e_bit))
    return acc


def test_combinational(n=8, ew=4, cases=1500):
    rng = np.random.default_rng(3)
    a = rng.integers(1 << n, size=cases)
    e = rng.integers(1 << ew, size=cases)
    truth = np.array([pow(int(x), int(k), 1 << n) for x, k in zip(a, e)])
    p = power(unpack(a, n), unpack(e, ew))
    assert (pack(p) == truth).all()
    cp = compile(power, (n, ew))(unpack(a, n), unpack(e, ew))
    assert (pack(cp) == truth).all()
    print(f"combinational a^e mod 2^{n} (e up to {(1 << ew) - 1}): "
          f"{cases} cases exact (dynamic + compiled)")

def test_sequential(n=8, ew=6, cases=256):
    rng = np.random.default_rng(9)
    a = rng.integers(1 << n, size=cases)
    e = rng.integers(1 << ew, size=cases)
    truth = np.array([pow(int(x), int(k), 1 << n) for x, k in zip(a, e)])
    sim = compile_seq(serial_power, (n, 1))
    x_a = np.repeat(unpack(a, n)[:, None, :], ew, axis=1)  # constant bus
    x_e = unpack(e, ew)[::-1][None]                        # MSB first
    trace = sim.run(ew + 1, np.concatenate([x_a, x_a[:, :1]], 1),
                    np.concatenate([x_e, np.zeros((1, 1, cases),
                                                  dtype=np.int64)], 1))
    got = pack(trace[:, ew])           # acc after ew commits
    assert (got == truth).all()
    print(f"sequential a^e mod 2^{n}: one accumulator register, "
          f"{ew}-bit exponents streamed MSB-first, {cases} cases exact")

if __name__ == '__main__':
    test_combinational()
    test_sequential()
    c = compile(power, (8, 4))
    gates = sum(1 for op in c.ops if op.type == 'GATE')
    seq = compile_seq(serial_power, (8, 1))
    m = seq.metrics()
    print(f"combinational 8-bit/4-bit-exponent: {gates} gates")
    print(f"sequential: {m['registers']} registers + {m['gates']} gates, "
          f"one tick per exponent bit, any exponent length")
    print("exponentiation verified: square-and-multiply as a recursion "
          "over the exponent bus")
