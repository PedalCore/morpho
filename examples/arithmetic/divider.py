# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Division in MorphoHDL: the remainder is a carry.

Restoring long division processes the dividend MSB-first: shift a bit
into the remainder, compare against the divisor, subtract if it fits,
and the comparison bit IS the quotient bit. The comparison comes free
from the article's own ripple adder: the carry-out of rem + ~b + 1 is
exactly (rem >= b), and the sum is the difference.

The recursive cell has precisely the ripple adder's shape — but where
the adder threads a 1-bit carry through the recursion, the divider
threads the whole remainder bus:

    div_rec(a, b, rem) :  SPLIT a; solve the high half first;
                          its remainder feeds the low half.

And because the per-bit stage is a pure function of (remainder, next
bit), wrapping it in REG gives a STREAMING divider: feed any dividend
MSB-first, one subtractor wide as the divisor, and the quotient bits
come out in real time — long division through time, the same rotation
that turned the ripple adder into the serial adder.

Divisor must be nonzero (restoring division's usual precondition)."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, SPLIT, REPEAT, Not, LUT,
                         ripple_adder, ZERO, ONE, unpack, pack, compile)
from tiny_morpho_seq import REG, DRIVE, compile_seq

Mux2 = LUT(3, 0b1100_1010)          # (x0, x1, sel) -> sel ? x1 : x0

@morpho
def div_bit(a_bit, b, rem):         # a_bit:[1], b:[M], rem:[M] -> q:[1], rem':[M]
    shifted = CAT(a_bit, rem)       # 2*rem + a_bit  (little-endian)
    diff, ge = ripple_adder(shifted, Not(CAT(b, ZERO)), ONE)
    rem_new = Mux2(shifted, diff, ge)[:-1]   # top bit provably 0
    return ge, rem_new

@morpho(fallback=div_bit)
def div_rec(a, b, rem):             # a:[N], b:[M], rem:[M] -> q:[N], rem':[M]
    a_lo, a_hi = SPLIT(a)           # high half first: MSB-first recursion
    q_hi, r_mid = div_rec(a_hi, b, rem)
    q_lo, r_out = div_rec(a_lo, b, r_mid)
    return CAT(q_lo, q_hi), r_out

@morpho
def divider(a, b):                  # a:[N], b:[M] -> q:[N], r:[M]
    return div_rec(a, b, REPEAT(ZERO, b))

@morpho
def serial_divider(a_bit, b):       # streams MSB-first; q bit per tick
    M = len(b)
    rem = REG(np.zeros(M, dtype=np.int32))
    q, rem_new = div_bit(a_bit, b, rem)
    DRIVE(rem, rem_new)
    return q


def test_combinational(n=16, m=8, cases=2000):
    rng = np.random.default_rng(42)
    a = rng.integers(1 << n, size=cases)
    b = rng.integers(1, 1 << m, size=cases)
    q, r = divider(unpack(a, n), unpack(b, m))
    assert (pack(q) == a // b).all() and (pack(r) == a % b).all()
    cq, cr = compile(divider, (n, m))(unpack(a, n), unpack(b, m))
    assert (pack(cq) == a // b).all() and (pack(cr) == a % b).all()
    print(f"combinational {n}/{m}: {cases} random divisions exact "
          f"(dynamic + compiled)")

def test_streaming(n=16, m=8, cases=512):
    rng = np.random.default_rng(7)
    a = rng.integers(1 << n, size=cases)
    b = rng.integers(1, 1 << m, size=cases)
    sim = compile_seq(serial_divider, (1, m))
    a_bits = unpack(a, n)[::-1]                    # MSB first
    x_a = a_bits[None]                             # (1, T=n, cases)
    x_b = np.repeat(unpack(b, m)[:, None, :], n, axis=1)   # constant bus
    q_stream = sim.run(n, x_a, x_b)                # (1, n, cases)
    q = pack(q_stream[0][::-1])                    # collect MSB-first bits
    assert (q == a // b).all()
    print(f"streaming {n}/{m}: quotient bits emitted in real time over "
          f"{n} ticks, {cases} cases exact — one subtractor, any length")

if __name__ == '__main__':
    test_combinational()
    test_streaming()
    c = compile(divider, (16, 8))
    gates = sum(1 for op in c.ops if op.type == 'GATE')
    print(f"16/8 combinational divider: {gates} gates after optimization")
    print("division verified: the remainder rides the recursion "
          "like a carry")
