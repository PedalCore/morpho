# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""The 2^(-x) unit for the wkv cell, as a real Morpho circuit.

Input: x in Q8.8 (unsigned, x >= 0 — wkv's max-subtraction guarantees
non-positive exponents, so only the decaying half exists). Output:
2^(-x) in Q0.16 on a 17-bit bus (value 65536 = 1.0 exactly at x = 0).

Structure, exactly as proposed for hardware:
  - fractional part: top 5 fraction bits address a 32-entry ROM
    (arity-5 LUTs, one per output bit — ROM tables are just ints to
    Morpho), linear interpolation on the remaining 3 bits via an 11x3
    Wallace multiply:  interp = A - ((A - B) * rem >> 3)
  - integer part: the article's logarithmic right_shifter, shift by n
  - n >= 16 underflows to zero (numerically correct for this use)

Verified EXHAUSTIVELY: all 65,536 inputs through the compiled circuit
against the reference model — the same semantics the wkv-cell page
simulates — so the page's 2^(-x) is bit-identical to compiled Morpho."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, REPEAT, Not, LUT, Or,
                         ripple_adder, right_shifter, wallace_multiplier,
                         ZERO, ONE, unpack, pack, compile)

LUTN = 32
TABLE = [round(65536 * 2 ** (-i / LUTN)) for i in range(LUTN)] + [32768]

Mux2 = LUT(3, 0b1100_1010, 'Mux2')

def _rom(bit, entries, name):
    """One output bit of a 32-entry ROM: two arity-4 LUTs muxed on the
    top address bit (the natural 4-LUT FPGA mapping; also keeps tables
    within the evaluator's 32-bit arithmetic)."""
    lo = hi = 0
    for i, v in enumerate(entries):
        b = (v >> bit) & 1
        if i < 16:
            lo |= b << i
        else:
            hi |= b << (i - 16)
    f_lo = LUT(4, lo, f'{name}{bit}L')
    f_hi = LUT(4, hi, f'{name}{bit}H')
    return lambda i0, i1, i2, i3, i4: Mux2(f_lo(i0, i1, i2, i3),
                                           f_hi(i0, i1, i2, i3), i4)

ROM_A = [_rom(b, TABLE[:LUTN], 'RomA') for b in range(17)]
ROM_B = [_rom(b, TABLE[1:LUTN + 1], 'RomB') for b in range(17)]

@morpho
def exp2neg(x):                     # x: [16] Q8.8 -> y: [17] Q0.16
    rem, rest = x[:3], x[3:]        # 3 interp bits
    idx, n = rest[:5], rest[5:]     # 5 ROM bits, 8 integer bits
    ib = [idx[j:j + 1] for j in range(5)]
    a = CAT(*[ROM_A[b](*ib) for b in range(17)])
    b_ = CAT(*[ROM_B[b](*ib) for b in range(17)])
    # d = a - b (positive, < 2^11)
    d_full, _ = ripple_adder(a, Not(b_), ONE)
    d = d_full[:11]
    prod = wallace_multiplier(d, rem)          # 14 bits
    delta = CAT(prod[3:], *([ZERO] * 6))       # (prod >> 3), padded to 17
    interp, _ = ripple_adder(a, Not(delta), ONE)
    shifted = right_shifter(interp, n[:4], ZERO)
    zero_out = Or(Or(n[4:5], n[5:6]), Or(n[6:7], n[7:8]))   # n >= 16
    return Mux2(shifted, REPEAT(ZERO, interp), zero_out)


def reference(xq):
    """The wkv-cell page's semantics (floor interpolation)."""
    n = xq >> 8
    idx = (xq >> 3) & 31
    rem = xq & 7
    a = np.array(TABLE)[idx]
    d = a - np.array(TABLE)[idx + 1]
    val = a - ((d * rem) >> 3)
    out = val >> np.minimum(n, 30)
    return np.where(n >= 16, 0, out)


if __name__ == '__main__':
    circuit = compile(exp2neg, (16,))
    xs = np.arange(1 << 16)
    got = np.zeros(1 << 16, dtype=np.int64)
    for lo in range(0, 1 << 16, 8192):
        chunk = xs[lo:lo + 8192]
        got[lo:lo + 8192] = pack(circuit(unpack(chunk, 16)))
    want = reference(xs)
    assert (got == want).all(), \
        f"first mismatch at x={int(np.argmax(got != want))}"
    gates = sum(1 for op in circuit.ops if op.type == 'GATE')
    worst = np.max(np.abs(want / 65536 - 2.0 ** (-xs / 256.0)))
    print(f"2^(-x): EXHAUSTIVE bit-exact over all 65,536 inputs "
          f"(compiled Morpho vs reference model)")
    print(f"unit size: {gates} gates | worst error vs true 2^(-x): "
          f"{worst:.2e} (32-entry ROM + 3-bit interpolation)")
