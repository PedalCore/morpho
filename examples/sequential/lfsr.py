# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""LFSR necklace: a shift register whose input is the XOR of two taps.
With primitive taps the state walks through every nonzero pattern —
pseudo-random sequences, test patterns and lightweight counters from a
handful of gates. Width 4 with taps (3, 2) is maximal: period 15."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import morpho, CAT, Xor, pack
from tiny_morpho_seq import REG, DRIVE, compile_seq

W, TAPS = 4, (3, 2)

@morpho
def lfsr():                     # -> s: [W]/step
    s = REG(np.eye(1, W, 0, dtype=np.int32)[0])
    fb = Xor(s[TAPS[0]:TAPS[0] + 1], s[TAPS[1]:TAPS[1] + 1])
    DRIVE(s, CAT(fb, s[:-1]))
    return s

if __name__ == '__main__':
    trace = compile_seq(lfsr, ()).run(2 ** W + 2)
    states = pack(trace)
    for t in range(2 ** W - 1):
        print(f"t={t:2d}  " + ''.join('#' if b else '.' for b in trace[:, t])
              + f"   {states[t]:2d}")
    period = next(t for t in range(1, len(states)) if states[t] == states[0])
    assert period == 2 ** W - 1 and 0 not in states
    print(f"maximal: period {period} = 2^{W}-1, never reaches zero")
