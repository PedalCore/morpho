# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Binary counter / clock divider: q <- q + 1 every tick, built from the
article's own ripple adder wrapped in registers. Each output bit toggles
at half the frequency of the bit below it — temporal hierarchy made
visible on an oscilloscope."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import morpho, CAT, REPEAT, ripple_adder, ZERO, ONE, pack
from tiny_morpho_seq import REG, DRIVE, compile_seq

N = 4

@morpho
def counter():                  # -> q: [N]/step, q(t) = t mod 2^N
    q = REG(np.zeros(N, dtype=np.int32))
    one = CAT(ONE, REPEAT(ZERO, q[1:]))
    total, _ = ripple_adder(q, one, ZERO)
    DRIVE(q, total)
    return q

if __name__ == '__main__':
    steps = 2 ** N + 3
    trace = compile_seq(counter, ()).run(steps)
    for k in range(N):
        wave = ''.join('▔' if b else '▁' for b in trace[k])
        print(f"bit {k} (/{2 ** (k + 1):2d}): {wave}")
    assert (pack(trace) == np.arange(steps) % 2 ** N).all()
    print(f"q(t) == t mod {2 ** N} verified for {steps} ticks")
