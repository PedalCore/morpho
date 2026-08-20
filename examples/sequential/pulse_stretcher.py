# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Pulse stretcher: hold any input pulse high for N extra ticks — a delay
chain ORed with the live input. The most quietly useful circuit in the
gallery: debouncing, edge widening, LED visibility, watchdog kicks."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import morpho, CAT, Or
from tiny_morpho_seq import REG, DRIVE, compile_seq

N = 3

@morpho
def stretch(x):                 # x: [1]/step -> y: [1]/step
    q = REG(np.zeros(N, dtype=np.int32))
    DRIVE(q, CAT(x, q[:-1]))
    y = x
    for i in range(N):
        y = Or(y, q[i:i + 1])
    return y

def scope(name, bits):
    print(f"{name}: " + ''.join('▔' if b else '▁' for b in bits))

if __name__ == '__main__':
    sim = compile_seq(stretch, (1,))
    steps = 24
    x = np.zeros((1, steps), dtype=np.int32)
    x[0, [3, 11, 12, 19]] = 1
    y = sim.run(steps, x)
    scope('x', x[0])
    scope('y', y[0])
    expect = x[0].copy()
    for d in range(1, N + 1):
        expect[d:] |= x[0][:-d]
    assert (y[0] == expect).all()
    print(f"every pulse held high for {N} extra ticks")
