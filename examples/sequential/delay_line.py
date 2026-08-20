# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Delay line: bits travel down a register chain, one cell per tick.

    q = REG(zeros(N)); DRIVE(q, CAT(x, q[:-1])); return q[-1:]

The simplest possible use of REG — remember one signal — and a real
primitive used everywhere in pipelines, synchronization and alignment."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import morpho, CAT
from tiny_morpho_seq import REG, DRIVE, compile_seq

N = 4

@morpho
def delay_line(x):              # x: [1]/step -> y: [1]/step, delayed N ticks
    q = REG(np.zeros(N, dtype=np.int32))
    DRIVE(q, CAT(x, q[:-1]))
    return q[-1:]

def scope(name, bits):
    print(f"{name}: " + ''.join('▔' if b else '▁' for b in bits))

if __name__ == '__main__':
    sim = compile_seq(delay_line, (1,))
    steps = 24
    x = np.zeros((1, steps), dtype=np.int32)
    x[0, [2, 3, 9, 15, 16, 17]] = 1
    y = sim.run(steps, x)
    scope('x', x[0])
    scope('y', y[0])
    assert (y[0][N:] == x[0][:-N]).all() and (y[0][:N] == 0).all()
    print(f"echo verified: y[t] == x[t-{N}]")
