# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Johnson counter (running light): a register ring with one inverted
feedback bit. A block of 1s grows around the loop, then shrinks — a 2N-
state sequencer from N registers and a single NOT."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import morpho, CAT, Not
from tiny_morpho_seq import REG, DRIVE, compile_seq

N = 5

@morpho
def johnson():                  # -> q: [N]/step
    q = REG(np.zeros(N, dtype=np.int32))
    DRIVE(q, CAT(Not(q[-1:]), q[:-1]))
    return q

if __name__ == '__main__':
    trace = compile_seq(johnson, ()).run(2 * N + 4)
    for t in range(trace.shape[1]):
        print(f"t={t:2d}  " + ' '.join('●' if b else '·' for b in trace[:, t]))
    states = [tuple(trace[:, t]) for t in range(trace.shape[1])]
    assert states[2 * N] == states[0] and len(set(states[:2 * N])) == 2 * N
    print(f"period 2N = {2 * N}, all {2 * N} states distinct")
