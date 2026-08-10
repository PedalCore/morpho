# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""2-D NCA lattice: vectorized numpy stepper and the honest Morpho circuit.

Both implement the identical semantics — synchronous update, von Neumann
neighbourhood, fixed-zero boundaries (documented choice: the developing
system has a physical edge; a torus is a later control) — and must agree
bit-exactly (asserted by the Experiment 4A selftest before any large-grid
result is reported).

The Morpho circuit is ordinary tiny_morpho_seq structure: one C-bit REG
bank per cell; each threshold neuron is a popcount tree of full adders over
its +1 inputs and its -1 inputs (biases become constant ONE inputs) and a
ripple comparator P >= M. Every recurrent path crosses REG; no FORWARD/TIE.
Register order is row-major cells x channels, matching pack_state()."""

import numpy as np

from tiny_morpho import morpho, CAT, ZERO, ONE, full_adder, ripple_adder
from tiny_morpho_seq import REG, DRIVE
from .nca_genome import C, H, IN_N


#@MARK: vectorized stepper

def perceive(s):
    """s: (C, ny, nx[, batch]) -> (5C, ny*nx[*batch]) neighbour inputs."""
    X = np.zeros((5,) + s.shape, dtype=np.int16)
    X[0] = s
    X[1][:, 1:] = s[:, :-1]        # N = (y-1, x)
    X[2][:, :-1] = s[:, 1:]        # S
    X[3][:, :, :-1] = s[:, :, 1:]  # E = (y, x+1)
    X[4][:, :, 1:] = s[:, :, :-1]  # W
    return X.reshape(IN_N, -1)

def step_np(g, s):
    h = (g['w1'].astype(np.int16) @ perceive(s)
         + g['b1'][:, None] >= 0).astype(np.int16)
    out = (g['w2'].astype(np.int16) @ h + g['b2'][:, None] >= 0)
    return out.astype(s.dtype).reshape(s.shape)

def rollout(g, s, steps, record=False):
    frames = [s.copy()]
    for _ in range(steps):
        s = step_np(g, s)
        if record:
            frames.append(s.copy())
    return frames if record else s


#@MARK: Morpho circuit

def _popcount(wires):
    """Sum a list of width-1 wires into a little-endian bus (list)."""
    if not wires:
        return [ZERO]
    cols, out, w = {0: list(wires)}, [], 0
    while w in cols:
        col = cols[w]
        while len(col) > 1:
            a = col.pop()
            b = col.pop()
            c = col.pop() if col else ZERO
            s, carry = full_adder(a, b, c)
            col.append(s)
            cols.setdefault(w + 1, []).append(carry)
        out.append(col[0])
        w += 1
    return out

def _ge(a_bus, b_bus):
    """1 iff value(a) >= value(b): carry-out of a + ~b + 1."""
    from tiny_morpho import Not
    n = max(len(a_bus), len(b_bus))
    a = CAT(*(a_bus + [ZERO] * (n - len(a_bus))))
    b = CAT(*(b_bus + [ZERO] * (n - len(b_bus))))
    _, c_out = ripple_adder(a, Not(b), ONE)
    return c_out

def _threshold(inputs, w_row, bias):
    pos = [inputs[i] for i in np.nonzero(w_row == 1)[0]] \
        + [ONE] * max(int(bias), 0)
    neg = [inputs[i] for i in np.nonzero(w_row == -1)[0]] \
        + [ONE] * max(-int(bias), 0)
    return _ge(_popcount(pos), _popcount(neg))

def instantiate_nca(g, ny, nx):
    @morpho
    def nca():                       # closed system: state develops from REGs
        cells = [[REG(np.zeros(C, dtype=np.int32)) for _ in range(nx)]
                 for _ in range(ny)]
        for y in range(ny):
            for x in range(nx):
                nbs = [cells[y][x],
                       cells[y - 1][x] if y > 0 else None,
                       cells[y + 1][x] if y < ny - 1 else None,
                       cells[y][x + 1] if x < nx - 1 else None,
                       cells[y][x - 1] if x > 0 else None]
                inputs = [nb[c:c + 1] if nb is not None else ZERO
                          for nb in nbs for c in range(C)]
                hidden = [_threshold(inputs, g['w1'][j], g['b1'][j])
                          for j in range(H)]
                outs = [_threshold(hidden, g['w2'][c], g['b2'][c])
                        for c in range(C)]
                DRIVE(cells[y][x], CAT(*outs))
        return CAT(*[cells[y][x] for y in range(ny) for x in range(nx)])
    return nca

def pack_state(s):
    """(C, ny, nx) numpy state -> flat register vector (row-major cells,
    channels within cell), the Morpho circuit's register order."""
    return np.moveaxis(s, 0, -1).reshape(-1)

def unpack_state(v, ny, nx):
    return np.moveaxis(v.reshape(ny, nx, C), -1, 0)
