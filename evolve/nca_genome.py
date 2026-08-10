# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Quantized neural-CA genome: one small shared threshold network.

Every lattice cell holds C binary state channels and applies the identical
rule synchronously:

    s_i(t+1) = F_theta(s_i(t), N, S, E, W neighbour states)

F_theta is a single-hidden-layer threshold network with ternary weights
{-1, 0, +1} and small integer biases; neuron output = 1 iff
bias + sum(w * x) >= 0. Genome length is a constant (450 ints for the
default C=6, H=12) regardless of lattice dimensions — the phenotype grows,
the hereditary description does not.

Input indexing (canonical, shared with both simulators):
    index = neighbour * C + channel,
    neighbours ordered [self, N, S, E, W], N = (y-1, x), E = (y, x+1).
Channel 0 is the visible/phenotype bit; channels 1..C-1 are free latent
state. Fixed-zero boundaries: off-lattice neighbours read 0.
"""

import numpy as np

C = 6           # state channels per cell
H = 12          # hidden threshold units
BIAS_MAX = 4    # biases in [-BIAS_MAX, BIAS_MAX]
IN_N = 5 * C


def random_genome(rng):
    return {'w1': rng.choice([-1, 0, 1], size=(H, IN_N),
                             p=[.2, .6, .2]).astype(np.int8),
            'b1': rng.integers(-2, 3, size=H).astype(np.int8),
            'w2': rng.choice([-1, 0, 1], size=(C, H),
                             p=[.2, .6, .2]).astype(np.int8),
            'b2': rng.integers(-2, 3, size=C).astype(np.int8)}

def genome_size(g):
    return sum(v.size for v in g.values())

def nonzero_weights(g):
    return int((g['w1'] != 0).sum() + (g['w2'] != 0).sum())


def hand_genome():
    """Hand-designed reference law (expressibility calibration, not a
    claimed neural optimum). Channels: v = ch0 (visible), r = ch1
    ('reached' wavefront). Growth: r spreads at speed 1 from any active
    cell; a cell joining the wave takes the phase opposite to its already-
    reached neighbours; v self-latches once set. Repair: damage clears v
    and r, so the wave regrows from the intact surround with a coherent
    phase. Hidden units: h0 = wavefront OR, h1 = v latch, h2..h5 = per-
    neighbour 'reached anti-phase frontier' detectors."""
    g = {'w1': np.zeros((H, IN_N), dtype=np.int8),
         'b1': np.full(H, -1, dtype=np.int8),
         'w2': np.zeros((C, H), dtype=np.int8),
         'b2': np.full(C, -1, dtype=np.int8)}
    V, R = 0, 1                                   # channel roles
    idx = lambda nb, c: nb * C + c                # [self, N, S, E, W]
    # h0: reached' source = OR(r_self, r_N, r_S, r_E, r_W, v_self)
    for nb in range(5):
        g['w1'][0][idx(nb, R)] = 1
    g['w1'][0][idx(0, V)] = 1
    # h1: v latch
    g['w1'][1][idx(0, V)] = 1
    # h2..h5: frontier from neighbour nb: r_nb & ~v_nb & ~r_self
    for j, nb in enumerate((1, 2, 3, 4)):
        g['w1'][2 + j][idx(nb, R)] = 1
        g['w1'][2 + j][idx(nb, V)] = -1
        g['w1'][2 + j][idx(0, R)] = -1
    # outputs: v' = h1 | h2 | h3 | h4 | h5 ; r' = h0
    g['w2'][V][1:6] = 1
    g['w2'][R][0] = 1
    return g
