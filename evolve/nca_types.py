# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Differentiated neural tissue: K shared cell types placed by a compact
developmental program (Experiment 4B).

The genome holds K complete neural rules (each the frozen 4A architecture)
plus a small recursive placement program: PG placement genes, each with
four child references (one per quadrant) and a base type. Instantiation
recursively splits the lattice into quadrants, following child references,
until single cells receive their type. Genome length is constant at every
lattice size:

    K=4:  4 x 450 neural ints + 6 x 5 placement ints = 1830
    K=1:  450 (placement degenerate; homogeneous control arm)

No coordinates are given to the cells; differentiation is purely which
shared rule a cell runs. The typed stepper and the typed Morpho circuit
share exact semantics with the 4A implementations (bit-exactness asserted
by the Experiment 4B selftest)."""

import numpy as np

from tiny_morpho import morpho, CAT, ZERO
from tiny_morpho_seq import REG, DRIVE
from .nca_genome import C, H, IN_N, random_genome, hand_genome
from .nca_grid import perceive, _threshold

K = 4           # cell types
PG = 6          # placement genes


def random_typed_genome(rng, k=K):
    g = {'nets': [random_genome(rng) for _ in range(k)], 'k': k}
    if k > 1:
        g['children'] = rng.integers(PG, size=(PG, 4)).astype(np.int8)
        g['base'] = rng.integers(k, size=PG).astype(np.int8)
    return g

def typed_genome_size(g):
    n = sum(sum(v.size for v in net.values()) for net in g['nets'])
    return n + (g['children'].size + g['base'].size if g['k'] > 1 else 0)

def typed_nonzero_weights(g):
    return int(sum((net['w1'] != 0).sum() + (net['w2'] != 0).sum()
                   for net in g['nets']))


def type_map(g, ny, nx):
    """Recursive quadrant development of the type assignment."""
    tm = np.zeros((ny, nx), dtype=np.int8)
    if g['k'] == 1:
        return tm

    def assign(gene, y0, y1, x0, x1):
        if y1 - y0 == 1 and x1 - x0 == 1:
            tm[y0, x0] = g['base'][gene]
            return
        ym = y0 + max(1, (y1 - y0) // 2) if y1 - y0 > 1 else y1
        xm = x0 + max(1, (x1 - x0) // 2) if x1 - x0 > 1 else x1
        quads = [(y0, ym, x0, xm), (y0, ym, xm, x1),
                 (ym, y1, x0, xm), (ym, y1, xm, x1)]
        for q, (a, b, c, d) in enumerate(quads):
            if b > a and d > c:
                assign(int(g['children'][gene][q]), a, b, c, d)

    assign(0, 0, ny, 0, nx)
    return tm


def _step_typed_ref(g, s, tm):
    """Reference implementation (kept for equivalence testing)."""
    X = perceive(s)
    out = np.zeros_like(s)
    flat_tm = tm.reshape(-1)
    for k in range(g['k']):
        net = g['nets'][k]
        h = (net['w1'].astype(np.int16) @ X + net['b1'][:, None] >= 0
             ).astype(np.int16)
        o = (net['w2'].astype(np.int16) @ h + net['b2'][:, None] >= 0
             ).astype(np.int16)
        o = o.reshape(s.shape)
        sel = flat_tm == k                     # cells of this type
        mask = sel.reshape(s.shape[1], s.shape[2])
        out[:, mask] = o[:, mask]
    return out.astype(s.dtype)

def step_typed(g, s, tm):
    """One synchronous step; cell (y, x) runs rule nets[tm[y, x]].
    Fast path: each net computes only over its own cells' columns —
    bit-exact with _step_typed_ref (asserted in the 5B selftest)."""
    ny, nx = s.shape[1], s.shape[2]
    X = perceive(s).reshape(IN_N, ny * nx, -1)
    out = np.zeros((C, ny * nx, X.shape[2]), dtype=np.int16)
    flat_tm = tm.reshape(-1)
    for k in range(g['k']):
        sel = flat_tm == k
        if not sel.any():
            continue
        net = g['nets'][k]
        Xk = X[:, sel].reshape(IN_N, -1)
        h = (net['w1'].astype(np.int16) @ Xk + net['b1'][:, None] >= 0
             ).astype(np.int16)
        o = (net['w2'].astype(np.int16) @ h + net['b2'][:, None] >= 0)
        out[:, sel] = o.reshape(C, int(sel.sum()), -1)
    return out.reshape(s.shape).astype(s.dtype)

def rollout_typed(g, tm, s, steps, record=False):
    frames = [s.copy()]
    for _ in range(steps):
        s = step_typed(g, s, tm)
        if record:
            frames.append(s.copy())
    return frames if record else s


def instantiate_typed(g, ny, nx):
    tm = type_map(g, ny, nx)

    @morpho
    def nca():
        cells = [[REG(np.zeros(C, dtype=np.int32)) for _ in range(nx)]
                 for _ in range(ny)]
        for y in range(ny):
            for x in range(nx):
                net = g['nets'][int(tm[y, x])]
                nbs = [cells[y][x],
                       cells[y - 1][x] if y > 0 else None,
                       cells[y + 1][x] if y < ny - 1 else None,
                       cells[y][x + 1] if x < nx - 1 else None,
                       cells[y][x - 1] if x > 0 else None]
                inputs = [nb[c:c + 1] if nb is not None else ZERO
                          for nb in nbs for c in range(C)]
                hidden = [_threshold(inputs, net['w1'][j], net['b1'][j])
                          for j in range(H)]
                outs = [_threshold(hidden, net['w2'][c], net['b2'][c])
                        for c in range(C)]
                DRIVE(cells[y][x], CAT(*outs))
        return CAT(*[cells[y][x] for y in range(ny) for x in range(nx)])
    return nca


def hand_rings_genome():
    """Homogeneous (K=1) hand reference for the rings task. Channels:
    v=0 visible, r=1 reached-from-boundary, m=2 constant medium flag,
    e=3 edge flag. The medium floods to 1; edge cells detect missing
    neighbours through the missing medium; a reached-wave propagates
    inward from the edge; each newly reached cell takes the phase opposite
    to its already-reached (outward) neighbours; v self-latches."""
    net = {'w1': np.zeros((H, IN_N), np.int8),
           'b1': np.full(H, -1, np.int8),
           'w2': np.zeros((C, H), np.int8),
           'b2': np.full(C, -1, np.int8)}
    V, R, M, E = 0, 1, 2, 3
    idx = lambda nb, c: nb * C + c            # [self, N, S, E, W]
    # h0: self.m present
    net['w1'][0][idx(0, M)] = 1
    # h1: some neighbour medium missing (sum of nb.m <= 3)
    for nb in (1, 2, 3, 4):
        net['w1'][1][idx(nb, M)] = -1
    net['b1'][1] = 3
    # h2: wave source/OR: e_self + r_self + r_nbs
    net['w1'][2][idx(0, E)] = 1
    net['w1'][2][idx(0, R)] = 1
    for nb in (1, 2, 3, 4):
        net['w1'][2][idx(nb, R)] = 1
    # h3: origin: edge cell not yet reached -> v = 1 (outermost ring)
    net['w1'][3][idx(0, E)] = 1
    net['w1'][3][idx(0, R)] = -1
    # h4: v latch, conditional on being reached (the canonical seed is
    # phase-irrelevant for a boundary-anchored target and must die out)
    net['w1'][4][idx(0, V)] = 1
    net['w1'][4][idx(0, R)] = 1
    net['b1'][4] = -2
    # h5..h8: frontier anti-phase per neighbour: r_nb & ~v_nb & ~r_self
    for j, nb in enumerate((1, 2, 3, 4)):
        net['w1'][5 + j][idx(nb, R)] = 1
        net['w1'][5 + j][idx(nb, V)] = -1
        net['w1'][5 + j][idx(0, R)] = -1
    # outputs
    net['w2'][V][3:9] = 1                     # v' = h3|h4|h5|h6|h7|h8
    net['w2'][R][2] = 1                       # r' = h2
    net['b2'][M] = 0                          # m' = constant 1
    net['w2'][E][0] = 1                       # e' = h0 & h1
    net['w2'][E][1] = 1
    net['b2'][E] = -2
    return {'nets': [net], 'k': 1}
