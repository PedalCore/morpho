# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Typed genotype for synchronous recurrent Boolean networks.

    s[t+1] = F(s[t], x[t])        y[t] = G(s[t], x[t])

The combinational part is a LUT DAG: each LUT gene has an arity (1-3), a
truth table, and source references that may point to constants 0/1, input
bits, current register outputs, or strictly earlier LUTs — never later ones.
Register next-state and output sources may point anywhere. Every recurrent
path therefore crosses a REG, so the compiled graph is always synchronous
(no FORWARD/TIE, no async settling).

Genome slots are fixed-size; unreferenced LUTs and registers vanish in the
compiler's dead-code elimination, so add/delete of hardware emerges from
rewiring and hardware cost is measured on the live optimized phenotype.
Register initial state is fixed to zero.
"""

import numpy as np

from tiny_morpho import LUT, morpho, CAT, ZERO, ONE
from tiny_morpho_seq import REG, DRIVE

# Signal index space: [const0, const1 | x bits | registers | LUT nodes]
N_CONST = 2

def sig_base(cfg):
    return N_CONST + cfg['x_n'] + cfg['state_n']

def random_genome(rng, cfg):
    node_n, state_n = cfg['node_n'], cfg['state_n']
    base = sig_base(cfg)
    return {
        'arity': rng.integers(1, 4, size=node_n).astype(np.uint8),
        'luts': rng.integers(256, size=node_n).astype(np.uint8),
        'refs': np.array([[rng.integers(base + j) for _ in range(3)]
                          for j in range(node_n)], dtype=np.int16),
        'drives': rng.integers(base + node_n, size=state_n).astype(np.int16),
        'outs': rng.integers(base + node_n, size=cfg['out_n']).astype(np.int16),
    }

def genome_to_cell(g, cfg):
    x_n, s_n = cfg['x_n'], cfg['state_n']
    luts = [LUT(int(a), int(l) & ((1 << (1 << int(a))) - 1), name=f'L{j}')
            for j, (a, l) in enumerate(zip(g['arity'], g['luts']))]
    @morpho
    def net(x):                     # x: [x_n] per step -> y: [out_n] per step
        state = REG(np.zeros(s_n, dtype=np.int32))
        sigs = [ZERO, ONE] + [x[i:i + 1] for i in range(x_n)] \
             + [state[i:i + 1] for i in range(s_n)]
        for j, lutf in enumerate(luts):
            a = int(g['arity'][j])
            sigs.append(lutf(*[sigs[int(r)] for r in g['refs'][j][:a]]))
        DRIVE(state, CAT(*[sigs[int(r)] for r in g['drives']]))
        return CAT(*[sigs[int(r)] for r in g['outs']])
    return net

def shift_register(d, cfg):
    """The hand optimum for delayed recall: d live registers, zero gates.
    drives: s0 <- x, s_i <- s_{i-1}; output <- s_{d-1}."""
    g = random_genome(np.random.default_rng(0), cfg)
    x_ref, s_ref = N_CONST, N_CONST + cfg['x_n']
    g['drives'][:] = 0
    g['drives'][0] = x_ref
    for i in range(1, d):
        g['drives'][i] = s_ref + i - 1
    g['outs'][0] = s_ref + d - 1
    return g
