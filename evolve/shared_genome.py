# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Shared-rule baseline: one evolved local stage rule repeated k times, with
no hierarchical development. Isolates the effect of parameter tying (one
mutation changes every stage) from recursive composition. |genome| is
constant (31 ints); the phenotype is a linear pipeline of k identical
stages, each with one register, threaded by a two-signal carry."""

import numpy as np

from tiny_morpho import LUT, morpho, ZERO, ONE
from tiny_morpho_seq import REG, DRIVE
from .develop_genome import _col, _nodes, spec_random

SHARED_SPEC = [
    # stage: sigs = [c0, c1, cin0, cin1, r] + 3 LUT nodes
    ('s_arity', (3,), 'arity'), ('s_lut', (3,), 'lut'),
    ('s_refs', (3, 3), _col(3, 5)),
    ('s_drive', (1,), 8), ('s_cout', (2,), 8),
    # top: init carry from [c0, c1, x]; readout over [c0, c1, x, co0, co1]
    ('i_in', (2,), 3),
    ('o_arity', (2,), 'arity'), ('o_lut', (2,), 'lut'),
    ('o_refs', (2, 3), _col(2, 5)),
    ('o_out', (1,), 7),
]

def instantiate_shared(g, k):
    @morpho
    def net(x):                                   # x: [1]/step -> y: [1]/step
        xs = x[0:1]
        tavail = [ZERO, ONE, xs]
        carry = [tavail[int(r) % 3] for r in g['i_in']]
        for _ in range(k):
            state = REG(np.zeros(1, dtype=np.int32))
            sigs = [ZERO, ONE] + carry + [state[0:1]]
            _nodes(sigs, g['s_arity'], g['s_lut'], g['s_refs'], 'S')
            DRIVE(state, sigs[int(g['s_drive'][0]) % len(sigs)])
            carry = [sigs[int(r) % len(sigs)] for r in g['s_cout']]
        rsigs = [ZERO, ONE, xs] + carry
        _nodes(rsigs, g['o_arity'], g['o_lut'], g['o_refs'], 'O')
        return rsigs[int(g['o_out'][0]) % len(rsigs)]
    return net

def hand_shared():
    """Hand parity law: carry = (delay line value, running parity).
    init: (x, 0); stage: r <- cin0; cout = (r, cin1 ^ r); y = x ^ cout1."""
    g = spec_random(np.random.default_rng(0), SHARED_SPEC)
    XOR = 0b0110
    g['i_in'][:] = [2, 0]
    g['s_drive'][0] = 2                      # r <- cin0
    g['s_arity'][0], g['s_lut'][0] = 2, XOR  # node = cin1 ^ r
    g['s_refs'][0][:2] = [3, 4]
    g['s_cout'][:] = [4, 5]                  # cout = (r, node)
    g['o_arity'][0], g['o_lut'][0] = 2, XOR  # y = x ^ co1
    g['o_refs'][0][:2] = [2, 4]
    g['o_out'][0] = 5
    return g
