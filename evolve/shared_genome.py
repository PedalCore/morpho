# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Shared-rule baseline: one evolved local stage rule repeated k times, with
no hierarchical development. Isolates the effect of parameter tying (one
mutation changes every stage) from recursive composition. |genome| is
constant (31 ints); the phenotype is a linear pipeline of k identical
stages, each with one register, threaded by a two-signal carry."""

import numpy as np

from tiny_morpho import LUT, morpho, CAT, ZERO, ONE
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

#@MARK: generic grammar family (carry width P, stage register count)
# The P=2 / 1-reg definitions above are FROZEN. make_shared_grammar builds
# the same family with carry width p and `regs` registers per stage.

def make_shared_grammar(p=2, x_n=1, regs=1):
    stage_sigs = 2 + p + regs
    spec = [
        ('s_arity', (3,), 'arity'), ('s_lut', (3,), 'lut'),
        ('s_refs', (3, 3), _col(3, stage_sigs)),
        ('s_drives', (regs,), stage_sigs + 3),
        ('s_cout', (p,), stage_sigs + 3),
        ('i_in', (p,), 2 + x_n),
        ('o_arity', (2,), 'arity'), ('o_lut', (2,), 'lut'),
        ('o_refs', (2, 3), _col(2, 2 + x_n + p)),
        ('o_out', (1,), 2 + x_n + p + 2),
    ]

    def instantiate(g, k):
        @morpho
        def net(x):
            xs = [x[i:i + 1] for i in range(x_n)]
            tavail = [ZERO, ONE] + xs
            carry = [tavail[int(r) % len(tavail)] for r in g['i_in']]
            for _ in range(k):
                state = REG(np.zeros(regs, dtype=np.int32))
                sigs = [ZERO, ONE] + carry + [state[i:i + 1]
                                              for i in range(regs)]
                _nodes(sigs, g['s_arity'], g['s_lut'], g['s_refs'], 'S')
                DRIVE(state, CAT(*[sigs[int(r) % len(sigs)]
                                   for r in g['s_drives']]))
                carry = [sigs[int(r) % len(sigs)] for r in g['s_cout']]
            rsigs = [ZERO, ONE] + xs + carry
            _nodes(rsigs, g['o_arity'], g['o_lut'], g['o_refs'], 'O')
            return rsigs[int(g['o_out'][0]) % len(rsigs)]
        return net

    return {'name': f'shared_p{p}', 'spec': spec, 'instantiate': instantiate,
            'p': p, 'x_n': x_n}

def hand_shared_copy(grammar):
    """Hand copy law; requires p=3, x_n=2, 2 regs/stage.
    init: (x_data, 0, x_cue); stage: r_w <- cin0; r_b <- MUX(cin1, r_w, cin2);
    cout = (r_w, r_b, cin2); y = cout1."""
    from .develop_genome import MUX
    assert grammar['p'] == 3 and grammar['x_n'] == 2
    g = spec_random(np.random.default_rng(0), grammar['spec'])
    # stage sigs: [c0, c1, cin0, cin1, cin2, r0, r1] -> node at 7
    g['s_arity'][0], g['s_lut'][0] = 3, MUX
    g['s_refs'][0][:] = [3, 5, 4]             # MUX(cin1, r0, cin2)
    g['s_drives'][:] = [2, 7]                 # r0 <- cin0, r1 <- mux node
    g['s_cout'][:] = [5, 6, 4]                # (r0, r1, cin2)
    g['i_in'][:] = [2, 0, 3]
    g['o_out'][0] = 5                         # y = cout1
    return g


def hand_shared(task='parity'):
    """Hand law: carry = (delay line value, running parity).
    init: (x, 0); stage: r <- cin0; cout = (r, cin1 ^ r);
    parity: y = x ^ cout1;  recall: y = cout0."""
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
    if task == 'recall':
        g['o_out'][0] = 3                    # y = cout0 = x delayed by k
    return g
