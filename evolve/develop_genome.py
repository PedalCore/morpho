# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Developmental genotype: a constant-length recursive Morpho program whose
instantiation at size k generates an O(k) synchronous circuit.

The grammar is generic — no task-specific primitives. A cell has a fixed
two-signal interface. The recursive case splits its size in two (any k, not
just powers of two), instantiates child A on evolvable selections of its
inputs, instantiates child B on selections that may include A's outputs,
then combines everything through a small bank of evolvable LUT nodes. The
base case (size 1) is a two-register micro-circuit with its own LUT bank.
Registers exist only in base cases, so every recurrent path crosses a REG
and phenotype register count grows with k while |genome| stays fixed.

Genome = fixed dict of small integer arrays (63 ints total), independent
of instantiation size.
"""

import numpy as np

from tiny_morpho import LUT, morpho, CAT, ZERO, ONE
from tiny_morpho_seq import REG, DRIVE

def _col(n, base):
    return (np.arange(n)[:, None] + base).astype(np.int16)

# (key, shape, bound): bound is 'lut', 'arity', an int, or a per-element array.
DEV_SPEC = [
    # base case: sigs = [c0, c1, u0, u1, r0, r1] + 4 LUT nodes
    ('b_arity', (4,), 'arity'), ('b_lut', (4,), 'lut'),
    ('b_refs', (4, 3), _col(4, 6)),
    ('b_drives', (2,), 10), ('b_outs', (2,), 10),
    # recursive case: avail = [c0, c1, u0, u1] -> A -> +[A0, A1] -> B ->
    # +[B0, B1] + 4 LUT nodes
    ('a_in', (2,), 4), ('b_in', (2,), 6),
    ('r_arity', (4,), 'arity'), ('r_lut', (4,), 'lut'),
    ('r_refs', (4, 3), _col(4, 8)),
    ('r_outs', (2,), 12),
    # top: inputs from [c0, c1, x]; readout over [c0, c1, x, v0, v1] + 2 nodes
    ('t_in', (2,), 3),
    ('t_arity', (2,), 'arity'), ('t_lut', (2,), 'lut'),
    ('t_refs', (2, 3), _col(2, 5)),
    ('t_out', (1,), 7),
]

def spec_random(rng, spec):
    g = {}
    for key, shape, bound in spec:
        if isinstance(bound, str) and bound == 'lut':
            g[key] = rng.integers(256, size=shape).astype(np.uint8)
        elif isinstance(bound, str):
            g[key] = rng.integers(1, 4, size=shape).astype(np.uint8)
        else:
            g[key] = rng.integers(np.broadcast_to(bound, shape)).astype(np.int16)
    return g

def spec_mutate(genome, rng, spec, op_n=None):
    g = {k: v.copy() for k, v in genome.items()}
    for _ in range(op_n if op_n is not None else rng.integers(1, 4)):
        key, shape, bound = spec[rng.integers(len(spec))]
        flat = g[key].reshape(-1)
        i = rng.integers(flat.size)
        if isinstance(bound, str) and bound == 'lut':
            if rng.random() < .5:
                flat[i] ^= np.uint8(1 << rng.integers(8))
            else:
                flat[i] = rng.integers(256)
        elif isinstance(bound, str):
            flat[i] = rng.integers(1, 4)
        else:
            flat[i] = rng.integers(np.broadcast_to(bound, shape).reshape(-1)[i])
    return g

def genome_size(spec):
    return sum(int(np.prod(shape)) for _, shape, _ in spec)


def _nodes(sigs, arity, luts, refs, tag):
    for j in range(len(arity)):
        a = int(arity[j])
        f = LUT(a, int(luts[j]) & ((1 << (1 << a)) - 1), name=f'{tag}{j}')
        f_args = [sigs[int(r) % len(sigs)] for r in refs[j][:a]]
        sigs.append(f(*f_args))

def instantiate_dev(g, k):
    @morpho
    def net(x):                                   # x: [1]/step -> y: [1]/step
        xs = x[0:1]

        def cell(us, size):
            if size == 1:
                state = REG(np.zeros(2, dtype=np.int32))
                sigs = [ZERO, ONE] + us + [state[0:1], state[1:2]]
                _nodes(sigs, g['b_arity'], g['b_lut'], g['b_refs'], 'B')
                DRIVE(state, CAT(sigs[int(g['b_drives'][0]) % len(sigs)],
                                 sigs[int(g['b_drives'][1]) % len(sigs)]))
                return [sigs[int(r) % len(sigs)] for r in g['b_outs']]
            avail = [ZERO, ONE] + us
            a = cell([avail[int(r) % len(avail)] for r in g['a_in']], size // 2)
            avail += a
            b = cell([avail[int(r) % len(avail)] for r in g['b_in']],
                     size - size // 2)
            sigs = avail + b
            _nodes(sigs, g['r_arity'], g['r_lut'], g['r_refs'], 'R')
            return [sigs[int(r) % len(sigs)] for r in g['r_outs']]

        tavail = [ZERO, ONE, xs]
        vs = cell([tavail[int(r) % 3] for r in g['t_in']], k)
        rsigs = [ZERO, ONE, xs] + vs
        _nodes(rsigs, g['t_arity'], g['t_lut'], g['t_refs'], 'T')
        return rsigs[int(g['t_out'][0]) % len(rsigs)]
    return net


def hand_dev(task='parity'):
    """The human-written developmental law, expressed in the grammar.
    Interface: v0 = input delayed by size, v1 = parity of the delayed window.
    base:  r0 <- u0; v0 = v1 = r0
    rec:   A on u0; B on A.v0; v0 = B.v0; v1 = A.v1 XOR B.v1
    top:   parity: y = x XOR v1;  recall: y = v0
    """
    g = spec_random(np.random.default_rng(0), DEV_SPEC)
    XOR = 0b0110
    g['b_drives'][:] = [2, 0]                 # r0 <- u0, r1 <- const 0
    g['b_outs'][:] = [4, 4]                   # v0 = v1 = r0
    g['a_in'][:] = [2, 0]                     # A gets (u0, const)
    g['b_in'][:] = [4, 0]                     # B gets (A.v0, const)
    g['r_arity'][0], g['r_lut'][0] = 2, XOR   # node = A.v1 ^ B.v1
    g['r_refs'][0][:2] = [5, 7]
    g['r_outs'][:] = [6, 8]                   # v0 = B.v0, v1 = node
    g['t_in'][:] = [2, 0]                     # cell input = x
    g['t_arity'][0], g['t_lut'][0] = 2, XOR   # y = x ^ v1
    g['t_refs'][0][:2] = [2, 4]
    g['t_out'][0] = 5
    if task == 'recall':
        g['t_out'][0] = 3                     # y = v0 = x delayed by k
    return g
