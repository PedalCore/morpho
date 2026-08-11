# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Typed component grammar for compositional synthesis.

The tack: small verified vocabulary -> automatic type-valid composition ->
larger useful programs, instead of random-LUT soup -> hopefully-useful
circuit. Every component carries a contract (input widths, output width,
parameter space, statefulness); compositions are DAGs over those
contracts, so search never constructs a type-invalid program, and a
mutation is 'replace Johnson counter with LFSR' or 'insert a delay', not
'flip truth-table bit 183'.

Each component has three synchronized definitions: a contract, a fast
numpy interpreter (used during search), and a Morpho builder producing
real tiny_morpho_seq structure. The interpreter is verified bit-exact
against the compiled Morpho circuit (Experiment C0 selftest), and
finalists are always re-verified through compile_seq — the same
fast-path-with-exactness pattern as Experiments 0 and 4A.

A program is a DAG: [(component, params, input_indices)], inputs
referring to earlier nodes only; its output is the last node. Outputs of
stateful components are their register Q (state before commit), matching
synchronous Morpho semantics exactly. W = 8 is the tentacle from the
'circuits that move' page."""

import numpy as np

from tiny_morpho import (morpho, CAT, LUT, Not, Xor, Or, And, ripple_adder,
                         ZERO, ONE)
from tiny_morpho_seq import REG, DRIVE, compile_seq

W = 8            # actuator segments
CTR_W = 3        # counter width


#@MARK: Morpho builders for stateful components

def _b_toggle():
    q = REG(np.zeros(1, np.int32))
    DRIVE(q, Not(q))
    return q

def _b_ring():
    q = REG(np.eye(1, W, 0, np.int32)[0])
    DRIVE(q, CAT(q[-1:], q[:-1]))
    return q

def _b_johnson():
    q = REG(np.zeros(W, np.int32))
    DRIVE(q, CAT(Not(q[-1:]), q[:-1]))
    return q

def _b_lfsr():
    q = REG(np.eye(1, W, 0, np.int32)[0])
    DRIVE(q, CAT(Xor(q[-1:], q[-2:-1]), q[:-1]))
    return q

def _b_counter():
    q = REG(np.zeros(CTR_W, np.int32))
    one = CAT(ONE, *([ZERO] * (CTR_W - 1)))
    total, _ = ripple_adder(q, one, ZERO)
    DRIVE(q, total)
    return q

def _b_delay(k, x):
    cur = np.atleast_1d(x)
    for _ in range(k):
        q = REG(np.zeros(len(cur), np.int32))
        DRIVE(q, cur)
        cur = q
    return cur

def _b_mux(a, b, sel):
    return LUT(3, 0b1100_1010, 'Mux2')(a, b, sel)


#@MARK: component library
# contract: ins (widths; 'n' = polymorphic, all 'n's equal), out, params
# step(state, ins, params) -> (out, state'); init(params, in_width)

def _ctr_inc(s):
    v = int((s * (1 << np.arange(CTR_W))).sum()) + 1
    return ((v >> np.arange(CTR_W)) & 1).astype(np.int8)

COMPONENTS = {
    # generators
    'const0':  dict(ins=(), out=1, params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (np.zeros(1, np.int8), s),
                    build=lambda p, i: ZERO),
    'const1':  dict(ins=(), out=1, params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (np.ones(1, np.int8), s),
                    build=lambda p, i: ONE),
    'stripes': dict(ins=(), out=W, params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: ((np.arange(W) % 2 == 0)
                                          .astype(np.int8), s),
                    build=lambda p, i: CAT(*[(ONE if k % 2 == 0 else ZERO)
                                             for k in range(W)])),
    'toggle':  dict(ins=(), out=1, params=[()],
                    init=lambda p, w: np.zeros(1, np.int8),
                    step=lambda s, i, p: (s, 1 - s),
                    build=lambda p, i: _b_toggle()),
    'ring':    dict(ins=(), out=W, params=[()],
                    init=lambda p, w: np.eye(1, W, 0, np.int8)[0],
                    step=lambda s, i, p: (s, np.concatenate([s[-1:], s[:-1]])),
                    build=lambda p, i: _b_ring()),
    'johnson': dict(ins=(), out=W, params=[()],
                    init=lambda p, w: np.zeros(W, np.int8),
                    step=lambda s, i, p: (s, np.concatenate([1 - s[-1:],
                                                             s[:-1]])),
                    build=lambda p, i: _b_johnson()),
    'lfsr':    dict(ins=(), out=W, params=[()],
                    init=lambda p, w: np.eye(1, W, 0, np.int8)[0],
                    step=lambda s, i, p: (s, np.concatenate(
                        [s[-1:] ^ s[-2:-1], s[:-1]])),
                    build=lambda p, i: _b_lfsr()),
    'counter': dict(ins=(), out=CTR_W, params=[()],
                    init=lambda p, w: np.zeros(CTR_W, np.int8),
                    step=lambda s, i, p: (s, _ctr_inc(s)),
                    build=lambda p, i: _b_counter()),
    # transformers
    'not_':    dict(ins=('n',), out='n', params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (1 - i[0], s),
                    build=lambda p, i: Not(i[0])),
    'xor_':    dict(ins=('n', 'n'), out='n', params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (i[0] ^ i[1], s),
                    build=lambda p, i: Xor(i[0], i[1])),
    'or_':     dict(ins=('n', 'n'), out='n', params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (i[0] | i[1], s),
                    build=lambda p, i: Or(i[0], i[1])),
    'and_':    dict(ins=('n', 'n'), out='n', params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (i[0] & i[1], s),
                    build=lambda p, i: And(i[0], i[1])),
    'mux':     dict(ins=('n', 'n', 1), out='n', params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (np.where(i[2][0] > 0, i[1], i[0])
                                          .astype(np.int8), s),
                    build=lambda p, i: _b_mux(i[0], i[1], i[2])),
    'rotate':  dict(ins=(W,), out=W, params=[(1,), (2,), (4,)],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (np.roll(i[0], p[0]), s),
                    build=lambda p, i: CAT(i[0][-p[0]:], i[0][:-p[0]])),
    'reverse': dict(ins=(W,), out=W, params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (i[0][::-1], s),
                    build=lambda p, i: i[0][::-1]),
    'delay':   dict(ins=('n',), out='n', params=[(1,), (2,)],
                    init=lambda p, w: np.zeros((p[0], w), np.int8),
                    step=lambda s, i, p: (s[-1], np.concatenate(
                        [i[0][None], s[:-1]], axis=0)),
                    build=lambda p, i: _b_delay(p[0], i[0])),
    'select':  dict(ins=(W,), out=1, params=[(k,) for k in range(W)],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (i[0][p[0]:p[0] + 1], s),
                    build=lambda p, i: i[0][p[0]:p[0] + 1]),
    'sel_ctr': dict(ins=(CTR_W,), out=1, params=[(k,) for k in range(CTR_W)],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (i[0][p[0]:p[0] + 1], s),
                    build=lambda p, i: i[0][p[0]:p[0] + 1]),
    'repeat8': dict(ins=(1,), out=W, params=[()],
                    init=lambda p, w: None,
                    step=lambda s, i, p: (np.repeat(i[0], W), s),
                    build=lambda p, i: np.repeat(np.atleast_1d(i[0]), W, 0)),
}


#@MARK: programs

def out_width(prog, idx):
    comp, params, ins = prog[idx]
    o = COMPONENTS[comp]['out']
    return o if o != 'n' else out_width(prog, ins[0])

def type_ok(prog, comp, ins):
    c = COMPONENTS[comp]
    if len(ins) != len(c['ins']):
        return False
    widths = [out_width(prog, i) for i in ins]
    n = None
    for spec, w in zip(c['ins'], widths):
        if spec == 'n':
            if n is None:
                n = w
            elif w != n:
                return False
        elif spec != w:
            return False
    return True

def _init_states(prog):
    states = []
    for comp, params, ins in prog:
        c = COMPONENTS[comp]
        w = out_width(prog, ins[0]) if c['ins'] and c['ins'][0] == 'n' else 0
        states.append(c['init'](params, w))
    return states

def _tick(prog, states):
    outs, nexts = [], []
    for k, (comp, params, ins) in enumerate(prog):
        c = COMPONENTS[comp]
        o, ns = c['step'](states[k], [outs[i] for i in ins], params)
        outs.append(np.asarray(o, np.int8).reshape(-1))
        nexts.append(ns)
    return outs, nexts

def interpret(prog, steps, all_nodes=False):
    """Trace over `steps` ticks (fast search path). With all_nodes=True,
    returns a list of per-node traces (for behaviour signatures)."""
    states = _init_states(prog)
    trace = [[] for _ in prog] if all_nodes else []
    for _ in range(steps):
        outs, states = _tick(prog, states)
        if all_nodes:
            for k, o in enumerate(outs):
                trace[k].append(o.copy())
        else:
            trace.append(outs[-1].copy())
    if all_nodes:
        return [np.stack(t) for t in trace]
    return np.stack(trace)

def state_key(states):
    """Hashable snapshot of all component states (for exact cycle finding
    in closed systems)."""
    return tuple(None if s is None else s.tobytes() for s in states)

def build_morpho(prog):
    @morpho
    def net():
        vals = []
        for comp, params, ins in prog:
            c = COMPONENTS[comp]
            vals.append(np.atleast_1d(c['build'](params,
                                                 [vals[i] for i in ins])))
        return vals[-1]
    return net

def morpho_trace(prog, steps):
    tr = compile_seq(build_morpho(prog), ()).run(steps)
    return np.atleast_2d(tr).T

def pretty(prog):
    lines = []
    for k, (comp, params, ins) in enumerate(prog):
        args = ', '.join(f'n{i}' for i in ins)
        ps = ', '.join(map(str, params))
        inner = ', '.join(x for x in (args, ps) if x)
        name = 'out' if k == len(prog) - 1 else f'n{k}'
        lines.append(f'{name} = {comp.rstrip("_")}({inner})')
    return '\n'.join(lines)
