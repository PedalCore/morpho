# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment SA1-dev: activity-driven circuit development.

Inspired by Lifelong Neural Developmental Programs (arXiv:2406.09787):
search specifies not a finished network but a developmental program that
differentiates nodes and creates/prunes connections, with most structural
change in an early plastic phase. The Morpho twist: the things being
assembled are TYPED, VERIFIED components, and after the plastic phase
closes the developed artifact is proven exactly:

    develop -> freeze -> compile -> product-FSM verifier
    (forall input sequences: C_grown == C_spec)

Substrate: a fabric of N_SLOTS typed slots between a fixed SENSOR and a
fixed 8-wide OUTPUT. Development actions (the typed action space):

    DIFFERENTIATE(slot, kind)     PRUNE(slot, port)
    CONNECT(slot, port, source)   SET_OUTPUT(source)

Connections may reference ANY slot; same-or-later references are routed
through a one-tick register automatically (the fbk idiom), and width
mismatches are adapted (broadcast / bit-select), so EVERY fabric state is
a legal synchronous Morpho circuit — feedback closure is always safe.
Unconnected inputs read 0.

Lifecycle: macro-steps of {run a spontaneous-rehearsal stimulus through
the current fabric, extract per-slot activity features, apply the
developmental rule}; then maturity (freeze) and exact verification.

This module establishes the machinery with HAND developmental rules —
the expressibility-first discipline: echo and hold assemble themselves,
with at least one decision made from rehearsal activity rather than
scripted structure. Rule SEARCH and the three-condition comparison
(direct search vs develop-once vs lifelong plasticity) follow SA0.

Usage (from repo root):
  python3 -m evolve.experiment_sa1dev selftest
  python3 -m evolve.experiment_sa1dev show hold
"""

import argparse

import numpy as np

from .compose import interpret, out_width, pretty, W
from .experiment_c1 import REFS
from .experiment_sa0 import verify_ce

N_SLOTS = 12

# slot kind -> (compose component, params, input port specs 'name:width')
KINDS = {
    'latch':   ('srlatch', (), ('set:1', 'reset:1')),
    'delay':   ('delay', (1,), ('in:1',)),
    'edge':    ('edge', (), ('in:1',)),
    'not':     ('not_', (), ('in:1',)),
    'and':     ('and_', (), ('a:1', 'b:1')),
    'or':      ('or_', (), ('a:1', 'b:1')),
    'xor':     ('xor_', (), ('a:1', 'b:1')),
    'ring_en': ('ring_en', (), ('en:1',)),
    'select6': ('select', (6,), ('in:8',)),
    'repeat8': ('repeat8', (), ('in:1',)),
    'and8':    ('and_', (), ('a:8', 'b:8')),
}


class Fabric:
    """Mutable typed fabric. Sources: ('sensor',), ('zero',), ('slot', j)."""

    def __init__(self):
        self.kind = [None] * N_SLOTS
        self.inputs = [{} for _ in range(N_SLOTS)]   # port -> source
        self.output = ('zero',)

    def differentiate(self, slot, kind):
        self.kind[slot] = kind
        self.inputs[slot] = {}

    def connect(self, slot, port, source):
        self.inputs[slot][port] = source

    def prune(self, slot, port):
        self.inputs[slot].pop(port, None)

    def set_output(self, source):
        self.output = source

    def _build(self):
        prog = [('sensor', (), ()), ('const0', (), ())]
        slot_node, fbk_fix = {}, []
        live = [j for j in range(N_SLOTS) if self.kind[j]]
        # topological build order: registers are inserted only for edges in
        # genuine cycles, never for acyclic wiring that happens to point
        # backwards in slot-index space
        deps = {i: {s[1] for s in self.inputs[i].values()
                    if s[0] == 'slot' and self.kind[s[1]]}
                for i in live}
        order, remaining = [], set(live)
        while remaining:
            ready = [i for i in remaining if not (deps[i] & remaining)]
            nxt = min(ready) if ready else min(remaining)   # cycle break
            order.append(nxt)
            remaining.discard(nxt)
        live = order

        def adapt(n, want):
            have = out_width(prog, n)
            if have == want:
                return n
            if want == W:
                prog.append(('repeat8', (), (n,)))
            else:
                prog.append(('select', (0,), (n,)))
            return len(prog) - 1

        def src_node(source, at_slot):
            if source[0] == 'sensor':
                return 0
            if source[0] == 'zero':
                return 1
            j = source[1]
            if self.kind[j] is None:
                return 1
            if j in slot_node:
                return slot_node[j]
            prog.append(('fbk', (-1,), ()))          # bind after all slots
            fbk_fix.append((len(prog) - 1, j))
            return len(prog) - 1

        for j in live:
            comp, params, ports = KINDS[self.kind[j]]
            ins = []
            for spec in ports:
                pname, wdt = spec.split(':')
                n = src_node(self.inputs[j].get(pname, ('zero',)), j)
                ins.append(adapt(n, int(wdt)))
            prog.append((comp, params, tuple(ins)))
            slot_node[j] = len(prog) - 1
        for i, j in fbk_fix:                          # fbk reads one bit
            n = slot_node.get(j, 1)
            if out_width(prog, n) == W:
                prog.append(('select', (0,), (n,)))
                n = len(prog) - 1
            prog[i] = ('fbk', (n,), ())
        out = adapt(src_node(self.output, None), W)
        prog.append(('or_', (), (out, out)))          # identity, output last
        return prog, slot_node

    def to_prog(self):
        return self._build()[0]


#@MARK: rehearsal + activity features

def rehearsal_stimulus(steps=32):
    x = np.zeros(steps, np.int8)
    x[[3, 12, 24]] = 1                     # spontaneous synthetic pulses
    return x

def activity_features(fab, steps=32):
    """Per-slot features from the CURRENT fabric under rehearsal:
    activity rate and correlation with recent sensor activity."""
    prog, slot_node = fab._build()
    x = rehearsal_stimulus(steps)
    traces = interpret(prog, steps, all_nodes=True, x=x)
    recent = np.convolve(x, np.ones(3), 'same') > 0
    feats = {}
    for j, n in slot_node.items():
        t = traces[n]
        feats[j] = {'activity': float(t.mean()),
                    'sensor_corr': float(t[recent].mean()
                                         - t[~recent].mean())}
    return feats


#@MARK: lifecycle with hand developmental rules

def develop(rule, macro_steps=6):
    fab = Fabric()
    for step in range(macro_steps):
        feats = activity_features(fab) if any(fab.kind) else {}
        if not rule(fab, feats, step):
            break                          # maturity: plastic phase closes
    return fab

def hand_rule_echo(fab, feats, step):
    """Attach a broadcast, then let rehearsal activity choose its source:
    the candidate whose activity correlates with the sensor wins, and the
    silent path is pruned."""
    if step == 0:
        fab.differentiate(0, 'delay')      # a spurious candidate path
        fab.connect(0, 'in', ('zero',))
        fab.differentiate(1, 'repeat8')
        fab.connect(1, 'in', ('slot', 0))
        fab.set_output(('slot', 1))
        return True
    if step == 1:
        best, score = ('zero',), -1.0
        for cand in (('sensor',), ('slot', 0)):
            fab.connect(1, 'in', cand)
            s = activity_features(fab)[1]['sensor_corr']
            if s > score:
                best, score = cand, s
        fab.connect(1, 'in', best)
        fab.prune(0, 'in')
        return True
    return False

def hand_rule_hold(fab, feats, step):
    """Feed-forward first; rehearsal shows output activity dies with the
    pulse, so memory is recruited upstream (activity-triggered
    differentiation)."""
    if step == 0:
        fab.differentiate(0, 'repeat8')
        fab.connect(0, 'in', ('sensor',))
        fab.set_output(('slot', 0))
        return True
    if step == 1:
        if feats[0]['activity'] < 0.5:     # transient, not persistent
            fab.differentiate(1, 'latch')
            fab.connect(1, 'set', ('sensor',))
            fab.connect(0, 'in', ('slot', 1))
        return True
    return False

HAND_RULES = {'echo': hand_rule_echo, 'hold': hand_rule_hold}


def selftest():
    zero_x = np.zeros(8, np.int8)
    fab = Fabric()
    interpret(fab.to_prog(), 8, x=zero_x)             # empty fabric legal
    fab.differentiate(0, 'latch')
    fab.connect(0, 'set', ('sensor',))
    fab.connect(0, 'reset', ('slot', 3))              # forward ref
    interpret(fab.to_prog(), 8, x=zero_x)
    fab.differentiate(3, 'ring_en')                   # 8-wide feedback src
    fab.connect(3, 'en', ('slot', 0))
    interpret(fab.to_prog(), 8, x=zero_x)
    print("1. every fabric state (empty, forward-wired, width-mismatched) "
          "compiles to a legal synchronous program")
    for name in ('echo', 'hold'):
        fab = develop(HAND_RULES[name])
        v = verify_ce(fab.to_prog(), REFS[name])
        assert v.get('exact'), (name, v)
        print(f"2. '{name}' assembled itself in the plastic phase -> "
              f"frozen -> product-exact ({v['states']} states)")
    fab = develop(hand_rule_echo)
    assert fab.inputs[1].get('in') == ('sensor',)
    assert 'in' not in fab.inputs[0]
    print("3. echo's wiring was chosen by rehearsal-activity correlation "
          "(and the silent candidate path was pruned)")
    print("selftest passed")


def show(name):
    fab = develop(HAND_RULES[name])
    print(f"developed fabric for '{name}':")
    for j in range(N_SLOTS):
        if fab.kind[j]:
            print(f"  slot {j}: {fab.kind[j]:<8} <- {fab.inputs[j]}")
    print(f"  output <- {fab.output}\n")
    print(pretty(fab.to_prog()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('show')
    s.add_argument('name', choices=list(HAND_RULES))
    args = ap.parse_args()
    if args.cmd == 'selftest':
        selftest()
    else:
        show(args.name)


if __name__ == '__main__':
    main()
