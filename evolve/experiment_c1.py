# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment C1: reactive compositional synthesis.

C0's systems were autonomous — generators entering cycles. C1 adds one
typed external input and asks search to construct real reactive state
machines. Success is not 'works on these sample pulses' but product-FSM
equivalence with a tiny reference machine for ALL possible input
sequences:

    forall x0, x1, ... : M_synth(x) == M_ref(x)

Library additions are generic only (sensor, srlatch, edge, enable/clear
gating, and 'fbk' — the language's own REG+forward-reference idiom). No
pulse_counter, no three_wave_controller.

Tier ladder (references defined as FSMs; hand programs in-grammar prove
expressibility and are themselves product-verified):

  echo       out = sensor on all 8 segments           (hand size 2)
  hold       first pulse turns the body on forever    (hand size 4)
  oneshot    idle; on pulse, emit ONE travelling wave
             (8 ticks), return to idle, re-triggerable (hand size 7+bind)
  threewave  idle; on pulse, exactly THREE waves, idle,
             re-triggerable                          (hand size 15+bind)

The three-wave tier is expected to exceed the plain beam's reach — that
is the motivating held-out case for library learning (C3/C4): the
counter/latch/done clusters that oneshot discovers should become the
vocabulary that makes threewave reachable.

Verification chain: product BFS runs on the interpreter machine, whose
bit-exactness against compiled Morpho is asserted per-finalist on
sampled schedules (and per-component in the C0/C1 selftests).

Usage (from repo root):
  python3 -m evolve.experiment_c1 selftest
  python3 -m evolve.experiment_c1 run
"""

import argparse
import itertools

import numpy as np

from .compose import (W, COMPONENTS, out_width, type_ok, interpret,
                      _init_states, _tick, state_key, morpho_trace,
                      n_sensors, pretty)
from .experiment_c0 import waveform

BEAM_FULL = 260
BEAM_PARTIAL = 160
C1_LIBRARY = ('const0', 'const1', 'toggle', 'ring', 'counter', 'not_',
              'xor_', 'or_', 'and_', 'mux', 'rotate', 'reverse', 'delay',
              'select', 'sel_ctr', 'repeat8',
              'sensor', 'fbk', 'srlatch', 'edge', 'ring_en', 'counter_enr')

def _onehot(k):
    v = np.zeros(W, np.int8)
    v[k] = 1
    return v

ZERO8 = np.zeros(W, np.int8)

# reference FSMs: state int; out(r, b) -> 8-bit array; step(r, b) -> r'
REFS = {
    'echo': dict(states=1,
                 out=lambda r, b: np.full(W, b, np.int8),
                 step=lambda r, b: 0),
    'hold': dict(states=2,
                 out=lambda r, b: np.full(W, r, np.int8),
                 step=lambda r, b: 1 if (r or b) else 0),
    'oneshot': dict(states=9,
                    out=lambda r, b: _onehot(r - 1) if r else ZERO8,
                    step=lambda r, b: (1 if b else 0) if r == 0
                                      else (r + 1 if r < 8 else 0)),
    'threewave': dict(states=25,
                      out=lambda r, b: _onehot((r - 1) % 8) if r else ZERO8,
                      step=lambda r, b: (1 if b else 0) if r == 0
                                        else (r + 1 if r < 24 else 0)),
}

HAND = {
    'echo': [('sensor', (), ()), ('repeat8', (), (0,))],
    'hold': [('sensor', (), ()), ('const0', (), ()),
             ('srlatch', (), (0, 1)), ('repeat8', (), (2,))],
    'oneshot': [('sensor', (), ()), ('fbk', (4,), ()),
                ('srlatch', (), (0, 1)), ('ring_en', (), (2,)),
                ('select', (6,), (3,)), ('repeat8', (), (2,)),
                ('and_', (), (3, 5))],
    'threewave': [('sensor', (), ()), ('fbk', (12,), ()),
                  ('srlatch', (), (0, 1)), ('ring_en', (), (2,)),
                  ('select', (7,), (3,)), ('not_', (), (2,)),
                  ('counter_enr', (), (4, 5)), ('sel_ctr', (0,), (6,)),
                  ('sel_ctr', (1,), (6,)), ('not_', (), (7,)),
                  ('and_', (), (8, 9)), ('select', (6,), (3,)),
                  ('and_', (), (11, 10)), ('repeat8', (), (2,)),
                  ('and_', (), (3, 13))],
}

# sampled input schedules (T=44) for search scoring — deterministic
def _schedules():
    scheds = []
    for pulses in ([3], [3, 30], [3, 7], [], [0], [3, 14, 40], [11, 12]):
        x = np.zeros(44, np.int8)
        x[pulses] = 1
        scheds.append(x)
    return scheds
SCHEDULES = _schedules()

def ref_trace(ref, x):
    r, out = 0, []
    for b in x:
        out.append(ref['out'](r, int(b)))
        r = ref['step'](r, int(b))
    return np.stack(out)


#@MARK: scoring and exact product verification

def score(prog, ref):
    if out_width(prog, len(prog) - 1) != W:
        return 1.0
    errs = []
    for x in SCHEDULES:
        tr = interpret(prog, len(x), x=x)
        errs.append(float((tr != ref_trace(ref, x)).mean()))
    return float(np.mean(errs))

def verify_product(prog, ref, max_states=20000):
    """BFS the product (interpreter state x reference state) over both
    input values; outputs must agree at EVERY reachable transition. A True
    verdict is equivalence for all possible input sequences."""
    init = _init_states(prog)
    start = (state_key(init), 0)
    store = {start: init}
    seen, frontier = {start}, [start]
    while frontier:
        nxt = []
        for key in frontier:
            states, r = store[key], key[1]
            for b in (0, 1):
                outs, ns = _tick(prog, states, b)
                if (outs[-1] != ref['out'](r, b)).any():
                    return {'exact': False, 'ref_state': r, 'input': b}
                child = (state_key(ns), ref['step'](r, b))
                if child not in seen:
                    seen.add(child)
                    store[child] = ns
                    nxt.append(child)
        if len(seen) > max_states:
            return {'exact': False, 'aborted': True}
        frontier = nxt
    return {'exact': True, 'states': len(seen)}

def verify_finalist(prog, ref):
    v = verify_product(prog, ref)
    if not v.get('exact'):
        return v
    for x in SCHEDULES[:3]:            # interpreter == compiled Morpho
        a = interpret(prog, len(x), x=x)
        b = morpho_trace(prog, len(x), x=x)
        if a.shape != b.shape or (a != b).any():
            return {'exact': False, 'reason': 'morpho mismatch'}
    return v


#@MARK: search (typed beam over the reactive library, incl. fbk binding)

def expansions(prog):
    n = len(prog)
    for comp in C1_LIBRARY:
        c = COMPONENTS[comp]
        arity = len(c['ins'])
        for ins in itertools.product(range(n), repeat=arity):
            if arity and not type_ok(prog, comp, ins):
                continue
            for params in c['params']:
                yield prog + [(comp, params, ins)]
    for k, (comp, params, ins) in enumerate(prog):   # bind an unbound fbk
        if comp == 'fbk' and params[0] < 0:
            for j in range(n):
                if j != k and out_width(prog, j) == 1:
                    yield (prog[:k] + [('fbk', (j,), ())] + prog[k + 1:])

def signature(prog):
    xs = np.concatenate(SCHEDULES[:4])
    traces = interpret(prog, len(xs), all_nodes=True, x=xs)
    return frozenset(
        (t.shape[1], comp == 'fbk' and params[0] < 0, t.tobytes())
        for t, (comp, params, _) in zip(traces, prog))

def synthesize(name, max_size=8, quiet=False):
    ref = REFS[name]
    gens = [[(comp, params, ())]
            for comp in C1_LIBRARY if not COMPONENTS[comp]['ins']
            for params in COMPONENTS[comp]['params']]
    level, seen, evals = gens, set(), 0
    best_err, best_prog = 1.0, None       # near-miss retained for analysis
    for size in range(1, max_size + 1):
        scored = []
        for prog in level:
            sig = signature(prog)
            if sig in seen:
                continue
            seen.add(sig)
            s = score(prog, ref)
            evals += 1
            scored.append((s, prog))
            if s < best_err:
                best_err, best_prog = s, prog
            if s == 0.0:
                v = verify_finalist(prog, ref)
                if v.get('exact'):
                    if not quiet:
                        print(f"[{name}] solved at size {len(prog)} after "
                              f"{evals} evaluations (product states "
                              f"{v['states']})")
                    return {'name': name, 'solved': True,
                            'size': len(prog), 'evals': evals,
                            'prog': prog, 'verify': v}
        scored.sort(key=lambda t: (t[0], len(t[1]), pretty(t[1])))
        full = [p for s, p in scored if s < 1.0][:BEAM_FULL]
        part = [p for s, p in scored if s >= 1.0][:BEAM_PARTIAL]
        if not quiet:
            print(f"[{name}] size {size}: {len(scored)} novel, best err "
                  f"{scored[0][0]:.3f}" if scored else f"[{name}] size "
                  f"{size}: exhausted", flush=True)
        level = [q for p in full + part for q in expansions(p)]
    return {'name': name, 'solved': False, 'evals': evals,
            'best_err': best_err, 'best_prog': best_prog}


def run_all():
    results = []
    for name, max_size in (('echo', 4), ('hold', 5), ('oneshot', 8),
                           ('threewave', 8)):
        r = synthesize(name, max_size)
        results.append(r)
        if r['solved']:
            print(pretty(r['prog']), '\n')
        elif r.get('best_prog'):
            print(f"best near-miss (err {r['best_err']:.4f}):")
            print(pretty(r['best_prog']), '\n')
    print(f"{'tier':<10} {'hand size':>10} {'found':>6} {'evals':>8} "
          f"{'best unsolved err':>18} {'product-exact':>14}")
    for r in results:
        err = '—' if r['solved'] else f"{r.get('best_err', 1.0):.4f}"
        print(f"{r['name']:<10} {len(HAND[r['name']]):>10} "
              f"{r.get('size', '-'):>6} {r['evals']:>8} {err:>18} "
              f"{str(r.get('verify', {}).get('exact', False)):>14}")


def selftest():
    # 1. reactive components: interpreter == compiled Morpho on schedules
    probes = [HAND['echo'], HAND['hold'], HAND['oneshot'],
              HAND['threewave'],
              [('sensor', (), ()), ('edge', (), (0,)),
               ('const1', (), ()), ('counter_enr', (), (1, 2)),
               ('sel_ctr', (0,), (3,)), ('repeat8', (), (4,))]]
    for prog in probes:
        for x in SCHEDULES[:3]:
            a = interpret(prog, len(x), x=x)
            b = morpho_trace(prog, len(x), x=x)
            assert a.shape == b.shape and (a == b).all(), pretty(prog)
    print("1. reactive interpreter == compiled Morpho, bit-exact "
          f"({len(probes)} programs x 3 schedules)")
    # 2. hand programs are product-FSM-exact against their references —
    # equivalence for ALL input sequences, and expressibility established
    for name in HAND:
        v = verify_finalist(HAND[name], REFS[name])
        assert v.get('exact'), (name, v)
        print(f"2. hand {name}: product-exact over {v['states']} states "
              f"(size {len(HAND[name])})")
    # 3. wrong pairings are rejected
    assert not verify_product(HAND['oneshot'], REFS['threewave'])['exact']
    assert not verify_product(HAND['echo'], REFS['hold'])['exact']
    print("3. verifier rejects mismatched machine/reference pairs")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('run')
    s = sub.add_parser('show')
    s.add_argument('name', choices=list(HAND))
    args = ap.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'run':
        run_all()
    else:
        x = np.zeros(30, np.int8)
        x[3] = 1
        print(pretty(HAND[args.name]), '\n')
        print(waveform(interpret(HAND[args.name], 30, x=x)))


if __name__ == '__main__':
    main()
