# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment C3: the solved training corpus (handoff Phase 3).

~20 small, diverse, exactly-verifiable single-sensor reactive tasks from
the handoff's families — delayed echoes, explicit-state memory, toggles,
fixed-duration pulses, timers, alternating outputs, gated waves,
edge-triggered state, refractory logic, counter-controlled patterns.

STRICT RULE: `oneshot` and `threewave` are NOT in this corpus, and no
solution or subgraph derived from them may enter library mining. This
corpus exists so that Phase 4 can mine reusable sub-DAGs from unrelated
tasks and Phase 5 can test held-out transfer onto the untouched hard
targets.

Each solved task stores: task id, program DAG, exactness verdict, found
size, post-hoc reduced size (Phase 2 reducer), CE count, evaluations.
Unsolved tasks record their best near-miss (reachability diagnostics).

Usage (from repo root):
  python3 -m evolve.experiment_c3 selftest
  python3 -m evolve.experiment_c3 run --seeds 2 --workers 6
  python3 -m evolve.experiment_c3 summarize runs/exp_c3_corpus.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from .compose import pretty, W
from .experiment_c1 import REFS as LADDER_REFS
from .experiment_sa0 import cegis, verify_ce
from .reduce import reduce_prog

ONES = np.ones(W, np.int8)
ZERO8 = np.zeros(W, np.int8)
STRIPES = (np.arange(W) % 2 == 0).astype(np.int8)

def _full(b):
    return ONES if b else ZERO8

def _onehot(k):
    v = np.zeros(W, np.int8)
    v[k % W] = 1
    return v

def _moore(states, out_of_r, step_fn):
    return dict(states=states, out=lambda r, b: out_of_r(r),
                step=step_fn)

def _delayed_echo(d):
    mask = (1 << d) - 1
    return dict(states=1 << d,
                out=lambda r, b: _full((r >> (d - 1)) & 1),
                step=lambda r, b: ((r << 1) | b) & mask)

def _pulse_len(n):
    return _moore(n + 1, lambda r: _full(r > 0),
                  lambda r, b: n if b else max(r - 1, 0))

# The corpus. Every task: single sensor in, 8-wide out, tiny exact FSM.
TASKS = {
    'echo': LADDER_REFS['echo'],
    'hold': LADDER_REFS['hold'],
    'delayed_echo1': _delayed_echo(1),
    'delayed_echo2': _delayed_echo(2),
    'delayed_echo3': _delayed_echo(3),
    # hold_off: on until the first pulse, then off forever
    'hold_off': _moore(2, lambda r: _full(1 - r), lambda r, b: r | b),
    # toggle_pulse: each pulse flips the whole output
    'toggle_pulse': _moore(2, lambda r: _full(r), lambda r, b: r ^ b),
    # fixed-duration pulses (retriggerable timers)
    'pulse_len2': _pulse_len(2),
    'pulse_len3': _pulse_len(3),
    'pulse_len4': _pulse_len(4),
    # two_pulse_latch: latches on at the second pulse
    'two_pulse_latch': _moore(3, lambda r: _full(r == 2),
                              lambda r, b: min(2, r + b)),
    # edge_blink: after the first pulse, blink forever
    # state r = armed*2 + phase
    'edge_blink': _moore(4,
                         lambda r: _full((r >> 1) & (r & 1)),
                         lambda r, b: (((r >> 1) | b) << 1)
                         | ((r & 1) ^ ((r >> 1) | b))),
    # wave_while_held: travelling position advances only while sensor high
    'wave_while_held': dict(states=W,
                            out=lambda r, b: _onehot(r) * b,
                            step=lambda r, b: (r + b) % W),
    # wave_free_gated: free-running wave, visible after arming pulse
    'wave_free_gated': dict(
        states=2 * W,
        out=lambda r, b: _onehot(r % W) * (r >= W),
        step=lambda r, b: ((r % W) + 1) % W + W * ((r >= W) | b)),
    # stripes_gate: stripes after arming pulse
    'stripes_gate': _moore(2, lambda r: STRIPES * r, lambda r, b: r | b),
    # alternator: each pulse swaps stripes phase
    'alternator': _moore(2,
                         lambda r: STRIPES if r == 0 else 1 - STRIPES,
                         lambda r, b: r ^ b),
    # delayed_hold2: hold, but onset 2 ticks after the pulse
    'delayed_hold2': _moore(4, lambda r: _full(r == 3),
                            lambda r, b: (1 if b else 0) if r == 0
                            else min(r + 1, 3)),
    # refractory_pulse: emit the pulse unless emitted on the previous tick
    'refractory_pulse': dict(states=2,
                             out=lambda r, b: _full(b & (1 - r)),
                             step=lambda r, b: b),
    # and_gate_hold: on only while (armed AND sensor high)
    'and_gate_hold': dict(states=2,
                          out=lambda r, b: _full(r & b),
                          step=lambda r, b: r | b),
    # pulse_stretch2: pulse tick plus one extra tick
    'pulse_stretch2': dict(states=2,
                           out=lambda r, b: _full(b | r),
                           step=lambda r, b: b),
    # stripes_after_two: stripes once two pulses have been seen
    'stripes_after_two': _moore(3, lambda r: STRIPES * (r == 2),
                                lambda r, b: min(2, r + b)),
}
assert 'oneshot' not in TASKS and 'threewave' not in TASKS


def _run_one(job):
    name, seed = job
    r = cegis(name, seed, ref=TASKS[name])
    rec = {'task': name, 'seed': seed, 'solved': r['solved'],
           'evals': r['evals'], 'rounds': r['rounds'],
           'n_ces': len(r.get('ces', []))}
    if r['solved']:
        reduced = reduce_prog(r['prog'], TASKS[name])
        assert verify_ce(reduced, TASKS[name]).get('exact')
        rec.update({'size': r['size'], 'reduced_size': len(reduced),
                    'prog': r['prog'], 'reduced_prog': reduced,
                    'prog_pretty': pretty(reduced)})
    else:
        rec['best_pretty'] = r.get('best_pretty')
    return rec

def run(seed_n, workers, out):
    open(out, 'w').close()
    jobs = [(t, s) for t in TASKS for s in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{rec['task']:<18} seed {rec['seed']}  "
                  f"solved {rec['solved']}  "
                  f"size {rec.get('size', '-')}->"
                  f"{rec.get('reduced_size', '-')}  "
                  f"evals {rec['evals']}  CEs {rec['n_ces']}", flush=True)
    summarize(out)

def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    tasks = sorted({r['task'] for r in recs})
    solved_tasks = 0
    print(f"\n== C3 corpus: {len(tasks)} tasks x "
          f"{len(recs) // len(tasks)} seeds ==")
    print(f"{'task':<18} {'solved':>7} {'med evals':>10} "
          f"{'found->reduced':>15}")
    for t in tasks:
        rs = [r for r in recs if r['task'] == t]
        sv = [r for r in rs if r['solved']]
        solved_tasks += bool(sv)
        ev = sorted(r['evals'] for r in sv)
        sizes = [(r['size'], r['reduced_size']) for r in sv]
        print(f"{t:<18} {f'{len(sv)}/{len(rs)}':>7} "
              f"{ev[len(ev) // 2] if ev else '-':>10} "
              f"{str(sizes[0]) if sizes else '-':>15}")
    n_sol = sum(1 for r in recs if r['solved'])
    print(f"\n{solved_tasks}/{len(tasks)} tasks solved at least once; "
          f"{n_sol} exact solutions available for Phase 4 mining")


def selftest():
    from .experiment_c1 import ref_trace
    # 1. every reference is well-formed and deterministic on schedules
    x = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], np.int8)
    for name, ref in TASKS.items():
        t1, t2 = ref_trace(ref, x), ref_trace(ref, x)
        assert t1.shape == (len(x), W) and (t1 == t2).all(), name
        r = 0
        for b in x:
            r = ref['step'](r, int(b))
            assert 0 <= r < ref['states'], (name, r)
        assert (ref_trace(ref, np.zeros(6, np.int8))[0] ==
                ref['out'](0, 0)).all()
    print(f"1. all {len(TASKS)} corpus references well-formed, "
          f"deterministic, state-bounded (oneshot/threewave excluded)")
    # 2. the reducer shrinks a bloated exact program and preserves
    # exactness (padding a hand solution with dead + bypassable nodes)
    from .experiment_c1 import HAND
    bloated = HAND['hold'] + [('toggle', (), ()),
                              ('not_', (), (2,)),
                              ('or_', (), (3, 3))]
    assert verify_ce(bloated, TASKS['hold']).get('exact')
    red = reduce_prog(bloated, TASKS['hold'])
    assert verify_ce(red, TASKS['hold']).get('exact')
    assert len(red) < len(bloated)
    print(f"2. reducer: {len(bloated)} -> {len(red)} nodes, still "
          f"product-exact")
    # 3. one cheap end-to-end corpus solve + reduce
    r = cegis('pulse_stretch2', 0, round_budget=6000, max_rounds=8,
              ref=TASKS['pulse_stretch2'])
    assert r['solved'], r
    red = reduce_prog(r['prog'], TASKS['pulse_stretch2'])
    assert verify_ce(red, TASKS['pulse_stretch2']).get('exact')
    print(f"3. CEGIS solves pulse_stretch2 (size {r['size']} -> "
          f"reduced {len(red)}), {len(r['ces'])} CEs, {r['evals']} evals")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('run')
    s.add_argument('--seeds', type=int, default=2)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp_c3_corpus.jsonl')
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    args = ap.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'run':
        run(args.seeds, args.workers, args.out)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
