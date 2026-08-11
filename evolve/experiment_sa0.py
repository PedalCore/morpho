# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment SA0: centrally-guided self-assembly with counterexample-
guided synthesis (CEGIS).

The shift: search over a GROWTH process, not finished netlists, and stop
using average behavioral error as the only guide. C1/C2-S showed why —
under trace-averaged scoring, a degenerate 'first-frame mimic' outranks
every honest trigger+wave intermediate, and both deterministic and
stochastic beams anchor in that basin.

Here the exact product-FSM verifier joins the loop:

    grow candidate (stochastic best-first over typed partial assemblies)
        -> passes all accumulated counterexamples?
        -> exact product verification
        -> if wrong: SHORTEST counterexample input sequence (BFS depth)
        -> add to the active specification set, next round
        -> if right: solved, for all possible input sequences

The mimic survives round 0 and is immediately counterexampled: the
verifier returns the exact pulse timing at which the missing wave
matters, and silence stops hiding behind idle zeros.

Growth actions are realized by the existing typed grammar: ATTACH+CONNECT
= adding a type-valid node; CLOSE_FEEDBACK = binding an fbk register (the
open feedback socket); STOP = candidate completion. Richer socket
semantics and shared local growth rules are SA1.

Targets: the C1 reactive ladder, with oneshot and threewave — unsolved by
deterministic (C1) and stochastic (C2-S) trace-averaged beams — as the
tiers under test.

Usage (from repo root):
  python3 -m evolve.experiment_sa0 selftest
  python3 -m evolve.experiment_sa0 run --seeds 3 --workers 6
  python3 -m evolve.experiment_sa0 summarize runs/exp_sa0.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from .compose import (COMPONENTS, out_width, interpret, _init_states,
                      _tick, state_key, morpho_trace, pretty, W)
from .experiment_c1 import REFS, HAND, expansions, C1_LIBRARY, ref_trace

ROUND_BUDGET = 25_000
MAX_ROUNDS = 14
MAX_SIZE = 16
TEMP = 0.08
CE_PAD = 12


#@MARK: exact verification with shortest-counterexample extraction

def verify_ce(prog, ref, max_states=30_000):
    init = _init_states(prog)
    start = (state_key(init), 0)
    store, parent = {start: init}, {start: None}
    seen, frontier = {start}, [start]
    while frontier:
        nxt = []
        for key in frontier:
            states, r = store[key], key[1]
            for b in (0, 1):
                outs, ns = _tick(prog, states, b)
                if (outs[-1] != ref['out'](r, b)).any():
                    path, k = [b], key
                    while parent[k] is not None:
                        k, pb = parent[k]
                        path.append(pb)
                    return {'exact': False, 'ce': path[::-1]}
                child = (state_key(ns), ref['step'](r, b))
                if child not in seen:
                    seen.add(child)
                    store[child] = ns
                    parent[child] = (key, b)
                    nxt.append(child)
        if len(seen) > max_states:
            return {'exact': False, 'aborted': True}
        frontier = nxt
    return {'exact': True, 'states': len(seen)}


#@MARK: scoring against the accumulated counterexample set

def ce_schedule(ce):
    return np.array(ce + [0] * CE_PAD, np.int8)

def score_on(prog, ref, tests):
    """(passed_count, mean error on failing tests). Partial programs
    (no 8-wide output) pass nothing."""
    if out_width(prog, len(prog) - 1) != W:
        return 0, 1.0
    passed, fails = 0, []
    for x in tests:
        tr = interpret(prog, len(x), x=x)
        e = float((tr != ref_trace(ref, x)).mean())
        if e == 0.0:
            passed += 1
        else:
            fails.append(e)
    return passed, (float(np.mean(fails)) if fails else 0.0)

def energy(passed, fail_err, size, n_tests):
    frac = passed / n_tests if n_tests else 0.0
    return (1.0 - frac) + 0.3 * fail_err + 0.004 * size


#@MARK: stochastic best-first growth (one CEGIS round)

def _signature(prog, tests):
    xs = (np.concatenate(tests) if tests else np.zeros(16, np.int8))
    traces = interpret(prog, len(xs), all_nodes=True, x=xs)
    return frozenset(
        (t.shape[1], comp == 'fbk' and params[0] < 0, t.tobytes())
        for t, (comp, params, _) in zip(traces, prog))

def grow_round(ref, tests, rng, budget=ROUND_BUDGET, max_size=MAX_SIZE):
    """Returns (candidate passing all tests | None, evals, best_record)."""
    pool, seen, evals = [], set(), 0
    best = (1e9, None)

    def admit(prog):
        nonlocal evals, best
        sig = _signature(prog, tests)
        if sig in seen:
            return None
        seen.add(sig)
        passed, ferr = score_on(prog, ref, tests)
        evals += 1
        e = energy(passed, ferr, len(prog), len(tests))
        if e < best[0]:
            best = (e, prog)
        if passed == len(tests) and out_width(prog, len(prog) - 1) == W:
            return prog
        pool.append((e, prog))
        return None

    for comp in C1_LIBRARY:
        if not COMPONENTS[comp]['ins']:
            for params in COMPONENTS[comp]['params']:
                done = admit([(comp, params, ())])
                if done is not None:
                    return done, evals, best
    while evals < budget and pool:
        # epsilon-mixed sampling: partial assemblies (open builds without a
        # W-wide output yet) score worst under the energy, so a pure
        # softmax starves them; 30% of draws come from the partial subset
        idx = np.arange(len(pool))
        partial = [k for k in idx
                   if out_width(pool[k][1], len(pool[k][1]) - 1) != W]
        if partial and rng.random() < 0.3:
            sub = np.array(partial)
        else:
            sub = idx
        es = np.array([pool[k][0] for k in sub])
        p = np.exp(-(es - es.min()) / TEMP)
        p /= p.sum()
        i = int(sub[rng.choice(len(sub), p=p)])
        _, prog = pool.pop(i)
        if len(prog) >= max_size:
            continue
        for child in expansions(prog):
            done = admit(child)
            if done is not None:
                return done, evals, best
            if evals >= budget:
                break
    return None, evals, best


#@MARK: CEGIS driver

def cegis(name, seed, round_budget=ROUND_BUDGET, max_rounds=MAX_ROUNDS,
          quiet=True):
    ref = REFS[name]
    rng = np.random.default_rng(seed)
    tests, ces, total = [], [], 0
    last_best = (None, None)
    for rnd in range(max_rounds):
        cand, evals, best = grow_round(ref, tests, rng, round_budget)
        total += evals
        last_best = best
        if cand is None:
            continue          # budget spent, spec unchanged — retry round
        v = verify_ce(cand, ref)
        if v.get('exact'):
            for ce in ces[-2:] or [[1] + [0] * 15]:   # Morpho bit-exactness
                x = ce_schedule(ce)
                a = interpret(cand, len(x), x=x)
                b = morpho_trace(cand, len(x), x=x)
                assert a.shape == b.shape and (a == b).all()
            return {'tier': name, 'seed': seed, 'solved': True,
                    'size': len(cand), 'rounds': rnd + 1, 'evals': total,
                    'n_ces': len(ces), 'ces': ces,
                    'product_states': v['states'],
                    'prog_pretty': pretty(cand), 'prog': cand}
        if v.get('aborted'):
            return {'tier': name, 'seed': seed, 'solved': False,
                    'rounds': rnd + 1, 'evals': total, 'ces': ces,
                    'reason': 'verifier aborted'}
        tests.append(ce_schedule(v['ce']))
        ces.append(v['ce'])
        if not quiet:
            print(f"[{name} s{seed}] round {rnd}: CE #{len(ces)} "
                  f"{''.join(map(str, v['ce']))} (evals {total})",
                  flush=True)
    return {'tier': name, 'seed': seed, 'solved': False,
            'rounds': max_rounds, 'evals': total, 'ces': ces,
            'reason': 'round limit', 'best_energy': last_best[0],
            'best_pretty': pretty(last_best[1]) if last_best[1] else None}


def _run_one(job):
    name, seed = job
    return cegis(name, seed)

def run(seed_n, workers, out):
    open(out, 'w').close()
    jobs = [(t, s) for t in ('echo', 'hold', 'oneshot', 'threewave')
            for s in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{rec['tier']:<10} seed {rec['seed']}  "
                  f"solved {rec['solved']}  size {rec.get('size', '-')}  "
                  f"rounds {rec['rounds']}  evals {rec['evals']}  "
                  f"CEs {len(rec.get('ces', []))}", flush=True)
    summarize(out)

def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    print(f"\n== SA0: CEGIS growth vs the C1 ladder "
          f"(round budget {ROUND_BUDGET}, max {MAX_ROUNDS} rounds) ==")
    print(f"{'tier':<10} {'solved':>7} {'med evals':>10} {'med CEs':>8} "
          f"{'sizes':>12}")
    for tier in ('echo', 'hold', 'oneshot', 'threewave'):
        rs = [r for r in recs if r['tier'] == tier]
        if not rs:
            continue
        sv = [r for r in rs if r['solved']]
        ev = sorted(r['evals'] for r in sv)
        ces = sorted(len(r['ces']) for r in sv)
        sizes = sorted(r['size'] for r in sv)
        print(f"{tier:<10} {f'{len(sv)}/{len(rs)}':>7} "
              f"{ev[len(ev) // 2] if ev else '-':>10} "
              f"{ces[len(ces) // 2] if ces else '-':>8} "
              f"{str(sizes) if sizes else '-':>12}")
    for tier in ('oneshot', 'threewave'):
        w = next((r for r in recs if r['tier'] == tier and r['solved']),
                 None)
        if w:
            print(f"\n[{tier}] solver (seed {w['seed']}, size {w['size']}, "
                  f"{w['n_ces']} counterexamples, product states "
                  f"{w['product_states']}):")
            print(w['prog_pretty'])
            print("counterexamples that built the spec:")
            for ce in w['ces']:
                print(f"  {''.join(map(str, ce))}")


def selftest():
    # 1. CE extraction: hand oneshot vs threewave ref yields a shortest CE
    v = verify_ce(HAND['oneshot'], REFS['threewave'])
    assert not v['exact'] and 'ce' in v and len(v['ce']) >= 1
    print(f"1. shortest counterexample extracted: "
          f"{''.join(map(str, v['ce']))} (len {len(v['ce'])})")
    # 2. correct machines verify exactly
    for name in ('echo', 'hold', 'oneshot', 'threewave'):
        assert verify_ce(HAND[name], REFS[name])['exact'], name
    print("2. all hand programs product-exact under the CE verifier")
    # 3. the hand programs satisfy any CE-derived schedule they generate
    ref = REFS['oneshot']
    ce = verify_ce([('const0', (), ()), ('repeat8', (), (0,))], ref)['ce']
    x = ce_schedule(ce)
    tr = interpret(HAND['oneshot'], len(x), x=x)
    assert (tr == ref_trace(ref, x)).all()
    print(f"3. CE schedules integrate with trace scoring "
          f"(silent-mimic CE: {''.join(map(str, ce))})")
    # 4. one cheap end-to-end CEGIS solve
    r = cegis('echo', seed=0, round_budget=4000, max_rounds=6)
    assert r['solved'], r
    print(f"4. CEGIS solves echo end-to-end: size {r['size']}, "
          f"{r['rounds']} rounds, {len(r['ces'])} CEs, "
          f"{r['evals']} evals")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('run')
    s.add_argument('--seeds', type=int, default=3)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp_sa0.jsonl')
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
