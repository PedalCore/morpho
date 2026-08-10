# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 3: interface bandwidth as the scientific variable.
Experiment 3B: the emission-masked fitness amendment.

Question (3): how much inter-cell communication bandwidth does a
developmental grammar need to evolve scalable stateful machines?

Task: copy-after-delay (word length k) — the first task whose natural law
needs three signals across every cell boundary (window chain, buffer chain,
cue broadcast), which the frozen P=2 grammars cannot carry.

Arms (identical mutation scheme / population / budget / training widths /
verifier): dev_p2, shared_p2 (frozen P=2 grammars with 2-input plumbing,
control) vs dev_p3, shared_p3 (generic P=3 — three anonymous signals, no
copy-specific primitives). Hand P=3 laws establish sufficiency.

3A result (--fitness whole): all arms 0/8; every run converged exactly to
the constant-zero circuit — whole-stream bit accuracy makes silence an
attracting local optimum (most steps demand output 0, and a partial copier
scores worse than staying quiet).

3B amendment (--fitness emission): TRAINING accuracy is scored only on
steps where the REFERENCE machine is emitting (phase > 0), a mask produced
by the task generator and untouchable by the candidate. Constant-zero then
scores ~0.5. Nothing else changes: same task semantics, targets, genotypes,
mutation, budget, verifier; zero-shot verdicts still use full-protocol
FSM exactness / whole-stream holdout.

Protocol: train ONLY k in {1,2,4} (min first, then mean, then -cost),
freeze, zero-shot k in {3,5,6,7,8,12,16}. FSM verification where the
product is tractable (k <= 8); k = 12, 16 report holdout labels.

Usage (from repo root):
  python3 -m evolve.experiment3 checks       # 3B pre-launch diagnostics
  python3 -m evolve.experiment3 selftest
  python3 -m evolve.experiment3 sweep --fitness emission --seeds 8
  python3 -m evolve.experiment3 summarize runs/exp3_copy_masked.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from tiny_morpho_seq import compile_seq
from .develop_genome import (DEV_SPEC, instantiate_dev, make_dev_grammar,
                             hand_dev_copy, spec_random, spec_mutate,
                             genome_size)
from .shared_genome import make_shared_grammar, hand_shared_copy
from .temporal_tasks import TASKS1, copy_case, score
from .fsm_verify import verify

ARMS = {'dev_p2': make_dev_grammar(2, 2),
        'dev_p3': make_dev_grammar(3, 2),
        'shared_p2': make_shared_grammar(2, 2, 1),
        'shared_p3': make_shared_grammar(3, 2, 2)}
HANDS = {'dev_p3': hand_dev_copy, 'shared_p3': hand_shared_copy}
TASK = 'copy'
TRAIN_KS = (1, 2, 4)
ZEROSHOT_KS = (3, 5, 6, 7, 8, 12, 16)
VERIFY_MAX_K = 8
FITNESS_MODES = ('whole', 'emission')


def _step_n(k):
    return max(40, 5 * k + 20)

def _acc(grammar, g, k, rng, case_n, long=False):
    """Returns (emission-masked acc, whole-stream acc, metrics) on the SAME
    streams. Both masks come from the task generator alone."""
    n = case_n * (4 if long else 1)
    step_n = _step_n(k) * (2 if long else 1)
    x, target, emit = copy_case(rng, k, n, step_n, emission_mask=True)
    sim = compile_seq(grammar['instantiate'](g, k), (grammar['x_n'],))
    y = sim.run(x.shape[1], x, samples=x.shape[2])
    m = sim.metrics()
    return (score(y[0], target, emit),
            score(y[0], target, np.arange(step_n) >= k), m)

def eval_multi(grammar, g, rng, fitness, case_n=64):
    ems, whs, cost, m = [], [], 0.0, None
    for k in TRAIN_KS:
        em, wh, m = _acc(grammar, g, k, rng, case_n)
        ems.append(em)
        whs.append(wh)
        cost += 3 * m['registers'] + m['gates'] + 0.2 * m['edges']
    sel = ems if fitness == 'emission' else whs
    fit = (min(sel), float(np.mean(sel)), -cost)
    aux = {'min_emission': min(ems), 'mean_emission': float(np.mean(ems)),
           'min_whole': min(whs), 'mean_whole': float(np.mean(whs)),
           'cost': cost, 'registers': m['registers'], 'gates': m['gates'],
           'depth': m['logic_depth']}
    return fit, aux

def _zeroshot(grammar, g, rng):
    per_k = {}
    for k in TRAIN_KS + ZEROSHOT_KS:
        em, wh, m = _acc(grammar, g, k, rng, 128, long=True)
        exact = None
        if wh == 1.0 and k <= VERIFY_MAX_K:
            v = verify(compile_seq(grammar['instantiate'](g, k),
                                   (grammar['x_n'],)),
                       TASKS1[TASK][1](k), warmup=k, max_states=1 << 21)
            exact = None if v['aborted'] else v['exact']
        per_k[k] = {'acc': wh, 'emission_acc': em, 'exact': exact,
                    'registers': m['registers'], 'gates': m['gates'],
                    'depth': m['logic_depth']}
    return per_k

def _solved_k(p):
    """Exact where verifiable; whole-stream-perfect holdout where not."""
    return p['exact'] if p['exact'] is not None else p['acc'] == 1.0


def train_run(arm, seed, fitness, pop_n=64, gen_max=800, case_n=64,
              elite_n=4, tourney_n=3, patience=150):
    grammar = ARMS[arm]
    rng = np.random.default_rng(seed)
    pop = [spec_random(rng, grammar['spec']) for _ in range(pop_n)]
    best_fit, stagnant, evals, solved_at, history = None, 0, 0, None, []
    for gen in range(gen_max):
        scored = []
        for g in pop:
            fit, aux = eval_multi(grammar, g, rng, fitness, case_n)
            evals += 1
            scored.append((fit, aux, g))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [g for _, _, g in scored]
        top, top_aux = scored[0][0], scored[0][1]
        first_solve = solved_at is None and top[0] == 1.0
        if first_solve:
            solved_at = evals
        if gen % 10 == 0 or first_solve or gen == gen_max - 1:
            history.append({'gen': gen, 'evals': evals, **top_aux})
        stagnant = stagnant + 1 if best_fit and top <= best_fit else 0
        best_fit = max(best_fit, top) if best_fit else top
        if top[0] == 1.0 and stagnant >= patience:
            break
        children = []
        while len(children) < pop_n - elite_n:
            winner = ranked[rng.integers(pop_n, size=tourney_n).min()]
            children.append(spec_mutate(winner, rng, grammar['spec']))
        pop = ranked[:elite_n] + children
    return _finish(ranked[0], arm, arm, seed, evals, solved_at, rng,
                   fitness, history)

def _finish(g, arm, label, seed, evals, solved_at, rng, fitness, history):
    grammar = ARMS[arm]
    return {'arm': label, 'task': TASK, 'fitness': fitness, 'seed': seed,
            'evals': evals, 'evals_to_solve': solved_at,
            'genome_size': genome_size(grammar['spec']),
            'per_k': _zeroshot(grammar, g, rng), 'history': history,
            'genome': {k: v.tolist() for k, v in g.items()}}

def hand_run(arm, fitness):
    return _finish(HANDS[arm](ARMS[arm]), arm, f'hand_{arm}', 0, 0, 0,
                   np.random.default_rng(777), fitness, [])


def _run_one(job):
    return train_run(*job)

def sweep(seed_n, workers, out, fitness):
    open(out, 'w').close()
    with open(out, 'a') as f:
        for arm in HANDS:
            f.write(json.dumps(hand_run(arm, fitness)) + '\n')
    jobs = [(arm, seed, fitness) for arm in ARMS for seed in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            k16 = rec['per_k'].get(16) or rec['per_k'].get('16')
            print(f"{rec['arm']:<15} seed {rec['seed']}  "
                  f"solved_at {rec['evals_to_solve']}  "
                  f"k=16 whole {k16['acc']:.4f} "
                  f"emission {k16['emission_acc']:.4f}", flush=True)
    summarize(out)


def _load(path):
    recs = [json.loads(line) for line in open(path)]
    for r in recs:
        r['per_k'] = {int(k): v for k, v in r['per_k'].items()}
    return recs

def summarize(path):
    recs = _load(path)
    all_ks = TRAIN_KS + ZEROSHOT_KS
    print(f"\n== copy-after-delay, fitness={recs[-1].get('fitness', 'whole')}"
          f" (train k=1,2,4; freeze; zero-shot) ==")
    print(f"{'arm':<16} {'|g|':>4} {'solved':>7} {'transfer':>9} "
          f"{'exact widths':>14} {'holdout widths':>15} {'med evals':>10} "
          f"{'regs':>5} {'gates':>6} {'depth':>6}")
    for arm in ('hand_dev_p3', 'hand_shared_p3',
                'dev_p2', 'shared_p2', 'dev_p3', 'shared_p3'):
        rs = [r for r in recs if r['arm'] == arm]
        if not rs:
            continue
        solved = [r for r in rs if all(_solved_k(r['per_k'][k])
                                       for k in TRAIN_KS)]
        transfer = [r for r in solved if all(_solved_k(r['per_k'][k])
                                             for k in ZEROSHOT_KS)]
        ev = [r['evals_to_solve'] for r in solved
              if r['evals_to_solve'] is not None] or None
        best = min(transfer, key=lambda r: r['per_k'][16]['gates']) \
            if transfer else None
        exact_w = [k for k in all_ks if best and best['per_k'][k]['exact']]
        hold_w = [k for k in all_ks if best and not best['per_k'][k]['exact']
                  and best['per_k'][k]['acc'] == 1.0]
        p16 = best and best['per_k'][16]
        print(f"{arm:<16} {rs[0]['genome_size']:>4} "
              f"{f'{len(solved)}/{len(rs)}':>7} "
              f"{f'{len(transfer)}/{len(solved)}' if solved else '-':>9} "
              f"{str(exact_w) if best else '-':>14} "
              f"{str(hold_w) if best else '-':>15} "
              f"{int(np.median(ev)) if ev else '-':>10} "
              f"{p16['registers'] if best else '-':>5} "
              f"{p16['gates'] if best else '-':>6} "
              f"{p16['depth'] if best else '-':>6}")


def checks():
    """3B pre-launch diagnostics (required before the masked sweep)."""
    rng = np.random.default_rng(11)
    # 3. mask selects only reference emission steps; targets identical
    for w in (1, 2, 4):
        r1, r2 = np.random.default_rng(w), np.random.default_rng(w)
        x1, t1, m1 = copy_case(r1, w, 500, 60, emission_mask=True)
        x2, t2, m2 = copy_case(r2, w, 500, 60, emission_mask=False)
        assert (x1 == x2).all() and (t1 == t2).all()
        assert m1.shape == t1.shape and m1.dtype == bool
        assert (t1[~m1] == 0).all(), "nonzero target outside emission phase"
        frac_ones = t1[m1].mean()
        print(f"W={w}: mask is reference-derived, targets unchanged, "
              f"P(target=1 | emitting) = {frac_ones:.3f}")
    # 1. constant-zero scores ~0.5 emission-masked (vs ~0.8 whole-stream)
    g0 = spec_random(np.random.default_rng(0), ARMS['dev_p3']['spec'])
    g0['t_out'][0] = 0                       # output = const 0
    em, wh, _ = _acc(ARMS['dev_p3'], g0, 4, rng, 500)
    assert abs(em - 0.5) < 0.05 and wh > 0.7
    print(f"constant-zero circuit: emission acc {em:.3f} (~chance), "
          f"whole-stream acc {wh:.3f} (the old attractor)")
    # 2. hand P=3 laws are emission-perfect and whole-perfect
    for arm in HANDS:
        g = HANDS[arm](ARMS[arm])
        for k in (1, 2, 4, 8):
            em, wh, _ = _acc(ARMS[arm], g, k, rng, 128)
            assert em == 1.0 and wh == 1.0, (arm, k, em, wh)
        print(f"hand {arm}: emission and whole-stream both 1.0 at k=1,2,4,8")
    # 4. the mask cannot depend on the candidate: identical rng state gives
    # an identical mask before any circuit is built or run
    r1, r2 = np.random.default_rng(99), np.random.default_rng(99)
    _, _, ma = copy_case(r1, 4, 200, 60, emission_mask=True)
    _, _, mb = copy_case(r2, 4, 200, 60, emission_mask=True)
    assert (ma == mb).all()
    print("mask is a pure function of the task stream (candidate-independent)")
    # 5. whole-stream metric remains available for reporting in every record
    print("whole-stream accuracy is logged in aux/per_k of every record; "
          "selection uses it only when --fitness whole")
    print("all 3B pre-launch checks passed")


def selftest():
    gen = make_dev_grammar(2, 1)
    assert len(gen['spec']) == len(DEV_SPEC)
    for (k1, s1, b1), (k2, s2, b2) in zip(gen['spec'], DEV_SPEC):
        assert k1 == k2 and s1 == s2
        assert (isinstance(b1, str) and b1 == b2) or (np.asarray(b1) == b2).all()
    rng = np.random.default_rng(3)
    x = rng.integers(2, size=(1, 40, 16)).astype(np.int32)
    for trial in range(5):
        g = spec_random(rng, DEV_SPEC)
        for k in (1, 3, 4):
            y1 = compile_seq(instantiate_dev(g, k), (1,)).run(40, x, samples=16)
            y2 = compile_seq(gen['instantiate'](g, k), (1,)).run(40, x, samples=16)
            assert (np.asarray(y1) == np.asarray(y2)).all()
    print("generic builder reproduces the frozen P=2 grammar "
          "(spec + behavior on 5 random genomes)")
    for arm in HANDS:
        rec = hand_run(arm, 'whole')
        for k in TRAIN_KS + ZEROSHOT_KS:
            assert _solved_k(rec['per_k'][k]), (arm, k, rec['per_k'][k])
        exact_ks = [k for k in TRAIN_KS + ZEROSHOT_KS
                    if rec['per_k'][k]['exact']]
        regs = [rec['per_k'][k]['registers'] for k in (1, 2, 4, 8, 16)]
        print(f"hand {arm}: |genome|={rec['genome_size']}, FSM-exact at "
              f"k={exact_ks}, holdout-perfect elsewhere, "
              f"registers(1,2,4,8,16)={regs}")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('checks')
    s = sub.add_parser('sweep')
    s.add_argument('--fitness', choices=FITNESS_MODES, default='whole')
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default=None)
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'checks':
        checks()
    elif args.cmd == 'sweep':
        out = args.out or ('runs/exp3_copy_masked.jsonl'
                           if args.fitness == 'emission'
                           else 'runs/exp3_copy.jsonl')
        sweep(args.seeds, args.workers, out, args.fitness)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
