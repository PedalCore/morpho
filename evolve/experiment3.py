# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 3: interface bandwidth as the scientific variable.

Question: how much inter-cell communication bandwidth does a developmental
grammar need to evolve scalable stateful machines?

Task: copy-after-delay (word length k) — the first task whose natural law
needs three signals across every cell boundary (window chain, buffer chain,
cue broadcast), which the frozen P=2 grammars cannot carry.

Preregistered arms, identical mutation scheme / population / budget /
training widths / verifier:

  dev_p2, shared_p2   frozen P=2 grammars with 2-input plumbing (control)
  dev_p3, shared_p3   generic P=3 grammars — three anonymous signals,
                      no cue/buffer/copy-specific primitives

Protocol as Experiment 2: train ONLY k in {1,2,4} (min-accuracy first),
freeze, zero-shot k in {3,5,6,7,8,12,16}. Product-FSM verification where
the product is tractable (k <= 8 for this task's reference: 2^(2k) x (k+1)
reference states); k = 12, 16 report perfect-holdout accuracy instead.
Hand P=3 laws establish sufficiency before evolution runs.

Usage (from repo root):
  python3 -m evolve.experiment3 selftest
  python3 -m evolve.experiment3 sweep --seeds 8 --workers 6
  python3 -m evolve.experiment3 summarize runs/exp3_copy.jsonl
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
from .temporal_tasks import TASKS1, score
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


def _step_n(k):
    return max(40, 5 * k + 20)

def _acc(grammar, g, k, rng, case_n, long=False):
    case_fn, _, x_n = TASKS1[TASK]
    x, target, mask = case_fn(rng, k, case_n * (4 if long else 1),
                              _step_n(k) * (2 if long else 1))
    sim = compile_seq(grammar['instantiate'](g, k), (x_n,))
    y = sim.run(x.shape[1], x, samples=x.shape[2])
    m = sim.metrics()
    return score(y[0], target, mask), m

def eval_multi(grammar, g, rng, case_n=64):
    accs, cost = [], 0.0
    for k in TRAIN_KS:
        acc, m = _acc(grammar, g, k, rng, case_n)
        accs.append(acc)
        cost += 3 * m['registers'] + m['gates'] + 0.2 * m['edges']
    return (min(accs), float(np.mean(accs)), -cost)

def _zeroshot(grammar, g, rng):
    per_k = {}
    for k in TRAIN_KS + ZEROSHOT_KS:
        acc, m = _acc(grammar, g, k, rng, 128, long=True)
        exact = None
        if acc == 1.0 and k <= VERIFY_MAX_K:
            v = verify(compile_seq(grammar['instantiate'](g, k),
                                   (grammar['x_n'],)),
                       TASKS1[TASK][1](k), warmup=k, max_states=1 << 21)
            exact = None if v['aborted'] else v['exact']
        per_k[k] = {'acc': acc, 'exact': exact, 'registers': m['registers'],
                    'gates': m['gates'], 'depth': m['logic_depth']}
    return per_k

def _solved_k(p):
    """Exact where verifiable; perfect holdout where not."""
    return p['exact'] if p['exact'] is not None else p['acc'] == 1.0


def train_run(arm, seed, pop_n=64, gen_max=800, case_n=64,
              elite_n=4, tourney_n=3, patience=150):
    grammar = ARMS[arm]
    rng = np.random.default_rng(seed)
    pop = [spec_random(rng, grammar['spec']) for _ in range(pop_n)]
    best_fit, stagnant, evals, solved_at = None, 0, 0, None
    for gen in range(gen_max):
        scored = []
        for g in pop:
            fit = eval_multi(grammar, g, rng, case_n)
            evals += 1
            scored.append((fit, g))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [g for _, g in scored]
        top = scored[0][0]
        if solved_at is None and top[0] == 1.0:
            solved_at = evals
        stagnant = stagnant + 1 if best_fit and top <= best_fit else 0
        best_fit = max(best_fit, top) if best_fit else top
        if top[0] == 1.0 and stagnant >= patience:
            break
        children = []
        while len(children) < pop_n - elite_n:
            winner = ranked[rng.integers(pop_n, size=tourney_n).min()]
            children.append(spec_mutate(winner, rng, grammar['spec']))
        pop = ranked[:elite_n] + children
    return _finish(ranked[0], arm, arm, seed, evals, solved_at, rng)

def _finish(g, arm, label, seed, evals, solved_at, rng):
    grammar = ARMS[arm]
    return {'arm': label, 'task': TASK, 'seed': seed, 'evals': evals,
            'evals_to_solve': solved_at,
            'genome_size': genome_size(grammar['spec']),
            'per_k': _zeroshot(grammar, g, rng),
            'genome': {k: v.tolist() for k, v in g.items()}}

def hand_run(arm):
    return _finish(HANDS[arm](ARMS[arm]), arm, f'hand_{arm}', 0, 0, 0,
                   np.random.default_rng(777))


def _run_one(job):
    return train_run(*job)

def sweep(seed_n, workers, out):
    open(out, 'w').close()
    with open(out, 'a') as f:
        for arm in HANDS:
            f.write(json.dumps(hand_run(arm)) + '\n')
    jobs = [(arm, seed) for arm in ARMS for seed in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            k16 = rec['per_k'].get(16) or rec['per_k'].get('16')
            print(f"{rec['arm']:<15} seed {rec['seed']}  "
                  f"solved_at {rec['evals_to_solve']}  "
                  f"k=16 acc {k16['acc']:.4f}", flush=True)
    summarize(out)


def _load(path):
    recs = [json.loads(line) for line in open(path)]
    for r in recs:
        r['per_k'] = {int(k): v for k, v in r['per_k'].items()}
    return recs

def summarize(path):
    recs = _load(path)
    nd = [k for k in ZEROSHOT_KS if k not in (8, 16)]
    print(f"\n== interface bandwidth: copy-after-delay "
          f"(train k=1,2,4; freeze; zero-shot) ==")
    print(f"{'arm':<18} {'|g|':>4} {'train 1,2,4':>12} {'zs 8 (exact)':>13} "
          f"{'zs 16 (holdout)':>16} {'non-dyadic':>11}")
    for arm in ('hand_dev_p3', 'hand_shared_p3',
                'dev_p2', 'shared_p2', 'dev_p3', 'shared_p3'):
        rs = [r for r in recs if r['arm'] == arm]
        if not rs:
            continue
        def frac(ks):
            n = sum(all(_solved_k(r['per_k'][k]) for k in ks) for r in rs)
            return f"{n}/{len(rs)}"
        print(f"{arm:<18} {rs[0]['genome_size']:>4} {frac(TRAIN_KS):>12} "
              f"{frac([8]):>13} {frac([16]):>16} {frac(nd):>11}")
    winners = [r for r in recs if not r['arm'].startswith('hand')
               and all(_solved_k(r['per_k'][k])
                       for k in TRAIN_KS + ZEROSHOT_KS)]
    if winners:
        w = min(winners, key=lambda r: r['per_k'][16]['gates'])
        print(f"\nscaling of a frozen winner ({w['arm']} seed {w['seed']}, "
              f"|genome|={w['genome_size']}):")
        print(f"{'k':>4} {'registers':>10} {'gates':>6} {'depth':>6} "
              f"{'verdict':>16}")
        for k in TRAIN_KS + ZEROSHOT_KS:
            p = w['per_k'][k]
            verdict = ('FSM-exact' if p['exact'] else
                       'holdout-perfect' if p['acc'] == 1.0 else
                       f"acc {p['acc']:.3f}")
            print(f"{k:>4} {p['registers']:>10} {p['gates']:>6} "
                  f"{p['depth']:>6} {verdict:>16}")


def selftest():
    # 1. The generic builder at P=2/x_n=1 reproduces the frozen grammar.
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

    # 2. Hand P=3 laws are sufficient: exact / perfect at every size.
    for arm in HANDS:
        rec = hand_run(arm)
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
    s = sub.add_parser('sweep')
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp3_copy.jsonl')
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'sweep':
        sweep(args.seeds, args.workers, args.out)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
