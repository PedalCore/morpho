# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 2: developmental recurrent programs.

Preregistered protocol (temporal parity):
  - constant-length genomes; instantiation size k never changes genome length
  - train ONLY at k in {1, 2, 4}: primary fitness = min accuracy over the
    three sizes (a law that breaks any training size is not a law),
    tiebreak mean accuracy, then hardware cost. k >= 8 is never exposed
    during evolution, not even for model selection.
  - freeze the winner, zero-shot instantiate at k = 3, 5, 6, 7, 8, 12, 16
    (non-dyadic sizes guard against ladder-specialized encodings)
  - success = FSM-exact at unseen sizes, via product-machine BFS
  - representations: developmental (recursive) vs shared-rule (parameter
    tying without recursion) vs direct (Experiment 1 result, at-size)
  - hand-written laws in both grammars as expressibility references
  - measure N_reg(k), N_gate(k), depth(k), |genome|

Usage (from repo root):
  python3 -m evolve.experiment2 selftest
  python3 -m evolve.experiment2 sweep --seeds 8 --workers 6
  python3 -m evolve.experiment2 summarize runs/exp2_parity.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from tiny_morpho_seq import compile_seq
from .develop_genome import (DEV_SPEC, instantiate_dev, hand_dev,
                             spec_random, spec_mutate, genome_size)
from .shared_genome import SHARED_SPEC, instantiate_shared, hand_shared
from .temporal_tasks import TASKS1, score
from .fsm_verify import verify

REPS = {'developmental': (DEV_SPEC, instantiate_dev, hand_dev),
        'shared_rule': (SHARED_SPEC, instantiate_shared, hand_shared)}
TRAIN_KS = (1, 2, 4)
ZEROSHOT_KS = (3, 5, 6, 7, 8, 12, 16)


def _step_n(k):
    return max(32, 3 * k + 16)

def _acc(inst, g, k, rng, case_n, task, long=False):
    x, target, mask = TASKS1[task][0](rng, k, case_n * (4 if long else 1),
                                      _step_n(k) * (2 if long else 1))
    sim = compile_seq(inst(g, k), (1,))
    y = sim.run(x.shape[1], x, samples=x.shape[2])
    m = sim.metrics()
    return score(y[0], target, mask), m

def eval_multi(g, inst, rng, task, case_n=64):
    accs, cost = [], 0.0
    for k in TRAIN_KS:
        acc, m = _acc(inst, g, k, rng, case_n, task)
        accs.append(acc)
        cost += 3 * m['registers'] + m['gates'] + 0.2 * m['edges']
    return (min(accs), float(np.mean(accs)), -cost)

def _zeroshot(g, inst, rng, task):
    per_k = {}
    for k in TRAIN_KS + ZEROSHOT_KS:
        acc, m = _acc(inst, g, k, rng, 128, task, long=True)
        exact = None
        if acc == 1.0:
            v = verify(compile_seq(inst(g, k), (1,)), TASKS1[task][1](k),
                       warmup=k, max_states=1 << 21)
            exact = v['exact']
        per_k[k] = {'acc': acc, 'exact': exact, 'registers': m['registers'],
                    'gates': m['gates'], 'depth': m['logic_depth']}
    return per_k


def train_run(rep, seed, task='parity', pop_n=64, gen_max=800, case_n=64,
              elite_n=4, tourney_n=3, patience=150, quiet=True):
    spec, inst, _ = REPS[rep]
    rng = np.random.default_rng(seed)
    pop = [spec_random(rng, spec) for _ in range(pop_n)]
    best_fit, stagnant, evals, solved_at = None, 0, 0, None

    for gen in range(gen_max):
        scored = []
        for g in pop:
            fit = eval_multi(g, inst, rng, task, case_n)
            evals += 1
            scored.append((fit, g))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [g for _, g in scored]
        top = scored[0][0]
        if solved_at is None and top[0] == 1.0:
            solved_at = evals
        stagnant = stagnant + 1 if best_fit and top <= best_fit else 0
        best_fit = max(best_fit, top) if best_fit else top
        if not quiet and gen % 20 == 0:
            print(f"gen {gen:4d}  min {top[0]:.4f}  mean {top[1]:.4f}  "
                  f"cost {-top[2]:.1f}")
        if top[0] == 1.0 and stagnant >= patience:
            break
        children = []
        while len(children) < pop_n - elite_n:
            winner = ranked[rng.integers(pop_n, size=tourney_n).min()]
            children.append(spec_mutate(winner, rng, spec))
        pop = ranked[:elite_n] + children

    return _finish(ranked[0], rep, rep, seed, evals, solved_at, rng, task)

def _finish(g, rep, label, seed, evals, solved_at, rng, task):
    spec, inst, _ = REPS[rep]
    return {'rep': label, 'task': task, 'seed': seed, 'evals': evals,
            'evals_to_solve': solved_at, 'genome_size': genome_size(spec),
            'per_k': _zeroshot(g, inst, rng, task),
            'genome': {k: v.tolist() for k, v in g.items()}}

def hand_run(rep, task='parity'):
    spec, inst, hand = REPS[rep]
    return _finish(hand(task), rep, f'hand_{rep}', 0, 0, 0,
                   np.random.default_rng(777), task)


def _run_one(job):
    rep, seed, task = job
    return train_run(rep, seed, task)

def sweep(seed_n, workers, out, task='parity'):
    open(out, 'w').close()
    with open(out, 'a') as f:
        for rep in REPS:
            f.write(json.dumps(hand_run(rep, task)) + '\n')
    jobs = [(rep, seed, task) for rep in REPS for seed in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            solved = all(rec['per_k'][k]['exact'] for k in map(str, TRAIN_KS)) \
                if isinstance(next(iter(rec['per_k'])), str) else \
                all(rec['per_k'][k]['exact'] for k in TRAIN_KS)
            zs = rec['per_k'].get(16) or rec['per_k'].get('16')
            print(f"{rec['rep']:<15} seed {rec['seed']}  "
                  f"train-exact {solved}  k=16 acc {zs['acc']:.4f} "
                  f"exact {zs['exact']}", flush=True)
    summarize(out)


def _load(path):
    recs = [json.loads(line) for line in open(path)]
    for r in recs:                       # JSON keys are strings; restore ints
        r['per_k'] = {int(k): v for k, v in r['per_k'].items()}
    return recs

def summarize(path):
    recs = _load(path)
    nd = [k for k in ZEROSHOT_KS if k not in (8, 16)]
    task = recs[0].get('task', 'parity')
    print(f"\n== developmental generalization: {task} "
          f"(train k=1,2,4; freeze; zero-shot) ==")
    print(f"{'representation':<22} {'|g|':>4} {'train 1,2,4':>12} "
          f"{'zs 8':>5} {'zs 16':>6} {'non-dyadic 3,5,6,7,12':>22}")
    for rep in ('hand_developmental', 'hand_shared_rule',
                'developmental', 'shared_rule'):
        rs = [r for r in recs if r['rep'] == rep]
        if not rs:
            continue
        def frac(ks):
            n = sum(all(r['per_k'][k]['exact'] for k in ks) for r in rs)
            return f"{n}/{len(rs)}"
        print(f"{rep:<22} {rs[0]['genome_size']:>4} {frac(TRAIN_KS):>12} "
              f"{frac([8]):>5} {frac([16]):>6} {frac(nd):>22}")
    winners = [r for r in recs if not r['rep'].startswith('hand')
               and all(r['per_k'][k]['exact'] for k in TRAIN_KS + ZEROSHOT_KS)]
    if winners:
        w = winners[0]
        print(f"\nscaling of a frozen winner ({w['rep']} seed {w['seed']}, "
              f"|genome|={w['genome_size']}):")
        print(f"{'k':>4} {'registers':>10} {'gates':>6} {'depth':>6} {'exact':>6}")
        for k in TRAIN_KS + ZEROSHOT_KS:
            p = w['per_k'][k]
            print(f"{k:>4} {p['registers']:>10} {p['gates']:>6} "
                  f"{p['depth']:>6} {str(p['exact']):>6}")


def selftest():
    for task in ('parity', 'recall'):
        for rep in REPS:
            rec = hand_run(rep, task)
            for k in TRAIN_KS + ZEROSHOT_KS:
                p = rec['per_k'][k]
                assert p['exact'], f"hand {rep}/{task} not exact at k={k}: {p}"
            regs = [rec['per_k'][k]['registers'] for k in (1, 2, 4, 8, 16)]
            print(f"hand {rep}/{task}: |genome|={rec['genome_size']}, exact "
                  f"at all unseen k, registers(1,2,4,8,16)={regs}")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('sweep')
    s.add_argument('--task', choices=TASKS1, default='parity')
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default=None)
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'sweep':
        out = args.out or f'runs/exp2_{args.task}.jsonl'
        sweep(args.seeds, args.workers, out, args.task)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
