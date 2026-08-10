# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 1: evolve input-driven synchronous recurrent circuits.

Calibration task is delayed recall, where the optimum is known: any exact
machine needs >= d state bits, and the d-register shift register achieves it
with zero gates. Preregistered protocol:

  - lags d = 1, 2, 4, 8, 16; several seeds each
  - lexicographic selection: correctness first, hardware cost second
  - hardware cost measured on the live optimized phenotype (post-DCE)
  - random-search control under the identical evaluation budget
  - hand-built shift register as the reference optimum
  - Pareto archive of (error, registers, gates, edges, depth)
  - solved circuits verified EXACTLY against the reference FSM by
    product-machine BFS (fsm_verify), not just on sampled streams

Usage (from repo root):
  python3 -m evolve.experiment1 selftest
  python3 -m evolve.experiment1 sweep --task recall --lags 1,2,4,8,16 --seeds 4
  python3 -m evolve.experiment1 summarize runs/exp1_recall.jsonl
  python3 -m evolve.experiment1 inspect runs/exp1_recall.jsonl --lag 8
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from tiny_morpho_seq import compile_seq
from .recurrent_genome import random_genome, genome_to_cell, shift_register
from .recurrent_mutate import mutate
from .temporal_tasks import TASKS1, score
from .fsm_verify import verify

COST_W = {'registers': 3.0, 'gates': 1.0, 'edges': 0.2}


def make_cfg(task, d):
    node_n = 16 if task == 'recall' else max(16, d + 8)
    return {'x_n': 1, 'out_n': 1, 'state_n': d + 6, 'node_n': node_n,
            'task': task, 'lag': d}

def evaluate_net(g, cfg, case):
    x, target, mask = case
    sim = compile_seq(genome_to_cell(g, cfg), (cfg['x_n'],))
    y = sim.run(x.shape[1], x, samples=x.shape[2])
    m = sim.metrics()
    cost = sum(COST_W[k] * m[k] for k in COST_W)
    return score(y[0], target, mask), cost, m

def _case(rng, task, d, case_n, long=False):
    make_case = TASKS1[task][0]
    step_n = max(32, 3 * d + 16) * (2 if long else 1)
    return make_case(rng, d, case_n * (4 if long else 1), step_n)

def _pareto_add(archive, point):
    """point: (error, registers, gates, edges, depth). Keep non-dominated."""
    dominated = lambda a, b: all(x >= y for x, y in zip(a, b)) and a != b
    if any(dominated(point, a) for a in archive):
        return archive
    return [a for a in archive if not dominated(a, point)] + [point]

def _finish(g, cfg, rng, evals, evals_to_solve, archive, method, seed):
    """Held-out evaluation and exact FSM verification of the final genome."""
    acc, cost, m = evaluate_net(g, cfg, _case(rng, cfg['task'], cfg['lag'],
                                              128, long=True))
    exact = None
    if acc == 1.0:
        sim = compile_seq(genome_to_cell(g, cfg), (cfg['x_n'],))
        ref = TASKS1[cfg['task']][1](cfg['lag'])
        exact = verify(sim, ref, warmup=cfg['lag'])
    return {'task': cfg['task'], 'lag': cfg['lag'], 'method': method,
            'seed': seed, 'evals': evals, 'evals_to_solve': evals_to_solve,
            'holdout_acc': acc, 'cost': cost, 'metrics': m,
            'exact': exact and exact['exact'],
            'exact_states': exact and exact['states'],
            'pareto': archive,
            'genome': {k: v.tolist() for k, v in g.items()}}


def evolve_run(task, d, seed, pop_n=64, gen_max=600, case_n=64,
               elite_n=4, tourney_n=3, patience=150, quiet=True):
    cfg = make_cfg(task, d)
    rng = np.random.default_rng(seed)
    pop = [random_genome(rng, cfg) for _ in range(pop_n)]
    archive, best_fit, stagnant, evals, solved_at = [], None, 0, 0, None

    for gen in range(gen_max):
        case = _case(rng, task, d, case_n)
        scored = []
        for g in pop:
            acc, cost, m = evaluate_net(g, cfg, case)
            evals += 1
            scored.append(((acc, -cost), g))
            archive = _pareto_add(archive, (round(1 - acc, 4), m['registers'],
                                            m['gates'], m['edges'],
                                            m['logic_depth']))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [g for _, g in scored]
        top = scored[0][0]
        if solved_at is None and top[0] == 1.0:
            solved_at = evals
        stagnant = stagnant + 1 if best_fit and top <= best_fit else 0
        best_fit = max(best_fit, top) if best_fit else top
        if not quiet and gen % 20 == 0:
            print(f"gen {gen:4d}  acc {top[0]:.4f}  cost {-top[1]:.1f}")
        if top[0] == 1.0 and stagnant >= patience:
            break
        children = []
        while len(children) < pop_n - elite_n:
            winner = ranked[rng.integers(pop_n, size=tourney_n).min()]
            children.append(mutate(winner, rng, cfg))
        pop = ranked[:elite_n] + children

    return _finish(ranked[0], cfg, rng, evals, solved_at, archive,
                   'evolution', seed)

def random_run(task, d, seed, budget, case_n=64):
    cfg = make_cfg(task, d)
    rng = np.random.default_rng(seed + 500_000)
    best, best_fit, solved_at, archive = None, None, None, []
    for k in range(budget):
        g = random_genome(rng, cfg)
        acc, cost, m = evaluate_net(g, cfg, _case(rng, task, d, case_n))
        fit = (acc, -cost)
        if best_fit is None or fit > best_fit:
            best, best_fit = g, fit
            archive = _pareto_add(archive, (round(1 - acc, 4), m['registers'],
                                            m['gates'], m['edges'],
                                            m['logic_depth']))
            if solved_at is None and acc == 1.0:
                solved_at = k + 1
    return _finish(best, cfg, rng, budget, solved_at, archive,
                   'random', seed)

def reference_run(task, d):
    if task != 'recall':
        return None
    cfg = make_cfg(task, d)
    cfg['state_n'] = d
    g = shift_register(d, cfg)
    rng = np.random.default_rng(0)
    return _finish(g, cfg, rng, 0, 0, [], 'shift_register', 0)


def _run_one(job):
    kind, task, d, seed, budget = job
    if kind == 'evolution':
        return evolve_run(task, d, seed)
    return random_run(task, d, seed, budget)

def sweep(task, lags, seed_n, workers, out, pop_n=64, gen_max=600):
    budget = pop_n * gen_max
    jobs = [(kind, task, d, seed, budget)
            for d in lags for seed in range(seed_n)
            for kind in ('evolution', 'random')]
    open(out, 'w').close()
    with open(out, 'a') as f:
        for d in lags:
            rec = reference_run(task, d)
            if rec:
                f.write(json.dumps(rec) + '\n')
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{rec['method']:<10} d={rec['lag']:<3} seed {rec['seed']}"
                  f"  acc {rec['holdout_acc']:.4f}  "
                  f"regs {rec['metrics']['registers']}  "
                  f"gates {rec['metrics']['gates']}  exact {rec['exact']}",
                  flush=True)
    summarize(out)


def _load(path):
    return [json.loads(line) for line in open(path)]

def summarize(path):
    recs = _load(path)
    lags = sorted({r['lag'] for r in recs})
    print(f"\n== {recs[0]['task']}: evolved recurrent circuits vs "
          f"d-bit lower bound ==")
    print(f"{'d':>3} {'method':<15} {'solved':>7} {'exact':>6} "
          f"{'evals->solve':>13} {'live regs':>10} {'gates':>6}")
    for d in lags:
        for method in ('shift_register', 'evolution', 'random'):
            rs = [r for r in recs if r['lag'] == d and r['method'] == method]
            if not rs:
                continue
            solved = [r for r in rs if r['holdout_acc'] == 1.0]
            exact = sum(1 for r in rs if r['exact'])
            ev = [r['evals_to_solve'] for r in solved
                  if r['evals_to_solve'] is not None]
            regs = sorted(r['metrics']['registers'] for r in solved)
            gates = sorted(r['metrics']['gates'] for r in solved)
            print(f"{d:>3} {method:<15} {len(solved)}/{len(rs):<5} {exact:>6} "
                  f"{(f'{int(np.median(ev))}' if ev else '-'):>13} "
                  f"{(str(regs) if regs else '-'):>10} "
                  f"{(str(gates) if gates else '-'):>6}")

def inspect(path, lag):
    recs = [r for r in _load(path) if r['lag'] == lag
            and r['method'] == 'evolution' and r['holdout_acc'] == 1.0]
    if not recs:
        print(f"no solved evolution runs at d={lag}")
        return
    rec = min(recs, key=lambda r: r['cost'])
    cfg = make_cfg(rec['task'], lag)
    g = {k: np.asarray(v) for k, v in rec['genome'].items()}
    sim = compile_seq(genome_to_cell(g, cfg), (cfg['x_n'],))
    print(f"best solved d={lag} (seed {rec['seed']}, exact={rec['exact']}, "
          f"cost {rec['cost']:.1f}):")
    for i, op in enumerate(sim.c.ops):
        args = f" <- {list(op.args)}" if op.args else ''
        lut = f" lut={op.lut:#x}" if op.type == 'GATE' else ''
        print(f"  [{i:3d}] {op.type:<6} {op.name}{lut}{args}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('sweep')
    s.add_argument('--task', choices=TASKS1, default='recall')
    s.add_argument('--lags', default='1,2,4,8,16')
    s.add_argument('--seeds', type=int, default=4)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default=None)
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    s3 = sub.add_parser('inspect')
    s3.add_argument('path')
    s3.add_argument('--lag', type=int, required=True)
    args = p.parse_args()

    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'sweep':
        out = args.out or f'runs/exp1_{args.task}.jsonl'
        sweep(args.task, [int(x) for x in args.lags.split(',')],
              args.seeds, args.workers, out)
    elif args.cmd == 'summarize':
        summarize(args.path)
    else:
        inspect(args.path, args.lag)


def selftest():
    for d in (1, 3, 5):
        rec = reference_run('recall', d)
        assert rec['holdout_acc'] == 1.0 and rec['exact'], f"d={d} failed"
        assert rec['metrics']['registers'] == d and rec['metrics']['gates'] == 0
        print(f"shift register d={d}: acc 1.0, {d} regs, 0 gates, "
              f"FSM-exact over {rec['exact_states']} product states")
    cfg = make_cfg('recall', 2)
    rng = np.random.default_rng(0)
    g = mutate(random_genome(rng, cfg), rng, cfg)
    acc, cost, m = evaluate_net(g, cfg, _case(rng, 'recall', 2, 32))
    print(f"random mutant evaluates: acc {acc:.3f}, cost {cost:.1f}, "
          f"metrics {m}")
    print("selftest passed")


if __name__ == '__main__':
    main()
