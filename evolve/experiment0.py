# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 0: evolve the per-cell rules of a non-uniform elementary CA
(cellular-programming style) on density classification or synchronization.

The scientific content is deliberately thin — the goal is to validate that
    genome -> Morpho -> compile_seq -> simulate -> fitness
holds up over thousands of full-pipeline evaluations, with deterministic
seeding, held-out evaluation, an oracle cross-check and behavioural metrics.

Usage (from repo root):
    python3 -m evolve.experiment0 --task density --generations 150
"""

import argparse
import numpy as np

from .genome import random_genome
from .mutate import mutate
from .evaluate import build_sim, evaluate, run_trace, oracle_check
from .tasks import TASKS
from .archive import RunLog
from . import metrics as dyn


def evolve(task='density', cell_n=15, pop_n=32, gen_n=150, step_n=30,
           case_n=100, elite_n=4, tourney_n=3, seed=0, log_path=None):
    make_ics, score = TASKS[task]
    rng = np.random.default_rng(seed)
    pop = [random_genome(rng, cell_n) for _ in range(pop_n)]
    log = RunLog(log_path)

    for gen in range(gen_n):
        ics = make_ics(rng, cell_n, case_n)  # fresh cases every generation
        fits = [evaluate(g, ics, step_n, score)[0] for g in pop]
        order = np.argsort(fits)[::-1]
        ranked = [pop[k] for k in order]
        best, mean = fits[order[0]], float(np.mean(fits))
        log.log({'gen': gen, 'best': best, 'mean': mean,
                 'best_genome': [int(r) for r in ranked[0]]},
                genome=ranked[0], fitness=best)
        if gen % 10 == 0 or gen == gen_n - 1:
            print(f"gen {gen:4d}  best {best:.4f}  mean {mean:.4f}")

        children = []
        while len(children) < pop_n - elite_n:
            winner = ranked[rng.integers(pop_n, size=tourney_n).min()]
            children.append(mutate(winner, rng))
        pop = ranked[:elite_n] + children
    return log


def report(genome, task, cell_n, step_n, seed, holdout_n=1000):
    make_ics, score = TASKS[task]
    rng = np.random.default_rng(seed + 10_000)  # disjoint from evolution rng
    ics = make_ics(rng, cell_n, holdout_n)
    oracle_check(genome, ics[:, :50], step_n)

    smooth, strict = evaluate(genome, ics, step_n, score)
    trace = run_trace(genome, ics, step_n)
    print(f"\nHeld-out ({holdout_n} fresh cases): "
          f"smooth {smooth:.4f}  strict {strict:.4f}")
    print("Genome:", [int(r) for r in genome])
    print("Static phenotype:", build_sim(genome).metrics())
    print("Dynamics:", dyn.describe(trace))
    print("Damage spread:",
          f"{dyn.damage_spread(run_trace, genome, ics, step_n, rng):.4f}")

    k = int(np.argmax(np.abs(ics.mean(0) - .5) < .12))  # a hard, near-tie case
    print(f"\nSample space-time diagram (IC density {ics[:, k].mean():.2f}):")
    for t in range(step_n):
        print(''.join('#' if v else '.' for v in trace[:, t, k]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, default='density')
    p.add_argument('--cells', type=int, default=15)
    p.add_argument('--pop', type=int, default=32)
    p.add_argument('--generations', type=int, default=150)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--cases', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--log', default=None, help='JSONL log path')
    args = p.parse_args()
    if args.task == 'density' and args.cells % 2 == 0:
        p.error('density needs an odd cell count')

    log = evolve(args.task, args.cells, args.pop, args.generations,
                 args.steps, args.cases, seed=args.seed, log_path=args.log)
    report(log.best, args.task, args.cells, args.steps, args.seed)


if __name__ == '__main__':
    main()
