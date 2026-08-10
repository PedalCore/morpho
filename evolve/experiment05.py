# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 0.5: replicate and causally probe the Experiment 0 discoveries
before building anything new.

  replicate  Many independent evolution runs per task/lattice size (genome
             length = N by design), each scored on disjoint held-out ICs,
             with 'sea'/defect motif analysis of every winner.
  summarize  Success distribution and sea-rule frequency across a batch.
  ablate     Causal tests on the best genome of a batch: revert each defect
             to the sea rule, move it, swap defect rules, flip single LUT
             bits — all scored on one fixed large IC bank.
  horizon    Re-score the best genome at T = 2N, 4N, 8N on a large bank.

Background: Rule 184 particle transport is the core of an exact two-rule
density classification scheme (Fuks, comp-gas/9703001); Sipper's cellular
programming evolved non-uniform radius-1 classifiers; Das, Mitchell &
Crutchfield analysed evolved classifiers as particle computations.

Usage (from repo root):
  python3 -m evolve.experiment05 replicate --task density --cells 15 --seeds 24
  python3 -m evolve.experiment05 summarize runs/exp05_density_n15.jsonl
  python3 -m evolve.experiment05 ablate    runs/exp05_density_n15.jsonl
  python3 -m evolve.experiment05 horizon   runs/exp05_density_n15.jsonl
"""

import argparse
import json
import multiprocessing as mp
from collections import Counter

import numpy as np

from .evaluate import evaluate
from .tasks import TASKS
from .experiment0 import evolve

GEN_DEFAULT = {'density': 300, 'sync': 150}


def motif(genome):
    """A winner's structure: the dominant 'sea' rule and the exceptional cells."""
    rules, counts = np.unique(genome, return_counts=True)
    sea = int(rules[counts.argmax()])
    defects = [[i, int(r)] for i, r in enumerate(genome) if r != sea]
    return sea, float(counts.max() / len(genome)), defects


def _holdout(task, cell_n, case_n, key):
    make_ics, score = TASKS[task]
    return make_ics(np.random.default_rng(key), cell_n, case_n), score

def _run_one(cfg):
    task, cell_n, seed, gen_n = cfg
    step_n = 2 * cell_n
    log = evolve(task, cell_n=cell_n, gen_n=gen_n, step_n=step_n, seed=seed,
                 quiet=True)
    ics, score = _holdout(task, cell_n, 2000, seed + 10_000)
    smooth, strict = evaluate(log.best, ics, step_n, score)
    sea, sea_frac, defects = motif(log.best)
    return {'task': task, 'cell_n': cell_n, 'seed': seed, 'gen_n': gen_n,
            'step_n': step_n, 'train_best': log.best_fit,
            'holdout_smooth': smooth, 'holdout_strict': strict,
            'sea_rule': sea, 'sea_frac': sea_frac, 'defects': defects,
            'genome': [int(r) for r in log.best]}

def replicate(task, cell_n, seed_n, gen_n, workers, out):
    cfgs = [(task, cell_n, seed, gen_n) for seed in range(seed_n)]
    open(out, 'w').close()
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, cfgs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"seed {rec['seed']:3d}  smooth {rec['holdout_smooth']:.4f}"
                  f"  strict {rec['holdout_strict']:.4f}"
                  f"  sea {rec['sea_rule']:3d} x{rec['sea_frac']:.2f}"
                  f"  defects {len(rec['defects'])}", flush=True)
    summarize(out)


def _load(path):
    return [json.loads(line) for line in open(path)]

def _best(recs):
    return max(recs, key=lambda r: r['holdout_smooth'])

def summarize(path):
    recs = _load(path)
    smooth = np.array([r['holdout_smooth'] for r in recs])
    strict = np.array([r['holdout_strict'] for r in recs])
    print(f"\n== {recs[0]['task']}  N={recs[0]['cell_n']}  T={recs[0]['step_n']}"
          f"  {len(recs)} seeds x {recs[0]['gen_n']} generations ==")
    print(f"holdout smooth: median {np.median(smooth):.4f}  "
          f"IQR [{np.percentile(smooth, 25):.4f}, {np.percentile(smooth, 75):.4f}]  "
          f"best {smooth.max():.4f}  worst {smooth.min():.4f}")
    print(f"holdout strict: median {np.median(strict):.4f}  "
          f"best {strict.max():.4f}  solved(=1.0): {(strict == 1.0).sum()}/{len(recs)}")
    seas = Counter(r['sea_rule'] for r in recs)
    print(f"sea rules: {seas.most_common()}")
    print(f"mean sea fraction {np.mean([r['sea_frac'] for r in recs]):.2f}  "
          f"mean defects {np.mean([len(r['defects']) for r in recs]):.1f}")


def ablate(path, bank_n=5000):
    rec = _best(_load(path))
    task, cell_n, step_n = rec['task'], rec['cell_n'], rec['step_n']
    genome = np.array(rec['genome'], dtype=np.uint8)
    sea, defects = rec['sea_rule'], rec['defects']
    ics, score = _holdout(task, cell_n, bank_n, 424_242)
    ev = lambda g: evaluate(np.asarray(g, dtype=np.uint8), ics, step_n, score)

    base_smooth, base_strict = ev(genome)
    print(f"== ablation of best {task} genome (seed {rec['seed']}, N={cell_n}, "
          f"{bank_n} ICs) ==\ngenome {rec['genome']}\n"
          f"sea {sea}, defects {defects}\n")
    print(f"{'variant':<18} {'smooth':>8} {'strict':>8} {'d_smooth':>9}")
    def row(name, g):
        s, st = ev(g)
        print(f"{name:<18} {s:8.4f} {st:8.4f} {s - base_smooth:+9.4f}")
    row('evolved', genome)
    row('uniform_sea', np.full(cell_n, sea))

    for pos, rule in defects:
        g = genome.copy()
        g[pos] = sea
        row(f'revert@{pos}', g)
        for d in (-1, 1):
            g = genome.copy()
            g[pos] = sea
            g[(pos + d) % cell_n] = rule
            row(f'move@{pos}{d:+d}', g)
    for a, (pi, ri) in enumerate(defects):
        for pj, rj in defects[a + 1:]:
            if ri == rj:
                continue
            g = genome.copy()
            g[pi], g[pj] = rj, ri
            row(f'swap@{pi},{pj}', g)
    for pos, rule in defects:
        flips = []
        for bit in range(8):
            g = genome.copy()
            g[pos] ^= np.uint8(1 << bit)
            flips.append(ev(g)[0])
        print(f"{f'bitflips@{pos}':<18} mean {np.mean(flips):.4f}  "
              f"min {np.min(flips):.4f}  max {np.max(flips):.4f}  "
              f"(evolved {base_smooth:.4f})")


def horizon(path, bank_n=10_000):
    rec = _best(_load(path))
    task, cell_n = rec['task'], rec['cell_n']
    genome = np.array(rec['genome'], dtype=np.uint8)
    ics, score = _holdout(task, cell_n, bank_n, 434_343)
    print(f"== horizon sweep, best {task} genome (seed {rec['seed']}, "
          f"N={cell_n}, {bank_n} ICs) ==")
    for mult in (2, 4, 8):
        smooth, strict = evaluate(genome, ics, mult * cell_n, score)
        print(f"T={mult}N={mult * cell_n:4d}   smooth {smooth:.4f}   "
              f"strict {strict:.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('replicate')
    r.add_argument('--task', choices=TASKS, required=True)
    r.add_argument('--cells', type=int, default=15)
    r.add_argument('--seeds', type=int, default=24)
    r.add_argument('--generations', type=int, default=None)
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--out', default=None)
    for name in ('summarize', 'ablate', 'horizon'):
        s = sub.add_parser(name)
        s.add_argument('path')
    args = p.parse_args()

    if args.cmd == 'replicate':
        out = args.out or f'runs/exp05_{args.task}_n{args.cells}.jsonl'
        replicate(args.task, args.cells, args.seeds,
                  args.generations or GEN_DEFAULT[args.task], args.workers, out)
    elif args.cmd == 'summarize':
        summarize(args.path)
    elif args.cmd == 'ablate':
        ablate(args.path)
    elif args.cmd == 'horizon':
        horizon(args.path)


if __name__ == '__main__':
    main()
