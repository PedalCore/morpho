# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Mechanistic analyses of the evolved density classifiers.

The Experiment 0.5 finding under test: a conservative traffic-rule medium
(184/226) plus sparse non-conservative defects performs approximate global
density classification. These analyses probe how.

  margin     strict accuracy vs initial majority margin |ones - N/2|
  failures   attractor taxonomy of misclassified trajectories
  geometry   every (i, j) placement of the two evolved defect rules
  landscape  all 256 rules at each defect position, others held fixed
  scaling    defect count and sea fraction vs lattice size N

Usage: python3 -m evolve.mechanism <cmd> runs/exp05_density_n15.jsonl
       python3 -m evolve.mechanism scaling
"""

import argparse
import glob
import json
from collections import Counter

import numpy as np

from .evaluate import evaluate, run_trace
from .tasks import TASKS
from .experiment05 import _load, _best, motif


def _setup(path, bank_n, key):
    rec = _best(_load(path))
    genome = np.array(rec['genome'], dtype=np.uint8)
    make_ics, score = TASKS[rec['task']]
    ics = make_ics(np.random.default_rng(key), rec['cell_n'], bank_n)
    return rec, genome, ics, score

def _per_case(trace_final, ics):
    target = (ics.sum(0) > ics.shape[0] // 2).astype(np.int32)
    return (trace_final == target).mean(0), target


def margin(path, bank_n=20_000):
    rec, genome, ics, _ = _setup(path, bank_n, 555_555)
    n = rec['cell_n']
    trace = run_trace(genome, ics, rec['step_n'])
    per_case, _ = _per_case(trace[:, -1], ics)
    ones = ics.sum(0)
    print(f"== strict accuracy vs majority margin (seed {rec['seed']}, "
          f"N={n}, {bank_n} ICs) ==")
    print(f"{'ones':>5} {'margin':>7} {'cases':>7} {'strict':>8} {'smooth':>8}")
    for k in range(n + 1):
        sel = ones == k
        if not sel.any():
            continue
        print(f"{k:5d} {abs(k - n / 2):7.1f} {sel.sum():7d} "
              f"{(per_case[sel] == 1.0).mean():8.4f} {per_case[sel].mean():8.4f}")


def failures(path, bank_n=20_000, horizon_mult=8):
    rec, genome, ics, _ = _setup(path, bank_n, 555_555)
    n, step_n = rec['cell_n'], rec['step_n']
    trace = run_trace(genome, ics, horizon_mult * n)
    per_case, target = _per_case(trace[:, step_n - 1], ics)
    failed = (per_case < 1.0).nonzero()[0]
    print(f"== attractor taxonomy of {len(failed)} failures / {bank_n} ICs "
          f"(judged at T=2N, evolved horizon x{horizon_mult}N) ==")

    def classify(k):
        last, prev = trace[:, -1, k], trace[:, -2, k]
        if (last == last[0]).all():
            correct = (last[0] == target[k])
            return 'uniform_correct_late' if correct else 'uniform_wrong'
        if any((last == np.roll(prev, s)).all() for s in range(1, n)):
            return 'traveling_orbit'
        tail = trace[:, -4 * n:, k]
        for p in range(1, tail.shape[1]):
            if (tail[:, -1] == tail[:, -1 - p]).all():
                return f'periodic_p{p}'
        return 'no_period_found'

    classes = Counter(classify(k) for k in failed)
    for name, count in classes.most_common():
        sel = [k for k in failed if classify(k) == name]
        print(f"{name:<22} {count:6d}  ({count / bank_n:6.2%} of all)  "
              f"mean ones {np.mean(ics[:, sel].sum(0)):5.2f}")


def geometry(path, bank_n=4000):
    rec, genome, ics, score = _setup(path, bank_n, 565_656)
    n, step_n, sea = rec['cell_n'], rec['step_n'], rec['sea_rule']
    (pa, ra), (pb, rb) = rec['defects']
    fit = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g = np.full(n, sea, dtype=np.uint8)
            g[i], g[j] = ra, rb
            fit[i, j] = evaluate(g, ics, step_n, score)[0]
    evolved = fit[pa, pb]
    print(f"== defect geometry: rules {ra}@i, {rb}@j on sea {sea} "
          f"(N={n}, {bank_n} ICs) ==")
    print(f"evolved placement ({pa},{pb}): {evolved:.4f}   "
          f"grid mean {np.nanmean(fit):.4f}  min {np.nanmin(fit):.4f}  "
          f"max {np.nanmax(fit):.4f}")
    print(f"{'directed sep':>12} {'mean':>8} {'std':>7} {'min':>8} {'max':>8}")
    for d in range(1, n):
        vals = np.array([fit[i, (i + d) % n] for i in range(n)])
        print(f"{d:12d} {vals.mean():8.4f} {vals.std():7.4f} "
              f"{vals.min():8.4f} {vals.max():8.4f}")
    out = path.replace('.jsonl', '_geometry.json')
    json.dump({'sea': sea, 'rules': [int(ra), int(rb)], 'fit': fit.tolist()},
              open(out, 'w'))
    print(f"full matrix -> {out}")


def landscape(path, bank_n=4000, tol=0.005):
    rec, genome, ics, score = _setup(path, bank_n, 575_757)
    step_n, sea = rec['step_n'], rec['sea_rule']
    evolved = evaluate(genome, ics, step_n, score)[0]
    print(f"== full LUT landscape per defect position "
          f"(N={rec['cell_n']}, {bank_n} ICs, evolved {evolved:.4f}) ==")
    curves = {}
    for pos, rule in rec['defects']:
        fits = np.zeros(256)
        for r in range(256):
            g = genome.copy()
            g[pos] = r
            fits[r] = evaluate(g, ics, step_n, score)[0]
        curves[pos] = fits.tolist()
        good = (fits >= fits.max() - tol).nonzero()[0]
        top = np.argsort(fits)[::-1][:8]
        print(f"\nposition {pos} (evolved rule {rule}, sea {sea}):")
        print(f"  sea rule here: {fits[sea]:.4f}   best {fits.max():.4f} "
              f"(rule {fits.argmax()})   rules within {tol} of best: {len(good)}")
        print(f"  top rules: " + '  '.join(
            f"{r}({fits[r]:.4f},d{bin(r ^ sea).count('1')})" for r in top))
        print(f"  near-best rules vs sea {sea}: " + ', '.join(
            f"{r}(^{r ^ sea:08b})" for r in sorted(good)))
    out = path.replace('.jsonl', '_landscape.json')
    json.dump({'sea': sea, 'evolved': evolved, 'curves': curves}, open(out, 'w'))
    print(f"\nfull curves -> {out}")


def scaling(pattern='runs/exp05_*_n*.jsonl'):
    print(f"== defect scaling across lattice sizes ==")
    print(f"{'task':<9} {'N':>4} {'seeds':>6} {'defects (median [min,max])':>28} "
          f"{'sea frac':>9} {'sea rules':<24}")
    for p in sorted(glob.glob(pattern)):
        if 'geometry' in p or 'landscape' in p:
            continue
        recs = _load(p)
        d = np.array([len(r['defects']) for r in recs])
        seas = Counter(r['sea_rule'] for r in recs)
        print(f"{recs[0]['task']:<9} {recs[0]['cell_n']:>4} {len(recs):>6} "
              f"{np.median(d):>14.1f} [{d.min()},{d.max()}] "
              f"{np.mean([r['sea_frac'] for r in recs]):>9.2f} "
              f"{str(seas.most_common(3)):<24}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('margin', 'failures', 'geometry', 'landscape'):
        s = sub.add_parser(name)
        s.add_argument('path')
    sub.add_parser('scaling').add_argument('pattern', nargs='?',
                                           default='runs/exp05_*_n*.jsonl')
    args = p.parse_args()
    if args.cmd == 'scaling':
        scaling(args.pattern)
    else:
        globals()[args.cmd](args.path)


if __name__ == '__main__':
    main()
