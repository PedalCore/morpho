# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment C2-S: escaping the feedback bottleneck.

C1's deterministic beam solved the feed-forward (echo) and stateful
(hold) tiers but stalled on cross-module feedback (oneshot) and counted
sequencing (threewave), circling near-misses for six levels. The causal
question isolated here:

    Is the failure caused by deterministic top-K pruning eliminating
    temporarily-inferior but compositionally necessary intermediates?

Everything is held fixed — library, targets, size limits, scoring,
signatures, exact product verifier. The ONLY change is survivor
selection at each level:

    52 deterministic elites
  + 208 sampled without replacement, P ∝ exp(-error / T), T = 0.15
    (the partial-program pool stays top-160 deterministic, as in C1)

Preregistered readings: stochastic arm solves → the grammar was fine and
pruning was the bottleneck; markedly better near-misses but no solution
→ diversity traverses the composition path but feedback closure remains
hard; behaves like the deterministic beam → simple search diversity is
insufficient, motivating learned proposal policies and learned-library
transfer (C3-C5), with oneshot/threewave kept held out.

Arms: 'det' — a recorder rerun of the frozen C1 search (deterministic,
identical trajectory) capturing the canonical near-miss DAG and traces;
'stoch' — N independent seeds.

Usage (from repo root):
  python3 -m evolve.experiment_c2s run --seeds 8 --workers 6
  python3 -m evolve.experiment_c2s summarize runs/exp_c2s.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from .compose import interpret, pretty
from .experiment_c1 import (REFS, HAND, SCHEDULES, score, signature,
                            expansions, verify_finalist, synthesize,
                            BEAM_PARTIAL, C1_LIBRARY)
from .compose import COMPONENTS

ELITE_N, SAMPLE_N, TEMP = 52, 208, 0.15
TIERS = ('oneshot', 'threewave')


def synthesize_stochastic(name, seed, max_size=8):
    ref = REFS[name]
    rng = np.random.default_rng(seed)
    gens = [[(comp, params, ())]
            for comp in C1_LIBRARY if not COMPONENTS[comp]['ins']
            for params in COMPONENTS[comp]['params']]
    level, seen, evals = gens, set(), 0
    best_err, best_prog = 1.0, None
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
                    return {'tier': name, 'arm': 'stoch', 'seed': seed,
                            'solved': True, 'size': len(prog),
                            'evals': evals, 'prog_pretty': pretty(prog),
                            'prog': prog, 'verify': v}
        scored.sort(key=lambda t: (t[0], len(t[1]), pretty(t[1])))
        full = [(s, p) for s, p in scored if s < 1.0]
        part = [p for s, p in scored if s >= 1.0][:BEAM_PARTIAL]
        elites = [p for _, p in full[:ELITE_N]]
        rest = full[ELITE_N:]
        picked = []
        if rest:
            errs = np.array([s for s, _ in rest])
            w = np.exp(-errs / TEMP)
            probs = w / w.sum()
            take = min(SAMPLE_N, len(rest))
            idx = rng.choice(len(rest), size=take, replace=False, p=probs)
            picked = [rest[i][1] for i in idx]
        level = [q for p in elites + picked + part for q in expansions(p)]
    return {'tier': name, 'arm': 'stoch', 'seed': seed, 'solved': False,
            'evals': evals, 'best_err': best_err,
            'best_prog_pretty': pretty(best_prog) if best_prog else None,
            'best_prog': best_prog}


def det_recorder(name):
    """Frozen C1 search, rerun with the near-miss recorder — identical
    deterministic trajectory; captures the canonical near-miss."""
    r = synthesize(name, 8, quiet=True)
    rec = {'tier': name, 'arm': 'det', 'seed': 0,
           'solved': r['solved'], 'evals': r['evals']}
    prog = r.get('prog') or r.get('best_prog')
    if prog:
        rec.update({'best_err': r.get('best_err', 0.0),
                    'prog_pretty': pretty(prog), 'prog': prog})
        x = SCHEDULES[1]                       # two-pulse schedule
        rec['near_miss_trace'] = interpret(prog, len(x), x=x).tolist()
        rec['near_miss_input'] = x.tolist()
    return rec


def _run_one(job):
    arm, tier, seed = job
    if arm == 'det':
        return det_recorder(tier)
    return synthesize_stochastic(tier, seed)

def run(seed_n, workers, out):
    open(out, 'w').close()
    jobs = [('det', t, 0) for t in TIERS] + \
           [('stoch', t, s) for t in TIERS for s in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{rec['arm']:<6} {rec['tier']:<10} seed {rec['seed']}  "
                  f"solved {rec['solved']}  evals {rec['evals']}  "
                  f"err {rec.get('best_err', 0):.4f}", flush=True)
    summarize(out)


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    print("\n== C2-S: deterministic vs stochastic beam "
          f"(elites {ELITE_N} + sampled {SAMPLE_N}, T={TEMP}) ==")
    for tier in TIERS:
        det = next((r for r in recs
                    if r['tier'] == tier and r['arm'] == 'det'), None)
        st = [r for r in recs if r['tier'] == tier and r['arm'] == 'stoch']
        solved = [r for r in st if r['solved']]
        errs = sorted(r.get('best_err', 0) for r in st if not r['solved'])
        print(f"\n[{tier}]")
        if det:
            print(f"  det:   solved {det['solved']}  "
                  f"err {det.get('best_err', 0):.4f}  evals {det['evals']}")
        if st:
            ev = sorted(r['evals'] for r in solved)
            print(f"  stoch: solved {len(solved)}/{len(st)}"
                  + (f"  evals-to-solve median {ev[len(ev) // 2]}"
                     if solved else '')
                  + (f"  unsolved best-err min {errs[0]:.4f} "
                     f"median {errs[len(errs) // 2]:.4f}" if errs else ''))
        for r in solved[:2]:
            print(f"  -- solver (seed {r['seed']}, size {r['size']}):")
            for line in r['prog_pretty'].split('\n'):
                print(f"     {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('run')
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp_c2s.jsonl')
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    args = ap.parse_args()
    if args.cmd == 'run':
        run(args.seeds, args.workers, args.out)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
