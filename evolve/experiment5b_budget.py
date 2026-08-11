# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 5B: does evaluation budget dissolve the compositional wall?

As budget increases, does P(discover a multi-mechanism tissue law) rise
smoothly, or stay at zero? Everything except budget is frozen: same 5A
task, K=4 representation, mutation operators, tournament selection,
fitness, task distribution and hand reference. Budgets 1x/4x/16x/64x of
B = 48 x 400 = 19,200 evaluations (gen_max scaled; unsolved runs always
consume the full budget). The six K=4 runs completed in the stopped 5A
sweep are protocol-identical and are reused as 1x seeds 0-5.

Five mechanism probes, measured OFFLINE on final genomes and never
selected on, distinguish a sequential climb (stages accumulate with
budget) from a coordination cliff (stages never accumulate):

  1. class_info_spread    class bit reaches cells beyond the seed
  2. conditional_morph    visible development differs between classes
  3. stripe_dev           stripe-like morphology emerges (either target)
  4. memory_survives      class-discriminating state survives seed wound
  5. correct_regen        the full 5A criterion

Interpretation is preregistered: strong rise in P(solve) => unfavorable
search-time scaling, not a wall; flat zero with probes accumulating =>
mutation-only search fails to combine independently discovered
mechanisms (motivates module-aware crossover); flat zero without probe
accumulation => coordination cliff. No shaped fitness in any case.

Usage (from repo root):
  python3 -m evolve.experiment5b_budget selftest
  python3 -m evolve.experiment5b_budget sweep --workers 6
  python3 -m evolve.experiment5b_budget summarize runs/exp5b_budget.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from .nca_types import (random_typed_genome, typed_genome_size,
                        typed_nonzero_weights, type_map, step_typed,
                        _step_typed_ref, rollout_typed)
from .nca_metrics import morph, exact
from .nca_tasks import damage_random
from .experiment4b import mutate_typed
from .experiment5_memory import (TRAIN_SIZES, ZEROSHOT_SIZES, stripes,
                                 seed_bit, center_wound, zero_shot,
                                 train_fitness as train_fitness_5a)

BASE_BUDGET = 48 * 400
BUDGETS = (1, 4, 16, 64)
SEEDS = {1: 8, 4: 8, 16: 8, 64: 4}


#@MARK: batched fitness (bit-identical to 5A's, asserted in selftest)

def class_scores_both(g, tm, n):
    s0 = np.stack([seed_bit(n, n, 0), seed_bit(n, n, 1)], axis=-1)
    frames = rollout_typed(g, tm, s0, 4 * n, record=True)
    rec = rollout_typed(g, tm, center_wound(frames[4 * n]), 2 * n)
    out = []
    for b in (0, 1):
        t = stripes(n, n, b)
        out.append((morph(frames[2 * n][0, :, :, b], t),
                    min(morph(f[0, :, :, b], t) for f in frames[2 * n:]),
                    morph(rec[0, :, :, b], t)))
    return out

def train_fitness(g):
    devs, persists, mems = [], [], []
    for n in TRAIN_SIZES:
        tm = type_map(g, n, n)
        for d, p, m in class_scores_both(g, tm, n):
            devs.append(d)
            persists.append(p)
            mems.append(m)
    return (min(devs), float(np.mean(devs)), min(persists), min(mems),
            -typed_nonzero_weights(g))


#@MARK: mechanism probes (diagnostic only — never part of selection)

def probe_genome(g, n=16):
    tm = type_map(g, n, n)
    d = [rollout_typed(g, tm, seed_bit(n, n, b), 4 * n) for b in (0, 1)]
    w = [rollout_typed(g, tm, center_wound(s), 2 * n) for s in d]
    seed_mask = np.zeros((n, n), dtype=bool)
    seed_mask[n // 2, n // 2] = True

    def frac_differ(a, b):
        return float((a != b).any(axis=0)[~seed_mask].mean())

    stripe_dev = max(morph(s[0], stripes(n, n, c))
                     for s in d for c in (0, 1))
    correct = all(morph(w[b][0], stripes(n, n, b)) == 1.0 for b in (0, 1))
    return {'class_info_spread': frac_differ(d[0], d[1]) > 0.01,
            'conditional_morph':
                float((d[0][0] != d[1][0]).mean()) > 0.05,
            'stripe_dev': round(stripe_dev, 3),
            'stripe_dev_ok': stripe_dev >= 0.8,
            'memory_survives': frac_differ(w[0], w[1]) > 0.01,
            'correct_regen': correct}


#@MARK: runner (identical to 5A's K=4 arm; gen_max is the only knob)

def train_run(budget, seed, pop_n=48, elite_n=4, tourney_n=3, patience=120):
    gen_max = 400 * budget
    rng = np.random.default_rng(seed)
    pop = [random_typed_genome(rng, 4) for _ in range(pop_n)]
    best_fit, stagnant, evals, solved_at = None, 0, 0, None
    for gen in range(gen_max):
        scored = []
        for g in pop:
            fit = train_fitness(g)
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
            children.append(mutate_typed(winner, rng))
        pop = ranked[:elite_n] + children
    return _finish(ranked[0], budget, seed, evals, solved_at)

def _finish(g, budget, seed, evals, solved_at):
    fit = train_fitness(g)
    rec = {'budget': budget, 'seed': seed, 'evals': evals,
           'evals_to_solve': solved_at, 'train_fit': list(fit),
           'probes': probe_genome(g),
           'genome': {'k': g['k'],
                      'nets': [{k2: v.tolist() for k2, v in net.items()}
                               for net in g['nets']],
                      'children': g['children'].tolist(),
                      'base': g['base'].tolist()}}
    if fit[0] == 1.0:
        rec['zero_shot'] = [zero_shot(g, n, seed=(5100 + n))
                            for n in ZEROSHOT_SIZES]
    return rec


def _reuse_1x(out):
    """The stopped 5A sweep's completed K=4 runs are protocol-identical
    1x runs; import them (with probes) instead of recomputing."""
    reused = []
    try:
        for line in open('runs/exp5_memory.jsonl'):
            r = json.loads(line)
            if r['arm'] != 'k4':
                continue
            g = {'k': 4,
                 'nets': [{k2: np.asarray(v, dtype=np.int8)
                           for k2, v in net.items()}
                          for net in r['genome']['nets']],
                 'children': np.asarray(r['genome']['children'], np.int8),
                 'base': np.asarray(r['genome']['base'], np.int8)}
            rec = {'budget': 1, 'seed': r['seed'], 'evals': r['evals'],
                   'evals_to_solve': r['evals_to_solve'],
                   'train_fit': r['train_fit'], 'probes': probe_genome(g),
                   'genome': r['genome'], 'reused_from_5a': True}
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            reused.append(r['seed'])
    except FileNotFoundError:
        pass
    return reused


def _run_one(job):
    return train_run(*job)

def sweep(workers, out, budgets=(1, 4, 16), append=False):
    reused = []
    if not append:
        open(out, 'w').close()
        if 1 in budgets:
            reused = _reuse_1x(out)
            print(f"reused protocol-identical 1x runs from 5A: "
                  f"seeds {reused}", flush=True)
    jobs = [(b, seed) for b in budgets for seed in range(SEEDS[b])
            if not (b == 1 and seed in reused)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"budget {rec['budget']:>2}x seed {rec['seed']}  "
                  f"train {rec['train_fit'][0]:.3f}  "
                  f"solved_at {rec['evals_to_solve']}  "
                  f"probes {rec['probes']}", flush=True)
    summarize(out)


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    probe_keys = ('class_info_spread', 'conditional_morph', 'stripe_dev_ok',
                  'memory_survives', 'correct_regen')
    print(f"\n== Experiment 5B: P(solve | budget) on 5A K=4 "
          f"(B = {BASE_BUDGET}) ==")
    print(f"{'budget':>7} {'runs':>5} {'solved':>7} {'evals->solve':>13} "
          + ' '.join(f'{k[:12]:>13}' for k in probe_keys))
    for b in BUDGETS:
        rs = [r for r in recs if r['budget'] == b]
        if not rs:
            continue
        solved = [r for r in rs if r['train_fit'][0] == 1.0]
        ev = [r['evals_to_solve'] for r in solved
              if r['evals_to_solve'] is not None] or None
        frac = lambda k: f"{sum(bool(r['probes'][k]) for r in rs)}/{len(rs)}"
        print(f"{f'{b}x':>7} {len(rs):>5} {f'{len(solved)}/{len(rs)}':>7} "
              f"{int(np.median(ev)) if ev else '-':>13} "
              + ' '.join(f'{frac(k):>13}' for k in probe_keys))
    best = max(recs, key=lambda r: tuple(r['train_fit']))
    print(f"\nbest overall: budget {best['budget']}x seed {best['seed']}  "
          f"fit {[round(x, 3) for x in best['train_fit']]}  "
          f"stripe_dev {best['probes']['stripe_dev']}")


def selftest():
    rng = np.random.default_rng(0)
    # fast typed stepper == reference, with and without batch dims
    for _ in range(3):
        g = random_typed_genome(rng, 4)
        tm = type_map(g, 9, 9)
        s = rng.integers(2, size=(6, 9, 9)).astype(np.int16)
        assert (step_typed(g, s, tm) == _step_typed_ref(g, s, tm)).all()
        sb = rng.integers(2, size=(6, 9, 9, 3)).astype(np.int16)
        assert (step_typed(g, sb, tm) == _step_typed_ref(g, sb, tm)).all()
    print("1. fast typed stepper == reference, bit-exact (incl. batch)")
    # batched fitness == the 5A fitness, exactly
    for _ in range(3):
        g = random_typed_genome(rng, 4)
        assert train_fitness(g) == train_fitness_5a(g)
    print("2. class-batched fitness == 5A fitness on random genomes")
    # probes: hand law exhibits every stage; a dead genome exhibits none
    from .experiment5_memory import hand_stripes_genome
    p = probe_genome(hand_stripes_genome())
    assert all([p['class_info_spread'], p['conditional_morph'],
                p['stripe_dev_ok'], p['memory_survives'],
                p['correct_regen']])
    dead = random_typed_genome(rng, 4)
    for net in dead['nets']:
        net['w1'][:] = 0
        net['w2'][:] = 0
        net['b1'][:] = -1
        net['b2'][:] = -1
    pd = probe_genome(dead)
    assert not any([pd['class_info_spread'], pd['conditional_morph'],
                    pd['stripe_dev_ok'], pd['memory_survives'],
                    pd['correct_regen']])
    print("3. probes: hand law passes all five stages; the dead genome none")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('sweep')
    s.add_argument('--budgets', default='1,4,16',
                   help='which budget arms to run (64x gated on the curve)')
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp5b_budget.jsonl')
    s.add_argument('--append', action='store_true',
                   help='append to an existing results file (e.g. adding 64x)')
    s2 = sub.add_parser('summarize')
    s2.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'sweep':
        sweep(args.workers, args.out,
              tuple(int(x) for x in args.budgets.split(',')), args.append)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
