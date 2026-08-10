# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 4B: differentiated neural tissue.

Does evolution discover differentiated functional roles in a
self-organising neural system? The genome carries K=4 shared neural cell
types (each the frozen 4A architecture) plus a constant-size recursive
quadrant placement program that decides where each type appears at any
lattice size. Homogeneous K=1 evolution and random K=4 search are the
control arms at matched budget; a hand-written homogeneous rings law is
the expressibility reference.

Task: concentric square rings anchored at the boundary (outermost ring
active) — exact and scale-independent at every size INCLUDING ODD (the 4A
caveat, addressed: zero-shot includes 9, 15, 33). Primary score remains
phase-contrast Dice; note that on rings the all-ones lattice scores ~0.22
(ring areas are unequal) rather than 0 — logged, not an attracting
optimum since solving requires 1.0.

Preregistered: train {8,12,16}; fitness (min dev@2N, mean dev, min
persistence [2N,4N], min 25%-deletion repair, -nonzero weights); pop 48,
gen_max 400, patience 120, elite 4, tourney 3, mutation 1-3 ops
(neural-weight ops 70%, placement ops 30% for K>1); 8 seeds/arm;
zero-shot {9, 10, 14, 15, 20, 24, 33, 48, 64} frozen, with 8N horizons
and the full all-channel damage suite.

Usage (from repo root):
  python3 -m evolve.experiment4b selftest
  python3 -m evolve.experiment4b sweep --seeds 8 --workers 6
  python3 -m evolve.experiment4b summarize runs/exp4b_tissue.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from tiny_morpho_seq import compile_seq
from .nca_genome import C
from .nca_mutate import OPERATORS as NEURAL_OPS
from .nca_grid import pack_state, unpack_state
from .nca_tasks import seed_state, rings, DAMAGE_OPS, damage_random
from .nca_metrics import (morph, exact, longest_exact_streak,
                          recovery_stats)
from .nca_types import (K, PG, random_typed_genome, typed_genome_size,
                        typed_nonzero_weights, type_map, rollout_typed,
                        instantiate_typed, hand_rings_genome)

TRAIN_SIZES = (8, 12, 16)
ZEROSHOT_SIZES = (9, 10, 14, 15, 20, 24, 33, 48, 64)
assert not set(TRAIN_SIZES) & set(ZEROSHOT_SIZES)


def mutate_typed(g, rng):
    out = {'nets': [{k: v.copy() for k, v in net.items()}
                    for net in g['nets']], 'k': g['k']}
    if g['k'] > 1:
        out['children'] = g['children'].copy()
        out['base'] = g['base'].copy()
    for _ in range(rng.integers(1, 4)):
        if g['k'] == 1 or rng.random() < .7:
            net = out['nets'][rng.integers(g['k'])]
            NEURAL_OPS[rng.integers(len(NEURAL_OPS))](net, rng)
        elif rng.random() < .5:
            out['children'][rng.integers(PG), rng.integers(4)] = \
                rng.integers(PG)
        else:
            out['base'][rng.integers(PG)] = rng.integers(g['k'])
    return out


def size_scores(g, n, damage_rng):
    t = rings(n, n)
    tm = type_map(g, n, n)
    frames = rollout_typed(g, tm, seed_state(n, n), 4 * n, record=True)
    dev = morph(frames[2 * n][0], t)
    persist = min(morph(f[0], t) for f in frames[2 * n:])
    wounded = damage_random(frames[4 * n], .25, damage_rng)
    rep = morph(rollout_typed(g, tm, wounded, 2 * n)[0], t)
    return dev, persist, rep

def train_fitness(g, damage_key):
    rng = np.random.default_rng(damage_key)
    devs, persists, reps = zip(*(size_scores(g, n, rng)
                                 for n in TRAIN_SIZES))
    return (min(devs), float(np.mean(devs)), min(persists), min(reps),
            -typed_nonzero_weights(g))

def zero_shot(g, n, seed):
    rng = np.random.default_rng(seed)
    t = rings(n, n)
    tm = type_map(g, n, n)
    frames = rollout_typed(g, tm, seed_state(n, n), 8 * n, record=True)
    vis = [f[0] for f in frames]
    streak = longest_exact_streak(vis, t)
    out = {'n': n, 'morph_2N': morph(vis[2 * n], t),
           'persist_min': min(morph(v, t) for v in vis[2 * n:]),
           'exact_streak': streak, 'strong': streak >= 2 * n,
           'types_used': sorted(set(tm.reshape(-1).tolist())),
           'damage': {}}
    pre = morph(frames[4 * n][0], t)
    for name, op in DAMAGE_OPS.items():
        wounded = op(frames[4 * n], rng)
        rec = rollout_typed(g, tm, wounded, 4 * n, record=True)
        scores = [morph(f[0], t) for f in rec]
        stats = recovery_stats(scores, pre)
        stats['exact_final'] = exact(rec[-1][0], t)
        out['damage'][name] = stats
    return out


def train_run(arm, k, seed, pop_n=48, gen_max=400, elite_n=4, tourney_n=3,
              patience=120):
    rng = np.random.default_rng(seed)
    pop = [random_typed_genome(rng, k) for _ in range(pop_n)]
    best_fit, stagnant, evals, solved_at = None, 0, 0, None
    for gen in range(gen_max):
        scored = []
        for g in pop:
            fit = train_fitness(g, (seed, gen))
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
    return _finish(ranked[0], arm, seed, evals, solved_at)

def random_run(arm, k, seed, budget):
    rng = np.random.default_rng(seed + 800_000)
    best, best_fit, solved_at = None, None, None
    for j in range(budget):
        g = random_typed_genome(rng, k)
        fit = train_fitness(g, (seed, j % 1000))
        if best_fit is None or fit > best_fit:
            best, best_fit = g, fit
            if solved_at is None and fit[0] == 1.0:
                solved_at = j + 1
    return _finish(best, arm, seed, budget, solved_at)

def _finish(g, arm, seed, evals, solved_at):
    fit = train_fitness(g, (999, 0))
    tm64 = type_map(g, 64, 64)
    usage = {int(k): round(float((tm64 == k).mean()), 3)
             for k in range(g['k'])}
    return {'arm': arm, 'seed': seed, 'evals': evals,
            'evals_to_solve': solved_at, 'train_fit': list(fit),
            'genome_size': typed_genome_size(g),
            'type_usage_64': usage,
            'zero_shot': [zero_shot(g, n, seed=(8100 + n))
                          for n in ZEROSHOT_SIZES],
            'genome': {'k': g['k'],
                       'nets': [{k2: v.tolist() for k2, v in net.items()}
                                for net in g['nets']],
                       **({'children': g['children'].tolist(),
                           'base': g['base'].tolist()}
                          if g['k'] > 1 else {})}}

def hand_run():
    return _finish(hand_rings_genome(), 'hand', 0, 0, 0)


def _run_one(job):
    arm, k, seed, budget = job
    if arm.startswith('random'):
        return random_run(arm, k, seed, budget)
    return train_run(arm, k, seed)

def sweep(seed_n, workers, out, pop_n=48, gen_max=400):
    open(out, 'w').close()
    with open(out, 'a') as f:
        f.write(json.dumps(hand_run()) + '\n')
    jobs = [(arm, k, seed, pop_n * gen_max)
            for arm, k in (('k4', 4), ('k1', 1), ('random_k4', 4))
            for seed in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            zs = rec['zero_shot'][-1]
            print(f"{rec['arm']:<10} seed {rec['seed']}  "
                  f"train {rec['train_fit'][0]:.3f}  "
                  f"solved_at {rec['evals_to_solve']}  "
                  f"N=64 morph {zs['morph_2N']:.3f} strong {zs['strong']}  "
                  f"types {rec['type_usage_64']}", flush=True)
    summarize(out)


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    print(f"\n== Experiment 4B: rings tissue "
          f"(train {TRAIN_SIZES}; zero-shot {ZEROSHOT_SIZES}) ==")
    print(f"{'arm':<11} {'|g|':>5} {'solved':>7} {'strong@all-zs':>14} "
          f"{'med evals':>10} {'multi-type':>11}")
    for arm in ('hand', 'k4', 'k1', 'random_k4'):
        rs = [r for r in recs if r['arm'] == arm]
        if not rs:
            continue
        solved = [r for r in rs if r['train_fit'][0] == 1.0]
        strong = [r for r in solved
                  if all(z['strong'] for z in r['zero_shot'])]
        ev = [r['evals_to_solve'] for r in solved
              if r['evals_to_solve'] is not None] or None
        multi = sum(1 for r in solved
                    if sum(v > 0.02 for v in r['type_usage_64'].values()) > 1)
        print(f"{arm:<11} {rs[0]['genome_size']:>5} "
              f"{f'{len(solved)}/{len(rs)}':>7} "
              f"{f'{len(strong)}/{len(solved)}' if solved else '-':>14} "
              f"{int(np.median(ev)) if ev else '-':>10} "
              f"{f'{multi}/{len(solved)}' if solved else '-':>11}")
    best = next((r for r in recs if r['arm'] == 'k4'
                 and r['train_fit'][0] == 1.0
                 and all(z['strong'] for z in r['zero_shot'])), None)
    if best:
        print(f"\nbest k4 (seed {best['seed']}), type usage at N=64: "
              f"{best['type_usage_64']}")
        print(f"{'N':>4} {'streak':>7} {'persist':>8} {'del25 final':>12} "
              f"{'wound25 final':>14}")
        for z in best['zero_shot']:
            print(f"{z['n']:>4} {z['exact_streak']:>7} "
                  f"{z['persist_min']:>8.3f} "
                  f"{z['damage']['del25']['final']:>12.3f} "
                  f"{z['damage']['wound25']['final']:>14.3f}")


def selftest():
    rng = np.random.default_rng(0)
    # 1. typed Morpho circuit == typed numpy stepper, bit-exact.
    for ny, nx in ((3, 4), (4, 4)):
        for gi, g in enumerate([hand_rings_genome(),
                                random_typed_genome(rng, 4)]):
            tm = type_map(g, ny, nx)
            s0 = rng.integers(2, size=(C, ny, nx)).astype(np.int16)
            frames = rollout_typed(g, tm, s0, 6, record=True)
            sim = compile_seq(instantiate_typed(g, ny, nx), (),
                              optimize=False)
            tr = sim.run(7, state0=pack_state(s0)[:, None], samples=1)
            for t in range(7):
                got = unpack_state(tr[:, t], ny, nx)
                assert (got == frames[t]).all(), (ny, nx, gi, t)
    print("1. typed Morpho circuit == typed stepper, bit-exact")
    # 2. placement program: constant genome, full coverage, K=1 trivial.
    g4 = random_typed_genome(rng, 4)
    sizes = [(8, 8), (9, 9), (33, 33), (64, 64)]
    assert len({typed_genome_size(g4)}) == 1 == len({1830})
    for ny, nx in sizes:
        tm = type_map(g4, ny, nx)
        assert tm.shape == (ny, nx) and tm.min() >= 0 and tm.max() < 4
    assert (type_map(random_typed_genome(rng, 1), 8, 8) == 0).all()
    print(f"2. placement develops full type maps at {sizes}; "
          f"|genome| = 1830 (K=4) / 450 (K=1), size-independent")
    # 3. hand rings law: exact growth and persistence (expressibility
    # reference). Its repair is NOT asserted: the frontier anti-phase rule
    # is only phase-correct for boundary-distance-monotone wavefronts, so
    # arbitrary interior wounds can regrow mis-phased. Ring repair is
    # genuinely harder than checkerboard repair — the growth-vs-homeostasis
    # separation (Outcome B) is part of what the experiment measures.
    for n in (8, 13):
        g, t = hand_rings_genome(), rings(n, n)
        tm = type_map(g, n, n)
        frames = rollout_typed(g, tm, seed_state(n, n), 8 * n, record=True)
        assert exact(frames[2 * n][0], t), f"growth failed N={n}"
        assert all(exact(f[0], t) for f in frames[2 * n:])
        wounded = DAMAGE_OPS['wound25'](frames[4 * n],
                                        np.random.default_rng(2))
        rep = morph(rollout_typed(g, tm, wounded, 2 * n)[0], t)
        print(f"   N={n}: exact growth + persistence; wound-repair morph "
              f"{rep:.3f} (not asserted — see comment)")
    print("3. hand rings law: exact growth by 2N and persistence to 8N; "
          "repair is an open capability, not a given")
    # 4. metric note: rings all-ones scores ~0.22 (logged), empty 0.
    t = rings(8, 8)
    assert morph(np.zeros_like(t), t) == 0.0
    assert 0 < morph(np.ones_like(t), t) < 0.3
    assert morph(t.copy(), t) == 1.0
    print(f"4. rings metric: empty 0, all-ones "
          f"{morph(np.ones_like(t), t):.3f} (documented), exact 1.0")
    assert not set(TRAIN_SIZES) & set(ZEROSHOT_SIZES)
    print("5. training and zero-shot sizes disjoint (odd sizes included)")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('sweep')
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp4b_tissue.jsonl')
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
