# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 5A: multi-attractor distributed memory.

The core property under test, stronger than 4A regeneration:

    information survives destruction of the place where it originated.

One input bit — a single latent bit at the seed cell — selects which of two
scale-general attractors the tissue must develop: horizontal stripes (b=0)
or vertical stripes (b=1). Both classes use IDENTICAL lattice geometry and
boundary conditions; the environment must not identify the class. After
development, damage destroys the ENTIRE seed region (a centered square
wound covering ~25% of the lattice, always containing the origin, all
channels cleared) and more. Success is regenerating the morphology of the
ORIGINAL input bit — H -> damage -> H and V -> damage -> V. A system that
regenerates some valid morphology but forgets which one has regeneration
without memory.

Three separable quantities: development accuracy, attractor persistence,
and memory-preserving regeneration (the new science). Two localization
controls tell us where the memory lives: randomize the visible channel
while preserving latents; erase all latents while preserving the visible
morphology.

Preregistered: train {8,12,16} x both classes; fitness (min dev, mean dev,
min persistence, min memory-regeneration, -nonzero weights), lexicographic;
pop 48, gen_max 400, patience 120, elite 4, tourney 3; 8 seeds/arm; arms
k1 (homogeneous), k4 (differentiated tissue), random_k1 (matched budget),
hand (reference with flood-distributed class bit — passes seed destruction
by construction, fails the latent-erase control: latent-only memory).
Zero-shot {9,10,14,15,20,24,33,48,64}, both classes, full battery.

Usage (from repo root):
  python3 -m evolve.experiment5_memory selftest
  python3 -m evolve.experiment5_memory sweep --seeds 8 --workers 6
  python3 -m evolve.experiment5_memory summarize runs/exp5_memory.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from .nca_genome import C, H, IN_N
from .nca_metrics import morph, exact, longest_exact_streak, recovery_stats
from .nca_tasks import damage_random
from .nca_types import (random_typed_genome, typed_genome_size,
                        typed_nonzero_weights, type_map, rollout_typed,
                        instantiate_typed)
from .experiment4b import mutate_typed

TRAIN_SIZES = (8, 12, 16)
ZEROSHOT_SIZES = (9, 10, 14, 15, 20, 24, 33, 48, 64)
assert not set(TRAIN_SIZES) & set(ZEROSHOT_SIZES)


#@MARK: task

def stripes(ny, nx, cls):
    """cls=0: horizontal stripes (row 0 active); cls=1: vertical (col 0).
    Identical geometry and boundary for both classes at every size."""
    yy, xx = np.mgrid[:ny, :nx]
    return ((xx if cls else yy) % 2 == 0).astype(np.int16)

def seed_bit(ny, nx, b):
    """Canonical seed; the ONLY class difference is one latent bit."""
    s = np.zeros((C, ny, nx), dtype=np.int16)
    s[0, ny // 2, nx // 2] = 1
    s[1, ny // 2, nx // 2] = b
    return s

def center_wound(s):
    """Deterministic square wound centered on the seed cell, ~25% of the
    lattice, all channels cleared — the origin never survives."""
    ny, nx = s.shape[1:3]
    side = max(2, int(round(np.sqrt(.25 * ny * nx))))
    y0 = max(0, ny // 2 - side // 2)
    x0 = max(0, nx // 2 - side // 2)
    out = s.copy()
    out[:, y0:y0 + side, x0:x0 + side] = 0
    return out

def scramble_visible(s, rng):
    out = s.copy()
    out[0] = rng.integers(2, size=out[0].shape)
    return out

def erase_latents(s):
    out = s.copy()
    out[1:] = 0
    return out


#@MARK: hand reference

def hand_stripes_genome():
    """Distributed memory by construction. ch3 = class field: floods from
    the seed's input bit through the whole tissue and self-sustains. The
    visible rule is a class-conditional anti-alignment: b=0 -> v' = NOT
    north (top edge anchors horizontal stripes); b=1 -> v' = NOT west
    (left edge anchors vertical). The stripe pattern is globally
    attracting from any state, so any wound heals; the class survives seed
    destruction because it lives everywhere. Predicted control behaviour:
    survives visible scramble (memory is latent), loses class 1 under
    latent erase (defaults to horizontal)."""
    net = {'w1': np.zeros((H, IN_N), np.int8),
           'b1': np.full(H, -1, np.int8),
           'w2': np.zeros((C, H), np.int8),
           'b2': np.full(C, -1, np.int8)}
    V, B, F = 0, 1, 3                     # visible, input bit, class field
    idx = lambda nb, c: nb * C + c        # [self, N, S, E, W]
    # h0: class-field flood source: F anywhere nearby, or the seed bit
    for nb in range(5):
        net['w1'][0][idx(nb, F)] = 1
    net['w1'][0][idx(0, B)] = 1
    # h1: class 0 stripe rule: (not F) and (not N.v)
    net['w1'][1][idx(0, F)] = -1
    net['w1'][1][idx(1, V)] = -1
    net['b1'][1] = 0
    # h2: class 1 stripe rule: F and (not W.v)
    net['w1'][2][idx(0, F)] = 1
    net['w1'][2][idx(4, V)] = -1
    # outputs: v' = h1 | h2 ; F' = h0
    net['w2'][V][1:3] = 1
    net['w2'][F][0] = 1
    return {'nets': [net], 'k': 1}


#@MARK: evaluation

def _predict(v, n):
    mh, mv = morph(v, stripes(n, n, 0)), morph(v, stripes(n, n, 1))
    return (0 if mh >= mv else 1), mh, mv

def class_scores(g, tm, n, b):
    t = stripes(n, n, b)
    frames = rollout_typed(g, tm, seed_bit(n, n, b), 4 * n, record=True)
    dev = morph(frames[2 * n][0], t)
    persist = min(morph(f[0], t) for f in frames[2 * n:])
    rec = rollout_typed(g, tm, center_wound(frames[4 * n]), 2 * n)
    return dev, persist, morph(rec[0], t)

def train_fitness(g):
    devs, persists, mems = [], [], []
    for n in TRAIN_SIZES:
        tm = type_map(g, n, n)
        for b in (0, 1):
            d, p, m = class_scores(g, tm, n, b)
            devs.append(d)
            persists.append(p)
            mems.append(m)
    return (min(devs), float(np.mean(devs)), min(persists), min(mems),
            -typed_nonzero_weights(g))

def zero_shot(g, n, seed):
    rng = np.random.default_rng(seed)
    tm = type_map(g, n, n)
    out = {'n': n, 'classes': {}}
    for b in (0, 1):
        t = stripes(n, n, b)
        frames = rollout_typed(g, tm, seed_bit(n, n, b), 8 * n, record=True)
        vis = [f[0] for f in frames]
        dev4 = frames[4 * n]
        rec = {'morph_2N': morph(vis[2 * n], t),
               'persist_min': min(morph(v, t) for v in vis[2 * n:]),
               'strong': longest_exact_streak(vis, t) >= 2 * n}
        # memory battery: seed region always destroyed
        for name, state in (
                ('center_wound', center_wound(dev4)),
                ('wound_plus_del25', damage_random(center_wound(dev4),
                                                   .25, rng)),
                ('del40', damage_random(dev4, .40, rng))):
            end = rollout_typed(g, tm, state, 4 * n)
            cls, mh, mv = _predict(end[0], n)
            rec[name] = {'correct_class': cls == b,
                         'morph_correct': morph(end[0], t),
                         'margin': (mh - mv) * (1 if b == 0 else -1),
                         'exact': exact(end[0], t)}
        # localization controls
        for name, state in (('scramble_visible',
                             scramble_visible(dev4, rng)),
                            ('erase_latents', erase_latents(dev4))):
            end = rollout_typed(g, tm, state, 2 * n)
            cls, _, _ = _predict(end[0], n)
            rec[name] = {'correct_class': cls == b,
                         'morph_correct': morph(end[0], t)}
        out['classes'][b] = rec
    both = [out['classes'][b] for b in (0, 1)]
    out['memory'] = all(r['center_wound']['correct_class'] and
                        r['center_wound']['morph_correct'] == 1.0
                        for r in both)
    return out


#@MARK: driver

def train_run(arm, k, seed, pop_n=48, gen_max=400, elite_n=4, tourney_n=3,
              patience=120):
    rng = np.random.default_rng(seed)
    pop = [random_typed_genome(rng, k) for _ in range(pop_n)]
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
    return _finish(ranked[0], arm, seed, evals, solved_at)

def random_run(arm, k, seed, budget):
    rng = np.random.default_rng(seed + 700_000)
    best, best_fit, solved_at = None, None, None
    for j in range(budget):
        g = random_typed_genome(rng, k)
        fit = train_fitness(g)
        if best_fit is None or fit > best_fit:
            best, best_fit = g, fit
            if solved_at is None and fit[0] == 1.0:
                solved_at = j + 1
    return _finish(best, arm, seed, budget, solved_at)

def _finish(g, arm, seed, evals, solved_at):
    fit = train_fitness(g)
    return {'arm': arm, 'seed': seed, 'evals': evals,
            'evals_to_solve': solved_at, 'train_fit': list(fit),
            'genome_size': typed_genome_size(g),
            'zero_shot': [zero_shot(g, n, seed=(5100 + n))
                          for n in ZEROSHOT_SIZES],
            'genome': {'k': g['k'],
                       'nets': [{k2: v.tolist() for k2, v in net.items()}
                                for net in g['nets']],
                       **({'children': g['children'].tolist(),
                           'base': g['base'].tolist()}
                          if g['k'] > 1 else {})}}

def hand_run():
    return _finish(hand_stripes_genome(), 'hand', 0, 0, 0)


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
            for arm, k in (('k1', 1), ('k4', 4), ('random_k1', 1))
            for seed in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            mem = sum(z['memory'] for z in rec['zero_shot'])
            print(f"{rec['arm']:<10} seed {rec['seed']}  "
                  f"train {rec['train_fit'][0]:.3f}  "
                  f"solved_at {rec['evals_to_solve']}  "
                  f"memory-exact at {mem}/{len(ZEROSHOT_SIZES)} zs sizes",
                  flush=True)
    summarize(out)


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    print(f"\n== Experiment 5A: multi-attractor memory "
          f"(train {TRAIN_SIZES}; zero-shot {ZEROSHOT_SIZES}) ==")
    print(f"{'arm':<10} {'|g|':>5} {'solved':>7} {'memory@all-zs':>14} "
          f"{'med evals':>10} {'scramble-ok':>12} {'erase-ok':>9}")
    for arm in ('hand', 'k1', 'k4', 'random_k1'):
        rs = [r for r in recs if r['arm'] == arm]
        if not rs:
            continue
        solved = [r for r in rs if r['train_fit'][0] == 1.0]
        memall = [r for r in solved if all(z['memory']
                                           for z in r['zero_shot'])]
        ev = [r['evals_to_solve'] for r in solved
              if r['evals_to_solve'] is not None] or None

        def control_ok(r, name):
            return all(z['classes'][b][name]['correct_class']
                       for z in r['zero_shot'] for b in ('0', '1'))
        scr = sum(control_ok(r, 'scramble_visible') for r in solved)
        ers = sum(control_ok(r, 'erase_latents') for r in solved)
        print(f"{arm:<10} {rs[0]['genome_size']:>5} "
              f"{f'{len(solved)}/{len(rs)}':>7} "
              f"{f'{len(memall)}/{len(solved)}' if solved else '-':>14} "
              f"{int(np.median(ev)) if ev else '-':>10} "
              f"{f'{scr}/{len(solved)}' if solved else '-':>12} "
              f"{f'{ers}/{len(solved)}' if solved else '-':>9}")
    print("\nmemory = exact correct-class regeneration after the "
          "seed-destroying center wound, both classes.\n"
          "scramble-ok = class retained with visible randomized "
          "(memory in latents); erase-ok = class retained with latents "
          "erased (memory in visible).")


def selftest():
    from tiny_morpho_seq import compile_seq
    from .nca_grid import pack_state, unpack_state
    rng = np.random.default_rng(0)
    # 1. identical environment: the two class trials differ in exactly
    # one bit of the initial state; targets share geometry.
    for n in (8, 9):
        d = seed_bit(n, n, 1) - seed_bit(n, n, 0)
        assert d.sum() == 1 and d[1, n // 2, n // 2] == 1
        assert stripes(n, n, 0).shape == stripes(n, n, 1).shape
    print("1. classes differ by exactly one seed bit; identical geometry")
    # 2. center wound always destroys the origin, all channels.
    for n in (8, 13):
        s = np.ones((C, n, n), np.int16)
        w = center_wound(s)
        assert (w[:, n // 2, n // 2] == 0).all()
        assert (w[:, w[0] == 0] == 0).all() and (w[0] == 0).sum() >= n * n // 5
    print("2. center wound destroys the seed region, every channel")
    # 3. hand law: both classes develop exactly, persist, and survive the
    # seed-destroying wound with the CORRECT attractor (incl. odd N).
    g = hand_stripes_genome()
    for n in (8, 13):
        tm = type_map(g, n, n)
        for b in (0, 1):
            t = stripes(n, n, b)
            frames = rollout_typed(g, tm, seed_bit(n, n, b), 8 * n,
                                   record=True)
            assert exact(frames[2 * n][0], t), (n, b)
            assert all(exact(f[0], t) for f in frames[2 * n:])
            end = rollout_typed(g, tm, center_wound(frames[4 * n]), 2 * n)
            assert exact(end[0], t), f"memory lost N={n} b={b}"
    print("3. hand law: exact development, persistence, and "
          "memory-preserving regeneration for both classes (N=8, 13)")
    # 4. localization controls behave as predicted for the hand law.
    n, tm = 12, type_map(g, 12, 12)
    for b in (0, 1):
        dev = rollout_typed(g, tm, seed_bit(n, n, b), 4 * n)
        s_end = rollout_typed(g, tm, scramble_visible(dev, rng), 2 * n)
        assert _predict(s_end[0], n)[0] == b
        e_end = rollout_typed(g, tm, erase_latents(dev), 2 * n)
        e_cls = _predict(e_end[0], n)[0]
        print(f"   b={b}: scramble-visible retained class {b}; "
              f"erase-latents -> class {e_cls} "
              f"({'retained' if e_cls == b else 'LOST — latent memory'})")
    # 5. typed Morpho circuit == stepper for the hand law (small grid).
    s0 = seed_bit(4, 4, 1)
    frames = rollout_typed(g, type_map(g, 4, 4), s0, 5, record=True)
    sim = compile_seq(instantiate_typed(g, 4, 4), (), optimize=False)
    tr = sim.run(6, state0=pack_state(s0)[:, None], samples=1)
    for t in range(6):
        assert (unpack_state(tr[:, t], 4, 4) == frames[t]).all()
    print("5. typed Morpho circuit == stepper, bit-exact on the hand law")
    assert not set(TRAIN_SIZES) & set(ZEROSHOT_SIZES)
    print("6. training and zero-shot sizes disjoint")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('sweep')
    s.add_argument('--seeds', type=int, default=8)
    s.add_argument('--workers', type=int, default=6)
    s.add_argument('--out', default='runs/exp5_memory.jsonl')
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
