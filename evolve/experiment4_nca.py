# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment 4A: scale-general neural cellular automata.

Can evolution discover a constant-size local neural update law whose
repeated instantiation grows, maintains, and repairs an exact global
pattern at spatial scales never encountered during evolution?

Preregistered protocol:
  - quantized NCA: C=6 binary channels, H=12 hidden threshold units,
    ternary weights, integer biases; von Neumann neighbourhood;
    fixed-zero boundaries; canonical central seed; fully synchronous
    (every recurrent path crosses REG; no FORWARD/TIE)
  - genome = the shared rule only (450 ints), constant at every size
  - task: seed-phased checkerboard; primary score = phase-contrast Dice
    (empty and all-ones states score 0); accuracy is diagnostic only
  - train ONLY on 8, 12, 16; fitness = (min dev morph, mean dev morph,
    min persistence, min 25%-deletion repair, -nonzero weights)
  - zero-shot: 10, 14, 20, 24, 32, 48, 64 (frozen, incl. non-multiples);
    horizons 2N/4N/8N; strong solution = exact checkerboard held >= 2N
    consecutive steps; damage suite: 10/25/40% deletion + 25%-area wound,
    all channels cleared
  - arms: evolution vs random search at matched budget, plus a hand
    reference law (expressibility calibration, not a claimed optimum)
  - large-grid results only via the numpy stepper AFTER bit-exact
    agreement with the compiled Morpho circuit is established (selftest)

Usage (from repo root):
  python3 -m evolve.experiment4_nca selftest
  python3 -m evolve.experiment4_nca show          # hand-law demo frames
  python3 -m evolve.experiment4_nca sweep --seeds 8 --workers 6
  python3 -m evolve.experiment4_nca summarize runs/exp4_nca.jsonl
"""

import argparse
import json
import multiprocessing as mp

import numpy as np

from tiny_morpho_seq import compile_seq
from .nca_genome import (C, H, random_genome, genome_size, nonzero_weights,
                         hand_genome)
from .nca_mutate import mutate
from .nca_grid import (step_np, rollout, instantiate_nca, pack_state,
                       unpack_state)
from .nca_tasks import seed_state, checkerboard, DAMAGE_OPS, damage_random
from .nca_metrics import morph, exact
from .nca_evaluate import train_fitness, zero_shot

TRAIN_SIZES = (8, 12, 16)
ZEROSHOT_SIZES = (10, 14, 20, 24, 32, 48, 64)
assert not set(TRAIN_SIZES) & set(ZEROSHOT_SIZES)


def hardware_metrics(g, n=6):
    sim = compile_seq(instantiate_nca(g, n, n), ())
    m = sim.metrics()
    return {'nonzero_weights': nonzero_weights(g),
            'state_bits_per_cell': C,
            'threshold_units': H + C,
            'addsub_per_cell_step': nonzero_weights(g),
            'compiled_grid': f'{n}x{n}',
            'registers': m['registers'], 'gates': m['gates'],
            'edges': m['edges'], 'logic_depth': m['logic_depth'],
            'gates_per_cell': round(m['gates'] / (n * n), 1)}


def train_run(arm, seed, pop_n=64, gen_max=600, elite_n=4, tourney_n=3,
              patience=150):
    rng = np.random.default_rng(seed)
    pop = [random_genome(rng) for _ in range(pop_n)]
    best_fit, stagnant, evals, solved_at, history = None, 0, 0, None, []
    for gen in range(gen_max):
        scored = []
        for g in pop:
            fit = train_fitness(g, TRAIN_SIZES, (seed, gen))
            evals += 1
            scored.append((fit, g))
        scored.sort(key=lambda t: t[0], reverse=True)
        ranked = [g for _, g in scored]
        top = scored[0][0]
        first = solved_at is None and top[0] == 1.0
        if first:
            solved_at = evals
        if gen % 20 == 0 or first or gen == gen_max - 1:
            history.append({'gen': gen, 'evals': evals, 'fit': list(top)})
        stagnant = stagnant + 1 if best_fit and top <= best_fit else 0
        best_fit = max(best_fit, top) if best_fit else top
        if top[0] == 1.0 and stagnant >= patience:
            break
        children = []
        while len(children) < pop_n - elite_n:
            winner = ranked[rng.integers(pop_n, size=tourney_n).min()]
            children.append(mutate(winner, rng))
        pop = ranked[:elite_n] + children
    return _finish(ranked[0], arm, seed, evals, solved_at, history)

def random_run(arm, seed, budget):
    rng = np.random.default_rng(seed + 900_000)
    best, best_fit, solved_at = None, None, None
    for k in range(budget):
        g = random_genome(rng)
        fit = train_fitness(g, TRAIN_SIZES, (seed, k % 1000))
        if best_fit is None or fit > best_fit:
            best, best_fit = g, fit
            if solved_at is None and fit[0] == 1.0:
                solved_at = k + 1
    return _finish(best, arm, seed, budget, solved_at, [])

def _finish(g, arm, seed, evals, solved_at, history):
    fit = train_fitness(g, TRAIN_SIZES, (999, 0))
    return {'arm': arm, 'seed': seed, 'evals': evals,
            'evals_to_solve': solved_at, 'train_fit': list(fit),
            'genome_size': genome_size(g),
            'zero_shot': [zero_shot(g, n, seed=(7000 + n))
                          for n in ZEROSHOT_SIZES],
            'hardware': hardware_metrics(g),
            'history': history,
            'genome': {k: v.tolist() for k, v in g.items()}}

def hand_run():
    return _finish(hand_genome(), 'hand', 0, 0, 0, [])


def _run_one(job):
    arm, seed, budget = job
    if arm == 'evolution':
        return train_run(arm, seed)
    return random_run(arm, seed, budget)

def sweep(seed_n, workers, out, pop_n=64, gen_max=600):
    open(out, 'w').close()
    with open(out, 'a') as f:
        f.write(json.dumps(hand_run()) + '\n')
    jobs = [(arm, seed, pop_n * gen_max)
            for arm in ('evolution', 'random') for seed in range(seed_n)]
    with mp.Pool(workers) as pool:
        for rec in pool.imap_unordered(_run_one, jobs):
            with open(out, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            zs = rec['zero_shot'][-1]
            print(f"{rec['arm']:<10} seed {rec['seed']}  "
                  f"train min-morph {rec['train_fit'][0]:.3f}  "
                  f"solved_at {rec['evals_to_solve']}  "
                  f"N=64 morph {zs['morph_2N']:.3f} strong {zs['strong']}",
                  flush=True)
    summarize(out)


def summarize(path):
    recs = [json.loads(line) for line in open(path)]
    print(f"\n== Experiment 4A: checkerboard NCA "
          f"(train {TRAIN_SIZES}; zero-shot {ZEROSHOT_SIZES}) ==")
    print(f"{'arm':<10} {'|g|':>4} {'solved':>7} {'strong@all-zs':>14} "
          f"{'med evals':>10} {'min zs morph':>13} {'repair(del25@64)':>17}")
    for arm in ('hand', 'evolution', 'random'):
        rs = [r for r in recs if r['arm'] == arm]
        if not rs:
            continue
        solved = [r for r in rs if r['train_fit'][0] == 1.0]
        strong = [r for r in solved
                  if all(z['strong'] for z in r['zero_shot'])]
        ev = [r['evals_to_solve'] for r in solved
              if r['evals_to_solve'] is not None] or None
        minz = max((min(z['morph_2N'] for z in r['zero_shot'])
                    for r in rs), default=0)
        rep = max((r['zero_shot'][-1]['damage']['del25']['final']
                   for r in rs), default=0)
        print(f"{arm:<10} {rs[0]['genome_size']:>4} "
              f"{f'{len(solved)}/{len(rs)}':>7} "
              f"{f'{len(strong)}/{len(solved)}' if solved else '-':>14} "
              f"{int(np.median(ev)) if ev else '-':>10} "
              f"{minz:>13.3f} {rep:>17.3f}")
    best = max((r for r in recs if r['train_fit'][0] == 1.0
                and r['arm'] != 'hand'),
               key=lambda r: min(z['morph_2N'] for z in r['zero_shot']),
               default=None)
    for rec, label in ((next((r for r in recs if r['arm'] == 'hand'), None),
                        'hand law'), (best, 'best evolved')):
        if not rec:
            continue
        print(f"\n{label}: zero-shot per size "
              f"(morph@2N / persist-min / exact-streak>=2N / del25 final):")
        for z in rec['zero_shot']:
            d = z['damage']['del25']
            print(f"  N={z['n']:>3}: {z['morph_2N']:.3f} / "
                  f"{z['persist_min']:.3f} / {str(z['strong']):>5} / "
                  f"{d['final']:.3f}{' (exact)' if d['exact_final'] else ''}")


def _ascii(v):
    return '\n'.join(''.join('#' if b else '.' for b in row) for row in v)

def show(n=13):
    g = hand_genome()
    t = checkerboard(n, n)
    frames = rollout(g, seed_state(n, n), 2 * n, record=True)
    print(f"hand law, N={n}: seed -> t=N -> t=2N "
          f"(exact: {exact(frames[-1][0], t)})")
    for f in (frames[1], frames[n], frames[2 * n]):
        print(_ascii(f[0]), '\n')
    rng = np.random.default_rng(5)
    wounded = DAMAGE_OPS['wound25'](frames[-1], rng)
    rec = rollout(g, wounded, 2 * n, record=True)
    print(f"after 25%-area wound -> recovered at t=+2N "
          f"(exact: {exact(rec[-1][0], t)})")
    print(_ascii(wounded[0]), '\n')
    print(_ascii(rec[-1][0]))


#@MARK: required selftests

def selftest():
    rng = np.random.default_rng(0)

    def shift_genome(nb):
        g = {'w1': np.zeros((H, 5 * C), np.int8),
             'b1': np.full(H, -1, np.int8),
             'w2': np.zeros((C, H), np.int8),
             'b2': np.full(C, -1, np.int8)}
        g['w1'][0][nb * C] = 1      # h0 = visible bit of neighbour nb
        g['w2'][0][0] = 1           # out0 = h0
        return g

    # 1+2. synchronous updates and neighbour indexing: a copy-from-nb rule
    # translates a lone dot by exactly one cell per step, no smearing.
    moves = {1: (1, 0), 2: (-1, 0), 3: (0, -1), 4: (0, 1)}  # N,S,E,W sources
    for nb, (dy, dx) in moves.items():
        s = np.zeros((C, 7, 7), np.int16)
        s[0, 3, 3] = 1
        nxt = step_np(shift_genome(nb), s)
        assert nxt[0].sum() == 1 and nxt[0, 3 + dy, 3 + dx] == 1, nb
    print("1-2. synchronous update and N/S/E/W indexing verified")

    # 3. fixed-zero boundary: with copy-from-W, column 0 reads 0.
    s = np.ones((C, 5, 5), np.int16)
    nxt = step_np(shift_genome(4), s)
    assert (nxt[0][:, 0] == 0).all() and (nxt[0][:, 1:] == 1).all()
    print("3. fixed-zero boundary semantics verified (documented: no torus)")

    # 4. numpy stepper == compiled Morpho circuit, bit-exact.
    for ny, nx in ((3, 4), (4, 4), (5, 3)):
        for gi, g in enumerate([hand_genome(), random_genome(rng),
                                random_genome(rng)]):
            s0 = rng.integers(2, size=(C, ny, nx, 2)).astype(np.int16)
            frames = rollout(g, s0, 6, record=True)
            sim = compile_seq(instantiate_nca(g, ny, nx), (), optimize=False)
            state0 = np.stack([pack_state(s0[..., b]) for b in range(2)], 1)
            tr = sim.run(7, state0=state0, samples=2)
            for t in range(7):
                for b in range(2):
                    got = unpack_state(tr[:, t, b], ny, nx)
                    assert (got == frames[t][..., b]).all(), (ny, nx, gi, t)
    print("4. Morpho compiled circuit == numpy stepper, bit-exact "
          "(3 grids x 3 genomes x 7 steps x 2 states)")

    # 5. one shared rule: translation equivariance in the interior.
    g = random_genome(rng)
    s = np.zeros((C, 9, 9), np.int16)
    s[:, 3:5, 3:5] = rng.integers(2, size=(C, 2, 2))
    a, b = step_np(g, s), step_np(g, np.roll(s, (1, 1), axis=(1, 2)))
    assert (np.roll(a, (1, 1), axis=(1, 2))[:, 2:8, 2:8]
            == b[:, 2:8, 2:8]).all()
    print("5. identical shared weights: interior translation equivariance")

    # 6. genome length is size-independent.
    g = hand_genome()
    snap = {k: v.copy() for k, v in g.items()}
    rollout(g, seed_state(8, 8), 4)
    rollout(g, seed_state(32, 32), 4)
    assert genome_size(g) == 450
    assert all((snap[k] == g[k]).all() for k in g)
    print("6. genome constant (450 ints) across lattice sizes")

    # 7. damage clears every channel.
    s = np.ones((C, 10, 10), np.int16)
    for name, op in DAMAGE_OPS.items():
        d = op(s, np.random.default_rng(1))
        hit = d[0] == 0
        assert hit.sum() > 0 and (d[:, hit] == 0).all(), name
    print("7. all damage operators clear visible AND latent channels")

    # 8. phase-contrast Dice: trivial states score ~0.
    t = checkerboard(8, 8)
    assert morph(np.zeros_like(t), t) == 0.0
    assert morph(np.ones_like(t), t) == 0.0
    assert morph(t.copy(), t) == 1.0
    print("8. empty and all-ones phenotypes score 0; exact target scores 1")

    # 9. zero-shot sizes never appear in training (asserted at import too).
    assert not set(TRAIN_SIZES) & set(ZEROSHOT_SIZES)
    print("9. training and zero-shot size sets are disjoint")

    # 10. frozen genome instantiates at arbitrary valid dimensions.
    for ny, nx in ((7, 13), (10, 10)):
        rollout(hand_genome(), seed_state(ny, nx), 8)
    compile_seq(instantiate_nca(hand_genome(), 3, 3), ())
    print("10. frozen genome instantiates at arbitrary grid dimensions")

    # Hand-law calibration: grows exactly, persists, repairs.
    for n in (8, 13):
        g, t = hand_genome(), checkerboard(n, n)
        frames = rollout(g, seed_state(n, n), 8 * n, record=True)
        assert exact(frames[2 * n][0], t)
        assert all(exact(f[0], t) for f in frames[2 * n:])
        wounded = DAMAGE_OPS['wound25'](frames[4 * n],
                                        np.random.default_rng(2))
        rec = rollout(g, wounded, 2 * n)
        assert exact(rec[0], t), f"hand law failed to repair at N={n}"
    print("hand law: exact growth by 2N, exact persistence to 8N, "
          "exact wound repair at N=8 and N=13")
    print("selftest passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    s = sub.add_parser('show')
    s.add_argument('--n', type=int, default=13)
    s2 = sub.add_parser('sweep')
    s2.add_argument('--seeds', type=int, default=8)
    s2.add_argument('--workers', type=int, default=6)
    s2.add_argument('--out', default='runs/exp4_nca.jsonl')
    s3 = sub.add_parser('summarize')
    s3.add_argument('path')
    args = p.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'show':
        show(args.n)
    elif args.cmd == 'sweep':
        sweep(args.seeds, args.workers, args.out)
    else:
        summarize(args.path)


if __name__ == '__main__':
    main()
