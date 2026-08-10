# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Development / persistence / regeneration evaluation for NCA genomes.

Training fitness (lexicographic, correctness before compression):
    ( min development morph across training sizes,
      mean development morph,
      min persistence morph,
      min repair morph,
      -nonzero weights )

Development is scored at T = 2N; persistence is the minimum score over
[2N, 4N]; repair applies the preregistered 25% random deletion at 4N and
scores after 2N further steps. All rollouts start from the canonical seed
and are deterministic; damage randomness is keyed per generation so every
genome in a generation faces identical damage."""

import numpy as np

from .nca_genome import nonzero_weights
from .nca_grid import rollout
from .nca_tasks import seed_state, checkerboard, damage_random, DAMAGE_OPS
from .nca_metrics import (morph, accuracy, exact, longest_exact_streak,
                          activity, latent_entropy, recovery_stats)


def _develop(g, n, steps, record=False):
    return rollout(g, seed_state(n, n), steps, record=record)

def size_scores(g, n, damage_rng):
    t = checkerboard(n, n)
    frames = _develop(g, n, 4 * n, record=True)
    dev = morph(frames[2 * n][0], t)
    persist = min(morph(f[0], t) for f in frames[2 * n:])
    wounded = damage_random(frames[4 * n], .25, damage_rng)
    rep = morph(rollout(g, wounded, 2 * n)[0], t)
    return dev, persist, rep

def train_fitness(g, sizes, damage_key):
    rng = np.random.default_rng(damage_key)
    devs, persists, reps = zip(*(size_scores(g, n, rng) for n in sizes))
    return (min(devs), float(np.mean(devs)), min(persists), min(reps),
            -nonzero_weights(g))


def zero_shot(g, n, seed):
    """Frozen-genome battery at one unseen size: horizons, exactness,
    persistence, and the full damage suite."""
    rng = np.random.default_rng(seed)
    t = checkerboard(n, n)
    frames = _develop(g, n, 8 * n, record=True)
    vis = [f[0] for f in frames]
    dev_state = frames[4 * n]
    out = {'n': n,
           'morph_2N': morph(vis[2 * n], t),
           'morph_4N': morph(vis[4 * n], t),
           'morph_8N': morph(vis[8 * n], t),
           'acc_2N': accuracy(vis[2 * n], t),
           'exact_2N': exact(vis[2 * n], t),
           'exact_streak': longest_exact_streak(vis, t),
           'strong': longest_exact_streak(vis, t) >= 2 * n,
           'persist_min': min(morph(v, t) for v in vis[2 * n:]),
           'persist_mean': float(np.mean([morph(v, t)
                                          for v in vis[2 * n:]])),
           'activity': activity(vis[2 * n:]),
           'latent_entropy': latent_entropy(frames[4 * n]),
           'damage': {}}
    pre = morph(dev_state[0], t)
    for name, op in DAMAGE_OPS.items():
        wounded = op(dev_state, rng)
        rec = rollout(g, wounded, 4 * n, record=True)
        scores = [morph(f[0], t) for f in rec]
        stats = recovery_stats(scores, pre)
        stats['exact_final'] = exact(rec[-1][0], t)
        out['damage'][name] = stats
    return out
