# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Morphology metrics. Primary score is phase-contrast Dice:

    morph = max(0, Dice(A, T) - Dice(A, ~T))

so the empty phenotype scores 0 and the all-ones phenotype (which overlaps
both checkerboard phases equally) also scores 0 — Experiment 3's lesson
about trivial optima, applied. Raw accuracy is logged as a diagnostic only.
"""

import numpy as np


def dice(a, t):
    inter = int((a & t).sum())
    denom = int(a.sum()) + int(t.sum())
    return 2 * inter / denom if denom else 0.0

def morph(a, t):
    return max(0.0, dice(a, t) - dice(a, 1 - t))

def accuracy(a, t):
    return float((a == t).mean())

def exact(a, t):
    return bool((a == t).all())

def longest_exact_streak(visibles, t):
    best = run = 0
    for a in visibles:
        run = run + 1 if exact(a, t) else 0
        best = max(best, run)
    return best

def activity(frames):
    d = [np.abs(frames[i + 1] - frames[i]).mean()
         for i in range(len(frames) - 1)]
    return float(np.mean(d)) if d else 0.0

def latent_entropy(state):
    p = np.clip(state[1:].mean(axis=(1, 2)), 1e-9, 1 - 1e-9)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)).mean())

def recovery_stats(scores, pre, tol=0.99):
    """scores: morph per step after damage. Returns immediate/best/final
    and first step reaching tol*pre (None if never)."""
    t_rec = next((i for i, v in enumerate(scores) if v >= tol * pre), None)
    return {'pre': pre, 'post': scores[0], 'best': max(scores),
            'final': scores[-1], 'recovery_time': t_rec}
