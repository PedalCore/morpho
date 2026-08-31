"""Substrate-agnostic stability metrics for event-structure
representations (offered as a shared standard on hi-sci-collab;
used in M10 v6/v7 and the Hopfield-diagnostics program, M8).

Works on any representation that exposes:
  - patterns: (N, D) array of stored/derived representation vectors
  - a similarity (default cosine)
  - optionally per-pattern consolidation weights (e.g., visit/win
    counts) and a time-series of query vectors.

Three metrics:
1. retrieval_margin: Delta_i = s(x_i, x_i-hat) - max_{j!=i}
   s(x_j, x_i-hat) — separation of the correct memory from its
   nearest competitor (Modern Hopfield, arXiv:2008.02217).
   Distinguishes erased / ambiguous / blended failure modes.
2. consolidation_weighted_recall: recall@1 where retrieval scores
   are weighted by consolidation depth g/(g+k) — measures whether
   persistent structure (not recency) drives recall.
3. disturbance_recovery: similarity-to-reference trajectory after a
   perturbation window — the curve, not an endpoint; monotone
   recovery indicates attractor behaviour.

Demo on Groove MIDI slot memory: python3 -m whitebox.stability_metrics
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _cos(A, b):
    n = np.linalg.norm(A, axis=-1) * (np.linalg.norm(b) + 1e-12)
    return (A @ b) / (n + 1e-12)


def retrieval_margin(patterns, cues, targets=None):
    """patterns (N,D); cues (M,D) noisy/partial versions;
    targets (M,) index of the correct pattern (default: identity).
    Returns per-cue margins Delta_i (positive = separable) and the
    retrieval entropy of the softmax score distribution."""
    patterns = np.asarray(patterns, float)
    cues = np.asarray(cues, float)
    if targets is None:
        targets = np.arange(len(cues))
    margins, ents = [], []
    for c, t in zip(cues, targets):
        s = _cos(patterns, c)
        correct = s[t]
        s2 = np.delete(s, t)
        margins.append(float(correct - s2.max()) if len(s2) else 1.0)
        p = np.exp(s - s.max())
        p /= p.sum()
        ents.append(float(-(p * np.log2(p + 1e-12)).sum()))
    return np.array(margins), np.array(ents)


def consolidation_weighted_recall(patterns, weights, cues, targets,
                                  k=2.0):
    """recall@1 with scores s * g/(g+k): does persistent (deeply
    consolidated) structure drive retrieval rather than recency?"""
    patterns = np.asarray(patterns, float)
    w = np.asarray(weights, float)
    hits = 0
    for c, t in zip(np.asarray(cues, float), targets):
        s = _cos(patterns, c) * (w / (w + k))
        hits += int(np.argmax(s) == t)
    return hits / max(len(targets), 1)


def disturbance_recovery(sim_to_ref):
    """sim_to_ref: per-step (or per-bar) similarity to the reference
    structure, beginning at the disturbance. Returns (final - first),
    the fitted slope, and whether the trajectory is monotone-
    increasing under a 1-step tolerance — attractor signature."""
    s = np.asarray(sim_to_ref, float)
    slope = float(np.polyfit(np.arange(len(s)), s, 1)[0])
    mono = bool(np.all(np.diff(s) > -0.05))
    return dict(delta=float(s[-1] - s[0]), slope=slope,
                monotone=mono, trajectory=[round(float(v), 3)
                                           for v in s])


def _demo():
    """Run all three on the M10 slot memory over GMD bars."""
    from whitebox.rhythm3_exact import Config, load_store
    from whitebox.rhythm5_modes import bar_grids
    from whitebox.rhythm6_motif import MotifStore
    cfg = Config()
    store_d = load_store()
    ms = MotifStore()
    grids = []
    for it in store_d['train'][:40]:
        for g in bar_grids(it['x'], cfg):
            if g.sum():
                ms.store(g)
                grids.append(g)
    live = ms.g > 0
    patterns = ms.W[live]
    weights = ms.g[live]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(patterns), size=min(64, len(patterns)),
                     replace=False)
    cues = patterns[idx] * (rng.random(patterns[idx].shape) > 0.3)
    m, e = retrieval_margin(patterns, cues, idx)
    r = consolidation_weighted_recall(patterns, weights, cues, idx)
    print(f'slots {len(patterns)} | margin median {np.median(m):+.3f} '
          f'(frac>0 {np.mean(m > 0):.2f}) | retrieval entropy '
          f'{np.median(e):.2f} bits | consol-weighted recall@1 {r:.2f}')
    print('disturbance_recovery on the published motif-return '
          'trajectory:',
          disturbance_recovery([.07, .24, .35, .42, .35, .37, .44,
                                .53]))


if __name__ == '__main__':
    _demo()
