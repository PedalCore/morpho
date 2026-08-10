# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Behavioural descriptors of simulated dynamics. Static (structural)
metrics live on SequentialCircuit.metrics(); everything here is computed
from space-time traces of shape (cells, steps[, samples])."""

import numpy as np

from tiny_morpho import pack


def _3d(trace):
    return trace if trace.ndim == 3 else trace[:, :, None]

def activity(trace):
    """Mean fraction of cells flipping per step (0 = frozen, 0.5 ~ chaotic)."""
    return float(np.abs(np.diff(_3d(trace), axis=1)).mean())

def cell_entropy(trace):
    """Mean binary entropy of each cell's on-fraction over time (bits)."""
    p = np.clip(_3d(trace).mean(1), 1e-9, 1 - 1e-9)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)).mean())

def transient_period(trace):
    """Detect the attractor per sample by first repeated global state."""
    trace = _3d(trace)
    n, t, s = trace.shape
    codes = pack(trace.reshape(n, -1)).reshape(t, s)
    trans, periods = [], []
    for k in range(s):
        seen = {}
        for step, code in enumerate(codes[:, k]):
            code = int(code)
            if code in seen:
                trans.append(seen[code])
                periods.append(step - seen[code])
                break
            seen[code] = step
    found = len(periods)
    return {'cycle_found_frac': found / s,
            'mean_transient': float(np.mean(trans)) if found else None,
            'mean_period': float(np.mean(periods)) if found else None}

def damage_spread(run_trace, genome, ics, step_n, rng):
    """Flip one random cell per IC and measure final Hamming divergence per
    cell: ~0 = ordered, ~0.5 = chaotic; between = critical-ish."""
    flipped = ics.copy()
    flipped[rng.integers(ics.shape[0], size=ics.shape[1]),
            np.arange(ics.shape[1])] ^= 1
    a = run_trace(genome, ics, step_n)
    b = run_trace(genome, flipped, step_n)
    return float(np.abs(_3d(a)[:, -1] - _3d(b)[:, -1]).mean())

def describe(trace):
    return {'activity': activity(trace), 'cell_entropy': cell_entropy(trace),
            **transient_period(trace)}
