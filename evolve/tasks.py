# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""CA tasks: initial-condition generators and space-time trace scoring.

Every scorer returns (smooth, strict): a dense per-cell score for selection
gradient, and the strict fraction of cases classified perfectly.
"""

import numpy as np


def density_ics(rng, cell_n, case_n):
    """Initial densities sampled uniformly in [0,1] (the standard benchmark
    distribution; binomial sampling would concentrate cases near 0.5)."""
    dens = rng.random(case_n)
    return (rng.random((cell_n, case_n)) < dens).astype(np.int32)

def density_score(trace, ics):
    """Density classification: relax to all-ones iff the IC had majority ones.
    Requires an odd cell count so majority is well defined."""
    target = (ics.sum(0) > ics.shape[0] // 2).astype(np.int32)
    per_case = (trace[:, -1] == target).mean(0)
    return float(per_case.mean()), float((per_case == 1.0).mean())

def sync_ics(rng, cell_n, case_n):
    return rng.integers(2, size=(cell_n, case_n), dtype=np.int32)

def sync_score(trace, ics):
    """Synchronization: from any IC, reach the globally uniform blinking
    orbit (all-0 <-> all-1) by the end of the run."""
    a, b = trace[:, -2], trace[:, -1]
    phase = (b.mean(0) >= .5).astype(np.int32)
    per_case = ((b == phase) & (a == 1 - phase)).mean(0)
    return float(per_case.mean()), float((per_case == 1.0).mean())

TASKS = {'density': (density_ics, density_score),
         'sync': (sync_ics, sync_score)}
