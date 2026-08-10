# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Temporal tasks over binary streams. Each task, given lag parameter d,
returns (input stream (x_n, T, S), target (T, S), valid mask (T,)) and has a
known state lower bound and a reference finite-state machine used for exact
product verification.

  recall  y[t] = x[t-d]                      lower bound: d state bits
  parity  y[t] = x[t] ^ x[t-1] ^ .. ^ x[t-d] lower bound: d state bits
"""

import numpy as np


def recall_case(rng, d, case_n, step_n):
    x = rng.integers(2, size=(1, step_n, case_n), dtype=np.int32)
    target = np.zeros((step_n, case_n), dtype=np.int32)
    target[d:] = x[0, :-d] if d else x[0]
    mask = np.arange(step_n) >= d
    return x, target, mask

def recall_ref(d):
    """Reference FSM: state = last d inputs; Mealy output ignores current x."""
    return {'state_n': d,
            'step': lambda r, b: ((r << 1) | b) & ((1 << d) - 1),
            'out': lambda r, b: (r >> (d - 1)) & 1}

def parity_case(rng, d, case_n, step_n):
    x = rng.integers(2, size=(1, step_n, case_n), dtype=np.int32)
    target = x[0].copy()
    for i in range(1, d + 1):
        target[i:] ^= x[0, :-i]
    mask = np.arange(step_n) >= d
    return x, target, mask

def parity_ref(d):
    return {'state_n': d,
            'step': lambda r, b: ((r << 1) | b) & ((1 << d) - 1),
            'out': lambda r, b: (bin(r).count('1') + b) & 1}

TASKS1 = {'recall': (recall_case, recall_ref),
          'parity': (parity_case, parity_ref)}

def score(y, target, mask):
    """Mean bit accuracy over valid timesteps. y: (T, S)."""
    return float((y == target)[mask].mean())
