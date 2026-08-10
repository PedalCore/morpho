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

def copy_case(rng, w, case_n, step_n, emission_mask=False):
    """Copy-after-delay, streaming semantics with exact reference dynamics.

    Two inputs: data d and cue c. A W-bit window continuously shifts d in.
    When c=1 at time t, the pre-shift window (the W data bits before t) is
    latched and emitted in presentation order over the next W steps; output
    is 0 when idle. A new cue re-latches and restarts emission. Size
    parameter = word length W; retention emerges from cue sparsity.

    With emission_mask=True the returned mask is the (step, case) boolean
    array of REFERENCE emission steps (phase > 0) — derived from the task
    generator's own state, never from any candidate circuit. Targets are
    identical in both modes.
    """
    mask_w = (1 << w) - 1
    d = rng.integers(2, size=(step_n, case_n), dtype=np.int64)
    c = (rng.random((step_n, case_n)) < 1 / (w + 3)).astype(np.int64)
    win = np.zeros(case_n, dtype=np.int64)
    buf = np.zeros(case_n, dtype=np.int64)
    ph = np.zeros(case_n, dtype=np.int64)
    target = np.zeros((step_n, case_n), dtype=np.int32)
    emit = np.zeros((step_n, case_n), dtype=bool)
    for t in range(step_n):
        target[t] = np.where(ph > 0, (buf >> np.maximum(ph - 1, 0)) & 1, 0)
        emit[t] = ph > 0
        buf = np.where(c[t] == 1, win, buf)
        ph = np.where(c[t] == 1, w, np.maximum(ph - 1, 0))
        win = ((win << 1) | d[t]) & mask_w
    x = np.stack([d, c]).astype(np.int32)
    return x, target, emit if emission_mask else np.arange(step_n) >= w

def copy_ref(w):
    """Reference FSM. State int = window | buffer << W | phase << 2W.
    Input int b: bit0 = data, bit1 = cue. Moore output: emitting bit of the
    buffer when phase > 0, else 0."""
    mask_w = (1 << w) - 1
    def step(r, b):
        win, buf, ph = r & mask_w, (r >> w) & mask_w, r >> (2 * w)
        if (b >> 1) & 1:
            buf, ph = win, w
        else:
            ph = max(ph - 1, 0)
        win = ((win << 1) | (b & 1)) & mask_w
        return win | (buf << w) | (ph << (2 * w))
    def out(r, b):
        buf, ph = (r >> w) & mask_w, r >> (2 * w)
        return (buf >> (ph - 1)) & 1 if ph > 0 else 0
    return {'state_n': 2 * w + max(1, (w + 1).bit_length()),
            'step': step, 'out': out}

# (case generator, reference FSM, input width)
TASKS1 = {'recall': (recall_case, recall_ref, 1),
          'parity': (parity_case, parity_ref, 1),
          'copy': (copy_case, copy_ref, 2)}

def score(y, target, mask):
    """Mean bit accuracy over valid timesteps. y: (T, S)."""
    return float((y == target)[mask].mean())
