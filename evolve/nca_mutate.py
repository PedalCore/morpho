# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Mutation operators over quantized NCA genomes. Mutation-only, no
crossover; all randomness flows through the caller's rng."""

import numpy as np

from .nca_genome import BIAS_MAX

_W, _B = ('w1', 'w2'), ('b1', 'b2')

def _pick(g, rng, keys):
    k = keys[rng.integers(len(keys))]
    flat = g[k].reshape(-1)
    return flat, rng.integers(flat.size)

def change_weight(g, rng):
    flat, i = _pick(g, rng, _W)
    flat[i] = rng.choice([-1, 0, 1])

def change_bias(g, rng):
    flat, i = _pick(g, rng, _B)
    flat[i] = np.clip(flat[i] + rng.choice([-1, 1]), -BIAS_MAX, BIAS_MAX)

def mutate_several(g, rng):
    for _ in range(rng.integers(2, 7)):
        change_weight(g, rng)

def zero_connection(g, rng):
    k = _W[rng.integers(2)]
    nz = np.nonzero(g[k].reshape(-1))[0]
    if len(nz):
        g[k].reshape(-1)[nz[rng.integers(len(nz))]] = 0

def activate_connection(g, rng):
    k = _W[rng.integers(2)]
    z = np.nonzero(g[k].reshape(-1) == 0)[0]
    if len(z):
        g[k].reshape(-1)[z[rng.integers(len(z))]] = rng.choice([-1, 1])

OPERATORS = [change_weight, change_bias, mutate_several,
             zero_connection, activate_connection]

def mutate(genome, rng, op_n=None):
    g = {k: v.copy() for k, v in genome.items()}
    for _ in range(op_n if op_n is not None else rng.integers(1, 4)):
        OPERATORS[rng.integers(len(OPERATORS))](g, rng)
    return g
