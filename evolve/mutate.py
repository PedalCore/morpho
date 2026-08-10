# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Mutation operators over rule-vector genomes. All operate in place on a
copy made by mutate(); all randomness flows through the caller's rng."""

import numpy as np


def replace_rule(g, rng):
    g[rng.integers(len(g))] = rng.integers(256)

def flip_rule_bit(g, rng):
    g[rng.integers(len(g))] ^= np.uint8(1 << rng.integers(8))

def copy_neighbor(g, rng):
    i = rng.integers(len(g))
    g[i] = g[(i + (1 if rng.random() < .5 else -1)) % len(g)]

def swap_rules(g, rng):
    i, j = rng.integers(len(g), size=2)
    g[i], g[j] = g[j], g[i]

def duplicate_segment(g, rng):
    n = len(g)
    k = rng.integers(1, max(2, n // 4) + 1)
    src, dst = rng.integers(n, size=2)
    g[(dst + np.arange(k)) % n] = g[(src + np.arange(k)) % n]

OPERATORS = [replace_rule, flip_rule_bit, copy_neighbor, swap_rules,
             duplicate_segment]

def mutate(genome, rng, op_n=None):
    """Apply 1..3 random operators (or exactly op_n) to a copy of genome."""
    g = genome.copy()
    for _ in range(op_n if op_n is not None else rng.integers(1, 4)):
        OPERATORS[rng.integers(len(OPERATORS))](g, rng)
    return g
