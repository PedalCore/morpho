# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Mutation operators for recurrent-network genomes. Mutation-only, no
crossover. Structural add/delete emerges from rewiring plus DCE: pointing a
drive/output at a dormant LUT activates it; rewiring away deletes it."""

import numpy as np

from .recurrent_genome import sig_base


def flip_lut_bit(g, rng, cfg):
    j = rng.integers(len(g['luts']))
    g['luts'][j] ^= np.uint8(1 << rng.integers(1 << int(g['arity'][j])))

def replace_lut(g, rng, cfg):
    g['luts'][rng.integers(len(g['luts']))] = rng.integers(256)

def change_arity(g, rng, cfg):
    g['arity'][rng.integers(len(g['arity']))] = rng.integers(1, 4)

def rewire_node(g, rng, cfg):
    j = rng.integers(len(g['refs']))
    g['refs'][j][rng.integers(3)] = rng.integers(sig_base(cfg) + j)

def rewire_drive(g, rng, cfg):
    g['drives'][rng.integers(len(g['drives']))] = \
        rng.integers(sig_base(cfg) + cfg['node_n'])

def rewire_out(g, rng, cfg):
    g['outs'][rng.integers(len(g['outs']))] = \
        rng.integers(sig_base(cfg) + cfg['node_n'])

OPERATORS = [flip_lut_bit, replace_lut, change_arity, rewire_node,
             rewire_drive, rewire_out]

def mutate(genome, rng, cfg, op_n=None):
    g = {k: v.copy() for k, v in genome.items()}
    for _ in range(op_n if op_n is not None else rng.integers(1, 4)):
        OPERATORS[rng.integers(len(OPERATORS))](g, rng, cfg)
    return g
