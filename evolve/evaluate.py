# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Fitness evaluation through the full Morpho pipeline: every genome is
traced into a cyclic gate graph by compile_seq and simulated bit-exactly.
Test cases ride the simulator's vectorized samples axis, so one compile and
one run cover the whole IC batch."""

import numpy as np

from tiny_morpho_seq import compile_seq, eca_reference
from .genome import genome_to_cell


def build_sim(genome):
    return compile_seq(genome_to_cell(genome))

def run_trace(genome, ics, step_n):
    return build_sim(genome).run(step_n, state0=ics, samples=ics.shape[1])

def evaluate(genome, ics, step_n, score):
    return score(run_trace(genome, ics, step_n), ics)

def oracle_check(genome, ics, step_n):
    """Assert the compiled circuit matches the pure-numpy CA oracle."""
    trace = run_trace(genome, ics, step_n)
    ref = eca_reference(np.asarray(genome, dtype=np.int64), ics, step_n)
    assert (trace == ref).all(), "Morpho pipeline diverged from numpy oracle"
