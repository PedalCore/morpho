# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Typed genotypes that generate Morpho circuits.

Experiment 0 genotype: a fixed-size rule vector, one Wolfram rule per cell of
a non-uniform elementary CA. Deliberately minimal — genome size equals
phenotype size. Developmental (size-independent) genomes come later.
"""

import numpy as np

from tiny_morpho_seq import make_nonuniform_ca


def random_genome(rng, cell_n):
    return rng.integers(256, size=cell_n, dtype=np.uint8)

def genome_to_cell(genome):
    """genome -> @morpho cell definition (the genotype->phenotype map)."""
    return make_nonuniform_ca([int(r) for r in genome])
