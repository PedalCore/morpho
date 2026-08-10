# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Checkerboard morphogenesis task: seed, scale-independent target, damage.

The target phase is defined relative to the seed cell, so an exact target
exists at every lattice size (including non-multiples of training sizes).
Damage operators clear ALL channels (visible + latent) of affected cells —
never only the phenotype — so repair cannot lean on hidden survivors."""

import numpy as np

from .nca_genome import C


def seed_state(ny, nx):
    """All channels zero except the central cell's visible bit."""
    s = np.zeros((C, ny, nx), dtype=np.int16)
    s[0, ny // 2, nx // 2] = 1
    return s

def checkerboard(ny, nx):
    """Target visible pattern, phase anchored at the seed cell (active)."""
    yy, xx = np.mgrid[:ny, :nx]
    return (((yy + xx) - (ny // 2 + nx // 2)) % 2 == 0).astype(np.int16)

def rings(ny, nx):
    """Concentric-square-rings target, phase anchored at the boundary
    (outermost ring active). Defined at every size, including odd."""
    yy, xx = np.mgrid[:ny, :nx]
    bdist = np.minimum(np.minimum(yy, ny - 1 - yy),
                       np.minimum(xx, nx - 1 - xx))
    return (bdist % 2 == 0).astype(np.int16)


def damage_random(s, frac, rng):
    """Clear all channels of a random `frac` of cells."""
    ny, nx = s.shape[1:3]
    hit = rng.random((ny, nx)) < frac
    out = s.copy()
    out[:, hit] = 0
    return out

def damage_wound(s, area_frac, rng):
    """Clear all channels in a random square covering ~area_frac of cells."""
    ny, nx = s.shape[1:3]
    side = max(2, int(round(np.sqrt(area_frac * ny * nx))))
    y0 = rng.integers(0, max(1, ny - side + 1))
    x0 = rng.integers(0, max(1, nx - side + 1))
    out = s.copy()
    out[:, y0:y0 + side, x0:x0 + side] = 0
    return out

DAMAGE_OPS = {'del10': lambda s, r: damage_random(s, .10, r),
              'del25': lambda s, r: damage_random(s, .25, r),
              'del40': lambda s, r: damage_random(s, .40, r),
              'wound25': lambda s, r: damage_wound(s, .25, r)}
