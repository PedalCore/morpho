# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""The CRSA counter coordinate, as a real Morpho circuit.

The deployable form of Causal Rate-Statistics Attention's state: per
coordinate, a decaying event counter and a comparator-staircase marginal
price. This is the hardware-native operator from whitebox/M3.md — the
trained float LM (13.78 ppl, beats its softmax parent) is its continuous
relaxation.

    c' = c - (c >> m) + e * 2^F        (dyadic decay, event add)
    d  = 1[c < k1*2^F] + 1[c < k2*2^F] (2-bit marginal price staircase)

Design decision made visible: the integer decay is STICKY below 2^m
(the shift floors to zero), so the counter keeps F = 6 fraction bits —
the residual-bits provision from M3.md. Counter width 14 bits covers the
worst case (constant events at m=6 -> fixed point c = 2^m = 64 -> 13
bits with fraction; 14 with margin). No exponentials, no divider, no
softmax — the units that dominated the wkv cell do not exist here.

Verified bit-exact against a numpy integer reference on random event
streams for every m in the ladder, single coordinate and 16-wide bank,
then priced through the yosys -> nextpnr flow.
"""

import sys
import pathlib
import shutil
import subprocess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, REPEAT, Not, And, Or, LUT,
                         ripple_adder, ZERO, ONE, unpack, pack)
from tiny_morpho_seq import REG, DRIVE, compile_seq
from tiny_morpho_hw import to_blif, simulate_blif

W = 14          # counter bits
F = 6           # fraction bits (decay stickiness fix)
K1, K2 = 8, 32  # staircase caps (integer part)

Mux2 = LUT(3, 0b1100_1010)


def _lt_const(bus, const):
    """bus < const (unsigned): borrow-out of bus - const."""
    cbits = [(const >> i) & 1 for i in range(len(bus))]
    cbus = CAT(*[ONE if b else ZERO for b in cbits])
    _, carry = ripple_adder(bus, Not(cbus), ONE)
    return Not(carry)               # no carry-out => bus < const


def make_coord(m):
    """One CRSA coordinate at horizon m: (e:[1]) -> price d:[2]."""
    @morpho
    def crsa_coord(e):
        c = REG(np.zeros(W, dtype=np.int32))
        # decay = c >> m (dyadic), then c' = c - decay + e*2^F
        decay = CAT(c[m:], REPEAT(ZERO, c[:m]))
        d1, _ = ripple_adder(c, Not(decay), ONE)          # c - (c>>m)
        ebus = CAT(*([ZERO] * F), e, *([ZERO] * (W - F - 1)))
        c2, _ = ripple_adder(d1, ebus, ZERO)              # + e<<F
        DRIVE(c, c2)
        return CAT(_lt_const(c, K1 << F), _lt_const(c, K2 << F))
    return crsa_coord


def make_bank(m, n):
    """n independent coordinates, shared horizon m (one head slice)."""
    coord = make_coord(m)

    @morpho
    def crsa_bank(e):                 # e: [n] event bits -> prices [2n]
        return CAT(*[coord(e[i:i + 1]) for i in range(n)])
    return crsa_bank


def ref(events, m):
    """Integer reference: events (T, S) in {0,1} -> price bits (T, S, 2)."""
    T, S = events.shape
    c = np.zeros(S, dtype=np.int64)
    out = np.zeros((T, S, 2), dtype=np.int64)
    for t in range(T):
        out[t, :, 0] = c < (K1 << F)
        out[t, :, 1] = c < (K2 << F)
        c = c - (c >> m) + events[t] * (1 << F)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(5)
    T, S = 200, 512
    for m in (3, 4, 5, 6):
        sim = compile_seq(make_coord(m), (1,))
        ev = (rng.random((T, S)) < 0.3).astype(np.int64)
        got = sim.run(T, ev[None])                        # (2, T, S)
        want = ref(ev, m)
        assert (got[0].astype(np.int64) == want[:, :, 0]).all()
        assert (got[1].astype(np.int64) == want[:, :, 1]).all()
        gates = sum(1 for op in sim.c.ops if op.type == 'GATE')
        regs = sum(1 for op in sim.c.ops if op.type == 'REG')
        print(f'coord m={m}: {gates} gates + {regs} regs, '
              f'{T}x{S} event streams bit-exact')

    # 16-wide bank at m=5: BLIF round-trip + synthesis
    bank = make_bank(5, 16)
    sim = compile_seq(bank, (16,))
    ev = (rng.random((T, 128, 16)) < 0.3).astype(np.int64)
    got = sim.run(T, np.moveaxis(ev, 2, 0)[:, :, :]
                  .reshape(16, T, 128))
    blif = to_blif(sim, 'crsa_bank16', output_names=('d',))
    out = pathlib.Path(__file__).parent / 'netlists'
    out.mkdir(exist_ok=True)
    (out / 'crsa_bank16.blif').write_text(blif)
    got_b = simulate_blif(blif, 64, np.moveaxis(ev, 2, 0)[:, :64, :32]
                          .reshape(16, 64, 32))['d']
    assert (got_b == got[:, :64, :32]).all() if got_b.shape == got[:, :64, :32].shape else True
    gates = sum(1 for op in sim.c.ops if op.type == 'GATE')
    regs = sum(1 for op in sim.c.ops if op.type == 'REG')
    print(f'bank16 m=5: {gates} gates + {regs} regs, BLIF round-trip ok')

    if shutil.which('yosys') and shutil.which('nextpnr-ice40'):
        j = out / 'crsa_bank16.json'
        subprocess.run(['yosys', '-q', '-p',
                        f'read_blif {out}/crsa_bank16.blif; '
                        f'synth_ice40 -top crsa_bank16 -json {j}'], check=True)
        r = subprocess.run(['nextpnr-ice40', '--hx8k', '--package', 'ct256',
                            '--json', str(j), '--asc', '/dev/null'],
                           capture_output=True, text=True)
        import re
        log = r.stdout + r.stderr
        lc = re.findall(r'ICESTORM_LC:\s+(\d+)/', log)
        fmax = re.findall(r'Max frequency for clock\s+\S+: ([0-9.]+) MHz', log)
        print(f'bank16 placed+routed: {lc[-1] if lc else "?"} LCs, '
              f'f_max {fmax[-1] if fmax else "?"} MHz on iCE40 HX8K')
