# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""One diagonal-delta memory state element, as a real Morpho circuit —
the hardware costing owed by whitebox/M5.md (the 60-gate counter story
does not transfer to fast-weight memory; this measures what does).

The M5 operator's per-element update (Longhorn-form diagonal):

    S <- S - (u * S >> F) + w        u = clamp(eps*k_j^2), w = eps*v_i*k_j

Two cells, one design axis made visible:

  delta_cell_mul   — u is a DYNAMIC 8-bit input: needs a real 8x16
                     multiplier per element. The faithful cell.
  delta_cell_shift — u is LADDER-QUANTIZED to a fixed power of two per
                     cell instance (CRSA's dyadic trick applied to the
                     learning rate): decay = arithmetic shift by a
                     constant, no multiplier. Whether training survives
                     this quantization is a future probe (M2-style
                     calibration); the circuit prices the incentive.

Both cells are scoped to the STATE UPDATE (the read y = S q is a MAC
shared across elements, amortized as in the spike_mac units). w arrives
as a signed Q7.8 bus from the upstream write MAC in both cells — the
same scoping as the CRSA coordinate taking event bits from upstream.

State: signed Q7.8, 16 bits. Verified bit-exact against numpy integer
references on random streams, then priced through yosys/nextpnr.
"""

import sys
import pathlib
import shutil
import subprocess

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, REPEAT, Not, And,
                         ripple_adder, wallace_multiplier, ZERO, ONE)
from tiny_morpho_seq import REG, DRIVE, compile_seq
from tiny_morpho_hw import to_blif

W = 16          # state bits, signed Q7.8
F = 8           # fraction bits


def _mul_uq8_s16(u, s):
    """floor(u * s / 2^F): u unsigned Q0.8 [8], s signed Q7.8 [16] ->
    signed Q7.8 [16]. mul_q16 pattern at these widths."""
    prod = wallace_multiplier(u, s)                  # 24 bits, s as unsigned
    corr = CAT(REPEAT(ZERO, s), And(u, REPEAT(s[W-1:W], u)))  # (u<<16)*sign
    d, _ = ripple_adder(prod, Not(corr), ONE)
    return d[F:F+W]


def delta_cell_mul():
    """Faithful element: dynamic learning rate u[8], write w[16]."""
    @morpho
    def cell(u, w):
        s = REG(np.zeros(W, dtype=np.int32))
        dec = _mul_uq8_s16(u, s)                     # (u*S)>>F, signed
        t1, _ = ripple_adder(s, Not(dec), ONE)       # S - dec
        t2, _ = ripple_adder(t1, w, ZERO)            # + w
        DRIVE(s, t2)
        return s                                     # pre-update read tap
    return cell


def delta_cell_shift(shift):
    """Ladder element: u fixed at 2^-shift per instance -> decay is an
    arithmetic shift by a constant. No multiplier anywhere."""
    @morpho
    def cell(w):
        s = REG(np.zeros(W, dtype=np.int32))
        sign = s[W-1:W]
        dec = CAT(s[shift:], REPEAT(sign, s[:shift]))   # S >>a shift
        t1, _ = ripple_adder(s, Not(dec), ONE)
        t2, _ = ripple_adder(t1, w, ZERO)
        DRIVE(s, t2)
        return s
    return cell


# ---- integer references (two's complement, W bits) ----

def _wrap(x):
    return ((x + (1 << (W-1))) % (1 << W)) - (1 << (W-1))


def ref_mul(us, ws):
    T = len(us)
    s = 0
    out = np.zeros(T, dtype=np.int64)
    for t in range(T):
        out[t] = s
        dec = (us[t] * s) >> F                      # python >> floors: matches
        s = _wrap(s - dec + ws[t])
    return out


def ref_shift(shift, ws):
    T = len(ws)
    s = 0
    out = np.zeros(T, dtype=np.int64)
    for t in range(T):
        out[t] = s
        s = _wrap(s - (s >> shift) + ws[t])
    return out


def _to_bits(vals, bits):
    v = np.asarray(vals, dtype=np.int64) & ((1 << bits) - 1)
    return np.stack([(v >> i) & 1 for i in range(bits)])


def _from_bits(bits_arr):
    n = bits_arr.shape[0]
    v = sum(bits_arr[i].astype(np.int64) << i for i in range(n))
    return ((v + (1 << (n-1))) % (1 << n)) - (1 << (n-1))


if __name__ == '__main__':
    rng = np.random.default_rng(9)
    T, NS = 200, 128

    # faithful cell
    sim = compile_seq(delta_cell_mul(), (8, W))
    us = rng.integers(0, 231, (T, NS))              # u <= 0.9 in Q0.8
    ws = rng.integers(-2000, 2000, (T, NS))
    ub = np.stack([_to_bits(us[:, i], 8) for i in range(NS)], axis=2)
    wb = np.stack([_to_bits(ws[:, i], W) for i in range(NS)], axis=2)
    got = sim.run(T, ub, wb)
    got_v = _from_bits(got)
    want = np.stack([ref_mul(us[:, i], ws[:, i]) for i in range(NS)], axis=1)
    assert (got_v == want).all(), 'mul cell mismatch'
    g = sum(1 for op in sim.c.ops if op.type == 'GATE')
    r = sum(1 for op in sim.c.ops if op.type == 'REG')
    print(f'delta_cell_mul  (dynamic u): {g} gates + {r} regs '
          f'({T}x{NS} streams bit-exact)')

    # ladder cell across the shift ladder
    for shift in (1, 2, 4, 6):
        sim = compile_seq(delta_cell_shift(shift), (W,))
        got = sim.run(T, wb)
        got_v = _from_bits(got)
        want = np.stack([ref_shift(shift, ws[:, i]) for i in range(NS)],
                        axis=1)
        assert (got_v == want).all(), f'shift cell mismatch s={shift}'
        g = sum(1 for op in sim.c.ops if op.type == 'GATE')
        r = sum(1 for op in sim.c.ops if op.type == 'REG')
        print(f'delta_cell_shift s={shift}: {g} gates + {r} regs')

    # 8-element bank of the ladder cell -> BLIF + synthesis
    cell = delta_cell_shift(4)

    @morpho
    def bank8(w):                                    # w: [8*W]
        return CAT(*[cell(w[i*W:(i+1)*W]) for i in range(8)])

    simb = compile_seq(bank8, (8 * W,))
    blif = to_blif(simb, 'delta_bank8', output_names=('s',))
    out = pathlib.Path(__file__).parent / 'netlists'
    out.mkdir(exist_ok=True)
    (out / 'delta_bank8.blif').write_text(blif)
    g = sum(1 for op in simb.c.ops if op.type == 'GATE')
    r = sum(1 for op in simb.c.ops if op.type == 'REG')
    print(f'bank8 (ladder): {g} gates + {r} regs, BLIF written')

    if shutil.which('yosys') and shutil.which('nextpnr-ice40'):
        j = out / 'delta_bank8.json'
        subprocess.run(['yosys', '-q', '-p',
                        f'read_blif {out}/delta_bank8.blif; '
                        f'synth_ice40 -top delta_bank8 -json {j}'],
                       check=True)
        res = subprocess.run(['nextpnr-ice40', '--hx8k', '--package',
                              'ct256', '--json', str(j), '--asc',
                              '/dev/null'], capture_output=True, text=True)
        import re
        log = res.stdout + res.stderr
        lc = re.findall(r'ICESTORM_LC:\s+(\d+)/', log)
        fmax = re.findall(r'Max frequency for clock\s+\S+: ([0-9.]+) MHz',
                          log)
        print(f'bank8 placed+routed: {lc[-1] if lc else "?"} LCs, '
              f'f_max {fmax[-1] if fmax else "?"} MHz on iCE40 HX8K')
