# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Count the model's datapath gates by CONSTRUCTION, not arithmetic.

Builds the actual matmul engines — one MAC lane per output row, shared
activation broadcast, per-lane weights, exactly the organization the
silicon page assumes — as real compiled Morpho circuits at FULL width,
for the dense model and both spiking variants, and counts gates in the
emitted netlists. Where a full-width build is confirmed gate-exact
linear in lane count (it is verified below, not assumed), larger engines
reuse the measured per-lane increment; every such entry is labeled.

Engines (one RWKV block, one token per ~1536 cycles):
  time-mix   4 matmuls x 384 lanes, dense always (spikes don't reach it)
  wkv        1 time-multiplexed cell   (measured: wkv_cell.py, 7,201 gates)
  cm key     1536 lanes: dense | ternary-spike input (fully-spiking variant)
  cm value    384 lanes: dense | spike-4 | binary
  cm recept   384 lanes, dense always
  activation  1 time-multiplexed unit per block (relu^2 | threshold)
Head: 4096 dense lanes (shared across nothing; counted once).

Lane cells are verified bit-exact against numpy before counting.
"""

import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import (morpho, CAT, REPEAT, Not, And, Or, Xor, LUT,
                         ripple_adder, ZERO, ONE, unpack, pack, compile)
from examples.rwkv.spike_mac import (mac_dense, mac_spike4, mac_binary,
                                     act_relusq, act_spike4, _sext)

Mux2 = LUT(3, 0b1100_1010)


@morpho
def mac_ternary(w, s, f, acc):      # acc += f ? (s ? -w : w) : 0
    t = Xor(_sext(w, acc), REPEAT(s, acc))     # conditional one's complement
    tm = And(t, REPEAT(f, acc))                # gate on fire
    out, _ = ripple_adder(acc, tm, And(s, f))  # +1 completes the negate
    return out


def check_ternary(cases=200000):
    rng = np.random.default_rng(3)
    w = rng.integers(-128, 128, cases)
    s = rng.integers(0, 2, cases)
    f = rng.integers(0, 2, cases)
    acc = rng.integers(0, 1 << 24, cases)
    c = compile(mac_ternary, (8, 1, 1, 24))
    got = pack(c(unpack(w & 0xFF, 8), unpack(s, 1), unpack(f, 1),
                 unpack(acc, 24)))
    want = (acc + f * np.where(s == 1, -w, w)) & 0xFFFFFF
    assert (got == want).all()
    return c


LANES = {
    'dense': (mac_dense, (8, 8)),        # (cell, (w_bits, x_bits))
    'spike4': (mac_spike4, (8, 3)),
    'binary': (mac_binary, (8, 1)),
    'ternary': (mac_ternary, (8, 2)),    # x = (s, f)
}


def engine(kind, lanes):
    """A real multi-lane engine: shared activation x, per-lane weight w_i,
    per-lane accumulator; returns the compiled circuit."""
    cell, (wb, xb) = LANES[kind]

    @morpho
    def eng(x, w, acc):                  # w: [wb*lanes], acc: [24*lanes]
        outs = []
        for i in range(lanes):
            wi = w[i * wb:(i + 1) * wb]
            ai = acc[i * 24:(i + 1) * 24]
            if kind == 'ternary':
                outs.append(cell(wi, x[0:1], x[1:2], ai))
            else:
                outs.append(cell(wi, x, ai))
        return CAT(*outs)

    return compile(eng, (xb, wb * lanes, 24 * lanes))


def gates(c):
    return sum(1 for op in c.ops if op.type == 'GATE')


if __name__ == '__main__':
    check_ternary()
    print('mac_ternary verified bit-exact (200,000 cases)\n')

    # ---- 1. linearity: is engine(k) exactly k * increment + const? ----
    print('linearity check (gates at k lanes):')
    inc = {}
    for kind in LANES:
        counts = {}
        for k in (1, 2, 4, 8, 16):
            counts[k] = gates(engine(kind, k))
        d = {k: counts[k * 2] - counts[k] for k in (1, 2, 4, 8)}
        per = d[8] // 8
        exact = all(d[k] == per * k for k in d)
        inc[kind] = per
        print(f'  {kind:8s} k=1..16: {list(counts.values())}  '
              f'per-lane {per}  exactly linear: {exact}')
        assert exact, f'{kind}: engine not gate-linear in lanes'

    # ---- 2. full-width builds where feasible ----
    print('\nfull-width engines (really constructed):')
    full = {}
    for name, kind, lanes in [
        ('cm value, dense', 'dense', 384),
        ('cm value, spike-4', 'spike4', 384),
        ('cm value, binary', 'binary', 384),
        ('cm key, ternary', 'ternary', 1536),
    ]:
        t0 = time.time()
        c = engine(kind, lanes)
        g = gates(c)
        full[(kind, lanes)] = g
        print(f'  {name:22s} {lanes:5d} lanes  {g:9,d} gates  '
              f'({time.time() - t0:.0f}s to compile)')

    # dense at 1536: build if the 384 build was quick, else linear-verified
    t0 = time.time()
    try:
        g1536 = gates(engine('dense', 1536))
        label1536 = 'built'
    except MemoryError:
        g1536 = None
    if g1536 is None:
        g1536 = full[('dense', 384)] * 4
        label1536 = 'linear-verified x4'
    print(f'  {"cm key, dense":22s} {1536:5d} lanes  {g1536:9,d} gates  '
          f'({label1536}, {time.time() - t0:.0f}s)')

    # ---- 3. assemble block + model ----
    WKV = 7201                     # measured, examples/rwkv/wkv_cell.py
    ACT_D = 890                    # measured, spike_mac.py
    ACT_S = 15
    d384 = full[('dense', 384)]
    tm = g1536                     # 4 x 384 dense lanes == 1536 dense lanes
    head = g1536 * 4096 // 1536    # linear-verified (dense increment)

    def block(value_kind, key_kind, act):
        key = g1536 if key_kind == 'dense' else full[('ternary', 1536)]
        return tm + WKV + key + full[(value_kind, 384)] + d384 + act

    rows = [
        ('all dense (float model)', block('dense', 'dense', ACT_D), 'dense'),
        ('current spiking (spike-4 -> Wv)', block('spike4', 'dense', ACT_S), None),
        ('current at binary (binary -> Wv)', block('binary', 'dense', ACT_S), None),
        ('fully-spiking, binary (-> Wk too)', block('binary', 'ternary', ACT_S), None),
    ]
    print('\nper-block datapath (constructed engines + measured wkv cell):')
    base = rows[0][1]
    for name, g, _ in rows:
        print(f'  {name:36s} {g:9,d} gates  '
              f'{"" if g == base else f"(-{100 * (1 - g / base):.1f}%)"}')

    print(f'\nwhole model = 6 blocks + head ({head:,} gates, linear-verified):')
    for name, g, _ in rows:
        total = 6 * g + head
        print(f'  {name:36s} {total:10,d} gates  '
              f'{"" if g == base else f"(-{100 * (1 - (6 * g + head) / (6 * base + head)):.1f}%)"}')
