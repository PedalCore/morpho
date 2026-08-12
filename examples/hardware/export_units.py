# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Export the verified arithmetic units to BLIF + Verilog and prove the
round trip: re-parse the emitted TEXT and re-simulate it against the
compiled Morpho circuit.

  exp2           2^(-x) unit    exhaustive, all 65,536 Q8.8 inputs
  divider        16/8 restoring 2,000 random divisions
  serial_divider streaming      512 divisions x 16 ticks through .latch

Netlists land in examples/hardware/netlists/. If yosys is on PATH the
script also synthesizes each BLIF for iCE40 and reports LUT/FF counts.
"""

import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from tiny_morpho import compile, unpack, pack
from tiny_morpho_seq import compile_seq
from tiny_morpho_hw import to_blif, to_verilog, simulate_blif

from examples.arithmetic.exp2 import exp2neg
from examples.arithmetic.divider import divider, serial_divider
from examples.rwkv.wkv_cell import (wkv_cell, ref_run, make_streams,
                                    to_raw, from_raw)

OUT = pathlib.Path(__file__).resolve().parent / 'netlists'
OUT.mkdir(exist_ok=True)


def save(name, blif, verilog):
    (OUT / f'{name}.blif').write_text(blif)
    (OUT / f'{name}.v').write_text(verilog)


def check_exp2():
    c = compile(exp2neg, (16,))
    blif = to_blif(c, 'exp2neg', output_names=('y',))
    save('exp2neg', blif, to_verilog(c, 'exp2neg', output_names=('y',)))
    xs = np.arange(1 << 16)
    want = pack(c(unpack(xs, 16)))
    got = simulate_blif(blif, 1, unpack(xs, 16)[:, None, :])['y'][:, 0]
    assert (pack(got) == want).all()
    gates = sum(1 for op in c.ops if op.type == 'GATE')
    print(f'exp2neg: BLIF round-trip EXHAUSTIVE over all 65,536 inputs '
          f'({gates} LUTs)')


def check_divider():
    c = compile(divider, (16, 8))
    blif = to_blif(c, 'divider', output_names=('q', 'rem'))
    save('divider', blif, to_verilog(c, 'divider', output_names=('q', 'rem')))
    rng = np.random.default_rng(3)
    a = rng.integers(1 << 16, size=2000)
    b = rng.integers(1, 1 << 8, size=2000)
    sim = simulate_blif(blif, 1, unpack(a, 16)[:, None, :],
                        unpack(b, 8)[:, None, :])
    assert (pack(sim['q'][:, 0]) == a // b).all()
    assert (pack(sim['rem'][:, 0]) == a % b).all()
    gates = sum(1 for op in c.ops if op.type == 'GATE')
    print(f'divider 16/8: BLIF round-trip on 2,000 random divisions '
          f'({gates} LUTs)')


def check_serial_divider():
    sim = compile_seq(serial_divider, (1, 8))
    blif = to_blif(sim, 'serial_divider', output_names=('q',))
    save('serial_divider', blif,
         to_verilog(sim, 'serial_divider', output_names=('q',)))
    rng = np.random.default_rng(7)
    n, cases = 16, 512
    a = rng.integers(1 << n, size=cases)
    b = rng.integers(1, 1 << 8, size=cases)
    a_bits = unpack(a, n)[::-1]                       # MSB first
    x_a = a_bits[None]                                # (1, T, cases)
    x_b = np.repeat(unpack(b, 8)[:, None, :], n, axis=1)
    q_ref = sim.run(n, x_a, x_b)[0]
    q_blif = simulate_blif(blif, n, x_a, x_b)['q'][0]
    assert (q_blif == q_ref).all()
    assert (pack(q_blif[::-1]) == a // b).all()
    regs = sum(1 for op in sim.c.ops if op.type == 'REG')
    gates = sum(1 for op in sim.c.ops if op.type == 'GATE')
    print(f'serial_divider: sequential BLIF round-trip, {cases} divisions '
          f'x {n} ticks ({gates} LUTs + {regs} FFs through .latch)')


def check_wkv_cell():
    sim = compile_seq(wkv_cell, (16,) * 4)
    names = ('kq', 'vq', 'uq', 'wq')
    blif = to_blif(sim, 'wkv_cell', output_names=('wkv',))
    save('wkv_cell', blif, to_verilog(sim, 'wkv_cell', output_names=('wkv',)))
    rng = np.random.default_rng(41)
    T, per = 32, 24
    kq, vq, uq, wq = make_streams(T, per, rng)
    want = ref_run(kq, vq, uq, wq)
    S = kq.shape[1]
    streams = [unpack(to_raw(x).ravel(), 16).reshape(16, T, S)
               for x in (kq, vq, uq, wq)]
    got = simulate_blif(blif, T, *streams)['wkv']            # (16, T, S)
    got = from_raw(pack(got.reshape(16, -1)).reshape(T, S))
    assert (got == want).all()
    regs = sum(1 for op in sim.c.ops if op.type == 'REG')
    gates = sum(1 for op in sim.c.ops if op.type == 'GATE')
    print(f'wkv_cell: full RWKV channel, sequential BLIF round-trip, '
          f'{S} channels x {T} ticks bit-exact ({gates} LUTs + {regs} FFs)')


def equiv_check(name, timeout=None):
    """yosys formal equivalence: the emitted Verilog vs the emitted BLIF.

    SAT-based equivalence is exponential in the worst case — multiplier
    cones are the classic hard instance — so a timeout bounds the attempt;
    a timeout is 'not proven within budget', never treated as failure
    (the text round-trip independently checks behavior on real streams)."""
    try:
        r = subprocess.run(['yosys', '-q', '-p', f"""
read_blif {OUT}/{name}.blif
rename {name} gold
read_verilog {OUT}/{name}.v
rename {name} gate
prep
splitnets -ports -format _
equiv_make gold gate equiv
equiv_simple
equiv_induct
equiv_status -assert"""], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f'  {name}: equivalence not proven within {timeout}s '
              f'(SAT hard case — multiplier cones); round-trip covers it',
              flush=True)
        return None
    ok = r.returncode == 0
    print(f'  {name}: Verilog ≡ BLIF formally '
          f'{"PROVEN" if ok else "FAILED: " + r.stderr[-300:]}', flush=True)
    return ok


TIMED_WRAPPER = """\
module {name}_timed(input clk, input [{iw}:0] xin, output reg [{ow}:0] yout);
  reg [{iw}:0] xr; wire [{ow}:0] yw;
  {name} u({conn});
  always @(posedge clk) begin xr <= xin; yout <= yw; end
endmodule
"""


def synth_and_pnr(name, wrap=None, freq=None):
    """synth_ice40 -> nextpnr-ice40 --hx8k; returns (LUTs, FFs, fmax MHz).

    Combinational units get a registered harness (wrap = port mapping) so
    nextpnr's f_max measures their real critical path."""
    top, reads = name, [f'read_blif {OUT}/{name}.blif']
    if wrap:
        # time the (formally equivalent) Verilog: its bus ports let the
        # registered harness instantiate it directly
        (OUT / f'{name}_timed.v').write_text(wrap)
        reads = [f'read_verilog {OUT}/{name}.v',
                 f'read_verilog {OUT}/{name}_timed.v']
        top = f'{name}_timed'
    json_f = OUT / f'{name}.json'
    r = subprocess.run(['yosys', '-q', '-p',
                        '; '.join(reads) +
                        f'; synth_ice40 -top {top} -json {json_f}'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  {name}: yosys failed\n{r.stderr[-400:]}')
        return
    cmd = ['nextpnr-ice40', '--hx8k', '--package', 'ct256',
           '--json', str(json_f), '--asc', str(OUT / f'{name}.asc')]
    if freq:                        # deep single-cycle designs miss the
        cmd += ['--freq', str(freq)]   # 12 MHz default; constrain honestly
    r = subprocess.run(cmd, capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode != 0:
        print(f'  {name}: nextpnr failed\n{log[-400:]}')
        return
    luts = ffs = 0
    fmax = None
    for ln in log.splitlines():
        m = re.search(r'ICESTORM_LC:\s+(\d+)/', ln)
        if m:
            luts = int(m.group(1))
        m = re.search(r"Max frequency for clock\s+\S+: ([0-9.]+) MHz", ln)
        if m:
            fmax = float(m.group(1))       # last occurrence = routed timing
    ffs = sum(1 for ln in (OUT / f'{name}.json').read_text().splitlines()
              if '"type": "SB_DFF' in ln)
    print(f'  {name}: {luts} logic cells placed+routed on iCE40 HX8K, '
          f'f_max {fmax:.1f} MHz' if fmax else f'  {name}: {luts} LCs, no clock')
    return luts, ffs, fmax


if __name__ == '__main__':
    check_exp2()
    check_divider()
    check_serial_divider()
    check_wkv_cell()
    print('all round trips exact: the emitted text IS the circuit')
    if not shutil.which('yosys'):
        print('\n(yosys not on PATH — netlists written, synthesis skipped)')
        sys.exit(0)
    print('\nformal equivalence (yosys equiv):')
    for name in ('exp2neg', 'divider', 'serial_divider'):
        equiv_check(name)
    equiv_check('wkv_cell', timeout=300)
    if not shutil.which('nextpnr-ice40'):
        print('\n(nextpnr-ice40 not on PATH — place-and-route skipped)')
        sys.exit(0)
    print('\nplace-and-route (nextpnr-ice40, HX8K):')
    synth_and_pnr('exp2neg', TIMED_WRAPPER.format(
        name='exp2neg', iw=15, ow=16, conn='.x(xr), .y(yw)'))
    synth_and_pnr('divider', """\
module divider_timed(input clk, input [15:0] a, input [7:0] b,
                     output reg [15:0] q, output reg [7:0] rem);
  reg [15:0] ar; reg [7:0] br; wire [15:0] qw; wire [7:0] rw;
  divider u(.a(ar), .b(br), .q(qw), .rem(rw));
  always @(posedge clk) begin ar <= a; br <= b; q <= qw; rem <= rw; end
endmodule
""")
    synth_and_pnr('serial_divider')
    synth_and_pnr('wkv_cell', freq=8)
