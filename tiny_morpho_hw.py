# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Morpho -> hardware: mechanical netlist export.

What compile()/compile_seq() produce is already a hardware netlist, not a
description that needs elaborating into one: k-input LUTs (GATE ops with an
integer truth table), flip-flops (REG ops), two constants, wires, and one
implicit clock. An FPGA is a fabric of exactly those primitives, so the
export is a transcription, not a compilation:

    GATE(lut, args)  ->  a LUT with that exact truth table
    REG(init)        ->  a flip-flop with that init value
    synchronous commit -> the clock edge

Two formats, one function each:

    to_blif(circuit)      BLIF — LUTs are its native primitive; feeds the
                          open yosys -> nextpnr flow (iCE40/ECP5) directly
    to_verilog(circuit)   structural Verilog-2001 — one LUT constant + one
                          index per gate, one always block per register

Both accept a CircuitCompiler (combinational), a SequentialCompiler, or a
SequentialCircuit. The synthesizable subset is the REG-only discipline:
every feedback loop broken by a register, a single clock. FORWARD/TIE
fixed-point circuits (the SR-latch corner) are deliberately rejected —
they sit outside what synchronous synthesis tools guarantee anything about.

simulate_blif() is an independent check: it re-parses the emitted TEXT
(not the in-memory ops) and re-simulates it, so a round trip against the
compiled circuit validates the exporter end to end.
"""

import re

import numpy as np


# ---------------------------------------------------------------- netlist view

def _unwrap(circuit):
    """Accept CircuitCompiler / SequentialCompiler / SequentialCircuit."""
    return circuit.c if hasattr(circuit, 'c') and hasattr(circuit.c, 'ops') \
        else circuit


def _flat_outputs(c, output_names):
    """[(port_name, width, [op_idx, ...]), ...] little-endian bit order."""
    outs = c.outputs if isinstance(c.outputs, tuple) else (c.outputs,)
    groups = [np.atleast_1d(np.asarray(o)) for o in outs]
    if output_names is None:
        output_names = [f'y{k}' for k in range(len(groups))]
    if len(output_names) != len(groups):
        raise ValueError(f'{len(groups)} output groups, '
                         f'{len(output_names)} names given')
    return [(n, len(g), [int(i) for i in g])
            for n, g in zip(output_names, groups)]


def _check_synthesizable(ops):
    fwd = [op.name for op in ops if op.type == 'FWD']
    if fwd:
        raise ValueError(
            f'{len(fwd)} FORWARD wire(s) present ({fwd[:4]}...): '
            'combinational feedback is outside the synthesizable REG-only '
            'discipline. Break the loop with REG/DRIVE.')


def _wire_names(c):
    """One name per op index. Inputs keep their port names, bit-blasted."""
    names = {0: 'const0', 1: 'const1'}
    for (pname, n), idxs in zip(c.input_info, c.inputs):
        for b, i in enumerate(idxs):
            names[int(i)] = f'{pname}_{b}'
    for i, op in enumerate(c.ops):
        if i in names:
            continue
        names[i] = f'r{i}' if op.type == 'REG' else f'w{i}'
    return names


# ----------------------------------------------------------------------- BLIF

def to_blif(circuit, name='morpho', output_names=None):
    """BLIF netlist. Registers become `.latch d q re clk init`."""
    c = _unwrap(circuit)
    _check_synthesizable(c.ops)
    nm = _wire_names(c)
    outs = _flat_outputs(c, output_names)
    has_regs = any(op.type == 'REG' for op in c.ops)

    lines = [f'.model {name}']
    in_ports = [nm[int(i)] for idxs in c.inputs for i in idxs]
    if has_regs:
        in_ports = ['clk'] + in_ports
    lines.append('.inputs ' + ' '.join(in_ports))
    out_ports = []
    for pname, width, idxs in outs:
        for b, i in enumerate(idxs):
            out_ports.append(f'{pname}_{b}')
    lines.append('.outputs ' + ' '.join(out_ports))

    lines.append('.names const0')          # constant 0: empty cover
    lines.append('.names const1')          # constant 1
    lines.append('1')

    for i, op in enumerate(c.ops):
        if op.type == 'GATE':
            arity = len(op.args)
            lines.append('.names ' + ' '.join(nm[a] for a in op.args)
                         + ' ' + nm[i])
            for m in range(1 << arity):
                if (op.lut >> m) & 1:
                    lines.append(''.join(str((m >> k) & 1)
                                         for k in range(arity)) + ' 1')
        elif op.type == 'REG':
            d = int(op.args[0])
            lines.append(f'.latch {nm[d]} {nm[i]} re clk {int(op.lut)}')

    # output ports are aliases of internal wires
    for pname, width, idxs in outs:
        for b, i in enumerate(idxs):
            lines.append(f'.names {nm[i]} {pname}_{b}')
            lines.append('1 1')
    lines.append('.end')
    return '\n'.join(lines) + '\n'


# -------------------------------------------------------------------- Verilog

def to_verilog(circuit, name='morpho', output_names=None):
    """Structural Verilog-2001: buses on the ports, one LUT per gate."""
    c = _unwrap(circuit)
    _check_synthesizable(c.ops)
    nm = _wire_names(c)
    # internal alias wires must not collide with the names splitnets -ports
    # would give the bus bits (x[15] -> x_15), or equivalence checks against
    # the BLIF get renamed ports; use n<idx> internally instead
    for idxs in c.inputs:
        for i in idxs:
            nm[int(i)] = f'n{int(i)}'
    outs = _flat_outputs(c, output_names)
    has_regs = any(op.type == 'REG' for op in c.ops)

    ports, decls, body = [], [], []
    if has_regs:
        ports.append('clk')
        decls.append('  input clk;')
    for pname, n in c.input_info:
        ports.append(pname)
        decls.append(f'  input [{n - 1}:0] {pname};')
    for pname, width, _ in outs:
        ports.append(pname)
        decls.append(f'  output [{width - 1}:0] {pname};')

    body.append("  wire const0 = 1'b0;")
    body.append("  wire const1 = 1'b1;")
    for (pname, n), idxs in zip(c.input_info, c.inputs):
        for b, i in enumerate(idxs):
            body.append(f'  wire {nm[int(i)]} = {pname}[{b}];')

    for i, op in enumerate(c.ops):
        if op.type == 'GATE':
            arity = len(op.args)
            size = 1 << arity
            sel = '{' + ', '.join(nm[a] for a in reversed(op.args)) + '}'
            body.append(f"  wire [{size - 1}:0] l{i} = {size}'h{op.lut:x}; "
                        f'wire {nm[i]} = l{i}[{sel}];')
        elif op.type == 'REG':
            body.append(f"  reg {nm[i]} = 1'b{int(op.lut)};")
    for i, op in enumerate(c.ops):
        if op.type == 'REG':
            body.append(f'  always @(posedge clk) {nm[i]} <= '
                        f'{nm[int(op.args[0])]};')

    for pname, width, idxs in outs:
        for b, i in enumerate(idxs):
            body.append(f'  assign {pname}[{b}] = {nm[int(i)]};')

    return (f'module {name}(' + ', '.join(ports) + ');\n'
            + '\n'.join(decls) + '\n\n'
            + '\n'.join(body) + '\nendmodule\n')


# --------------------------------------------------- independent BLIF re-sim

def simulate_blif(blif, step_n, *streams, samples=None):
    """Parse a BLIF netlist (as text) and simulate it, vectorized.

    Input streams follow SequentialCircuit.run's convention: one array per
    input port group IN DECLARATION ORDER, shape (width, step_n[, samples]),
    little-endian bit 0 first. Returns {port_prefix: (width, step_n, samples)
    array}. Independent of the compiler's data structures on purpose: this
    reads only the text, so a round trip validates the exporter."""
    names_re = re.compile(r'^\.names\s+(.*)$')
    latch_re = re.compile(r'^\.latch\s+(\S+)\s+(\S+)\s+re\s+clk\s+([01])$')
    lines = [ln.strip() for ln in blif.splitlines() if ln.strip()]

    inputs, outputs, gates, latches = [], [], [], []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('.inputs'):
            inputs = ln.split()[1:]
        elif ln.startswith('.outputs'):
            outputs = ln.split()[1:]
        elif ln.startswith('.latch'):
            m = latch_re.match(ln)
            if not m:
                raise ValueError('unsupported latch line: ' + ln)
            latches.append((m.group(1), m.group(2), int(m.group(3))))
        elif ln.startswith('.names'):
            sig = names_re.match(ln).group(1).split()
            args, out = sig[:-1], sig[-1]
            i += 1
            lut = 0
            while i < len(lines) and not lines[i].startswith('.'):
                row = lines[i].split()
                if args:
                    bits, val = row[0], row[1]
                    if not set(bits) <= {'0', '1'}:
                        raise ValueError("don't-cares not emitted by exporter")
                    if val == '1':
                        lut |= 1 << sum((bits[k] == '1') << k
                                        for k in range(len(bits)))
                else:
                    lut = int(row[0])            # constant 1
                i += 1
            gates.append((args, out, lut))
            continue
        i += 1

    inputs = [p for p in inputs if p != 'clk']
    # group bit-blasted ports back into buses by prefix
    def group(ports):
        seen = []
        for p in ports:
            pre = p.rsplit('_', 1)[0]
            if pre not in seen:
                seen.append(pre)
        return {pre: [p for p in ports if p.rsplit('_', 1)[0] == pre]
                for pre in seen}

    in_groups, out_groups = group(inputs), group(outputs)
    if samples is None:
        samples = max([s.shape[2] for s in streams if s.ndim == 3], default=1)
    streams = [np.broadcast_to(s[:, :, None] if s.ndim == 2 else s,
                               (len(g), step_n, samples))
               for s, g in zip(streams, in_groups.values())]

    vals = {p: np.zeros(samples, dtype=np.int32)
            for p in inputs + [g[1] for g in latches]}
    for src, q, init in latches:
        vals[q] = np.full(samples, init, dtype=np.int32)

    result = {pre: np.zeros((len(g), step_n, samples), dtype=np.int32)
              for pre, g in out_groups.items()}
    for t in range(step_n):
        for (pre, g), s in zip(in_groups.items(), streams):
            for b, p in enumerate(g):
                vals[p] = s[b, t].astype(np.int32)
        for args, out, lut in gates:       # emitted in topological order
            if not args:
                vals[out] = np.full(samples, lut & 1, dtype=np.int32)
                continue
            idx = np.zeros(samples, dtype=np.int64)
            for k, a in enumerate(args):
                idx |= vals[a].astype(np.int64) << k
            table = np.array([(lut >> m) & 1 for m in range(1 << len(args))],
                             dtype=np.int32)
            vals[out] = table[idx]
        for pre, g in out_groups.items():
            for b, p in enumerate(g):
                result[pre][b, t] = vals[p]
        nxt = [vals[src].copy() for src, q, init in latches]
        for (src, q, init), v in zip(latches, nxt):
            vals[q] = v
    return result
