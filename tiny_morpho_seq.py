# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tiny MorphoHDL sequential extension: feedback, registers and dynamical systems.

Extends `tiny_morpho` from pure functions y = f(x) to dynamical systems
s[t+1] = F(s[t], x[t]). The circuit IR becomes an arbitrary cyclic graph;
cycles are classified by strongly-connected-component analysis:

  - acyclic ops                  -> ordinary combinational logic
  - cycles broken by REG         -> legal synchronous state
  - cycles without REG (async)   -> fixed-point circuits, opt-in via allow_async

New primitives:

  - REG(init)      Allocate a bank of registers (one per bit of `init`,
                   which must be constant 0/1 bits). Returns the Q output bus.
  - DRIVE(q, nxt)  Close the synchronous loop: q[t+1] = nxt[t].
  - FORWARD(ref)   Declare a forward wire bus shaped like `ref` (raw feedback).
  - TIE(fw, val)   Bind a forward wire to its driver.

Sections:

  - Sequential primitives   REG / DRIVE / FORWARD / TIE
  - SequentialCompiler      cyclic-graph tracing compiler (extends CircuitCompiler)
  - SequentialCircuit       discrete-time simulator with SCC scheduling and
                            fixed-point relaxation of async cycles
  - Example cells           serial adder, LFSR, SR latch
  - Cellular automata       elementary CA (Rule 30/90/110...) and non-uniform CA
  - tests                   verification against numpy references and the
                            combinational adders from `tiny_morpho`
"""

import inspect
import numpy as np

from tiny_morpho import (G_Runner, LUT, morpho, CAT, ZERO, ONE, Op,
                         CircuitCompiler, flatten_tree, unpack, pack,
                         Not, Xor, Xor3, Maj3)

Nor = LUT(2, 0b0001)

#@MARK: Sequential primitives

def _seq_compiler():
    c = G_Runner.get()
    if not isinstance(c, SequentialCompiler):
        raise RuntimeError("Sequential primitives require compile_seq()")
    return c

def REG(init):
    """Allocate registers initialized to the constant bits `init`, return Q bus."""
    return _seq_compiler().make_reg(np.atleast_1d(np.asarray(init)))

def DRIVE(q, nxt):
    """Close a synchronous loop: register bus q takes the value of nxt each step."""
    _seq_compiler().set_driver(q, nxt, 'REG')

def FORWARD(ref):
    """Declare a forward (feedback) wire bus with the width of `ref`."""
    return _seq_compiler().make_forward(len(np.atleast_1d(ref)))

def TIE(fw, val):
    """Bind a FORWARD bus to its driving value."""
    _seq_compiler().set_driver(fw, val, 'FWD')


#@MARK: SequentialCompiler

class SequentialCompiler(CircuitCompiler):
    """CircuitCompiler that additionally traces REG and FORWARD ops.

    Drivers are recorded separately during tracing (they don't exist yet when
    the op is created) and materialized into op args by finalize().
    """
    def __init__(self, optimize=True):
        super().__init__(optimize)
        self.drivers = {}  # op index -> driving wire index (None until set)

    def _alloc(self, kind, name, lut=None):
        idx = len(self.ops)
        self.ops.append(Op(kind, name, lut))
        self.depths.append(0)
        self.drivers[idx] = None
        return idx

    def make_reg(self, init):
        if ((init < 0) | (init > 1)).any():
            raise ValueError("REG init must be constant 0/1 bits")
        base = sum(op.type == 'REG' for op in self.ops)
        return np.array([self._alloc('REG', f'reg{base + i}', int(b))
                         for i, b in enumerate(init)], dtype=np.int32)

    def make_forward(self, n):
        base = sum(op.type == 'FWD' for op in self.ops)
        return np.array([self._alloc('FWD', f'fwd{base + i}')
                         for i in range(n)], dtype=np.int32)

    def set_driver(self, q, val, kind):
        q, val = np.atleast_1d(q).ravel(), np.atleast_1d(val).ravel()
        if len(q) != len(val):
            raise ValueError(f"width mismatch: {len(q)} vs {len(val)}")
        for qi, vi in zip(q, val):
            if self.ops[qi].type != kind or int(qi) not in self.drivers:
                raise ValueError(f"wire {qi} is not an undriven {kind}")
            if self.drivers[int(qi)] is not None:
                raise ValueError(f"{kind} wire {qi} driven twice")
            self.drivers[int(qi)] = int(vi)

    def finalize(self):
        undriven = [i for i, d in self.drivers.items() if d is None]
        if undriven:
            names = [self.ops[i].name for i in undriven]
            raise RuntimeError(f"undriven REG/FORWARD wires: {names}")
        for i, d in self.drivers.items():
            op = self.ops[i]
            self.ops[i] = Op(op.type, op.name, op.lut, (d,))

    def strip_unused_gates(self):
        # Like the base DCE, but driver edges point forward (a live REG/FWD at a
        # low index keeps a high-index driver alive), so iterate to a fixpoint.
        live = np.zeros(len(self.ops), dtype=bool)
        live[:2] = True
        live[flatten_tree(self.outputs)] = True
        for inp in self.inputs:
            live[inp] = True
        changed = True
        while changed:
            changed = False
            for i in range(len(self.ops) - 1, -1, -1):
                op = self.ops[i]
                if live[i] and op.args and not live[list(op.args)].all():
                    live[list(op.args)] = True
                    changed = True

        mapping = np.cumsum(live) - 1
        new_ops = []
        for i in range(len(self.ops)):
            if not live[i]:
                continue
            op = self.ops[i]
            args = tuple(mapping[list(op.args)]) if op.args else op.args
            new_ops.append(Op(op.type, op.name, op.lut, args))
        self.stats['dce_elim'] = len(self.ops) - len(new_ops)
        self.ops = new_ops
        self.depths = self.depths[live]
        self.inputs = [mapping[inp] for inp in self.inputs]

        def remap(x):
            if isinstance(x, (list, tuple)):
                return type(x)(remap(item) for item in x)
            return mapping[x]
        self.outputs = remap(self.outputs)


#@MARK: SequentialCircuit

def _tarjan_sccs(deps):
    """Iterative Tarjan. Emits SCCs with dependencies first (valid eval order)."""
    n = len(deps)
    index, low = [-1] * n, [0] * n
    on_stack, stack, sccs, counter = [False] * n, [], [], 0
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter = counter + 1
                stack.append(v)
                on_stack[v] = True
            advanced = False
            for k in range(pi, len(deps[v])):
                w = deps[v][k]
                if index[w] == -1:
                    work[-1] = (v, k + 1)
                    work.append((w, 0))
                    advanced = True
                    break
                if on_stack[w]:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if low[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
    return sccs

class SequentialCircuit:
    """Discrete-time simulator: s[t+1] = F(s[t], x[t]), y[t] = G(s[t], x[t]).

    Ops are scheduled by SCC condensation of the combinational dependency
    graph (REG outputs are sources: they read state, not this-step values).
    Cyclic SCCs are combinational feedback; they require allow_async=True and
    are relaxed to a fixed point each step. Wire values persist across steps,
    so async loops (latches) naturally hold state.
    """
    def __init__(self, circuit, allow_async=False, max_iter=64):
        self.c, self.max_iter = circuit, max_iter
        ops = circuit.ops
        self.reg_idx = np.array([i for i, op in enumerate(ops)
                                 if op.type == 'REG'], dtype=np.int32)
        self.reg_init = np.array([ops[i].lut for i in self.reg_idx], dtype=np.int32)
        self.reg_next = np.array([ops[i].args[0] for i in self.reg_idx], dtype=np.int32)

        deps = [list(op.args) if op.type in ('GATE', 'FWD') else [] for op in ops]
        self.schedule = []
        for scc in _tarjan_sccs(deps):
            if len(scc) == 1 and scc[0] not in deps[scc[0]]:
                if ops[scc[0]].type in ('GATE', 'FWD'):
                    self.schedule.append(('op', scc[0]))
            else:
                self.schedule.append(('scc', sorted(scc)))
        self.async_blocks = [blk for kind, blk in self.schedule if kind == 'scc']
        if self.async_blocks and not allow_async:
            raise RuntimeError(
                f"{len(self.async_blocks)} combinational cycle(s) without REG "
                "detected; pass allow_async=True to simulate them as "
                "fixed-point (asynchronous) circuits")

    def _eval_op(self, i, vals):
        op = self.c.ops[i]
        if op.type == 'FWD':
            vals[i] = vals[op.args[0]]
            return
        idx = sum(vals[a] << k for k, a in enumerate(op.args))
        vals[i] = (op.lut >> idx) & 1

    def _relax(self, block, vals):
        for _ in range(self.max_iter):
            before = vals[block].copy()
            for i in block:
                self._eval_op(i, vals)
            if (vals[block] == before).all():
                return
        raise RuntimeError("combinational feedback failed to settle "
                           "(oscillating circuit?)")

    def _capture(self, vals):
        out = self.c.outputs
        if isinstance(out, tuple):
            return tuple(vals[np.asarray(o)].copy() for o in out)
        return vals[np.asarray(out)].copy()

    def run(self, step_n, *streams, state0=None, samples=None):
        """Simulate step_n steps. Each input stream has shape (width, step_n)
        or (width, step_n, samples). Returns output arrays shaped
        (width, step_n[, samples]); the samples axis is dropped when it is 1.
        state0 optionally overrides register init, shape (n_regs[, samples])."""
        widths = [len(inp) for inp in self.c.inputs]
        if len(streams) != len(widths):
            raise ValueError(f"expected {len(widths)} input streams, got {len(streams)}")
        streams = [np.asarray(s) for s in streams]
        if samples is None:
            samples = max([s.shape[2] for s in streams if s.ndim == 3], default=1)
        streams = [np.broadcast_to(s[:, :, None] if s.ndim == 2 else s, (w, step_n, samples))
                   for s, w in zip(streams, widths)]

        vals = np.zeros((len(self.c.ops), samples), dtype=np.int32)
        vals[1] = 1
        state = self.reg_init[:, None] if state0 is None else np.atleast_2d(np.asarray(state0).T).T
        if len(self.reg_idx):
            vals[self.reg_idx] = np.broadcast_to(state, (len(self.reg_idx), samples))

        trace = []
        for t in range(step_n):
            for inp, s in zip(self.c.inputs, streams):
                vals[inp] = s[:, t]
            for kind, item in self.schedule:
                if kind == 'op':
                    self._eval_op(item, vals)
                else:
                    self._relax(item, vals)
            trace.append(self._capture(vals))
            if len(self.reg_idx):
                vals[self.reg_idx] = vals[self.reg_next]  # synchronous commit

        def stack(k=None):
            frames = [tr if k is None else tr[k] for tr in trace]
            out = np.stack(frames, axis=1)
            return out[..., 0] if samples == 1 else out
        if isinstance(self.c.outputs, tuple):
            return tuple(stack(k) for k in range(len(self.c.outputs)))
        return stack()

    def report(self):
        self.c.report()
        cycle_ops = sum(len(b) for b in self.async_blocks)
        print(f"Sequential: {len(self.reg_idx)} registers, "
              f"{len(self.async_blocks)} async feedback block(s) "
              f"({cycle_ops} ops in cycles)")


def compile_seq(f, arg_sizes=(), optimize=True, allow_async=False, max_iter=64):
    """Trace f into a cyclic gate graph and wrap it in a SequentialCircuit."""
    circuit = SequentialCompiler(optimize)
    token = G_Runner.set(circuit)
    try:
        args = [circuit.add_input(name, size) for name, size
                in zip(inspect.signature(f).parameters, arg_sizes)]
        circuit.outputs = f(*args)
    finally:
        G_Runner.reset(token)
    circuit.finalize()
    circuit.depths = np.array(circuit.depths, dtype=np.int32)
    if optimize:
        circuit.strip_unused_gates()
    return SequentialCircuit(circuit, allow_async=allow_async, max_iter=max_iter)


#@MARK: Example cells

# Bit-serial adder: the ripple adder's carry chain rotated from space into
# time. One full adder + one register processes any operand width, LSB first.
@morpho
def serial_adder(a, b):        # a: [1]/step, b: [1]/step -> sum: [1]/step
    carry = REG(ZERO)
    s = Xor3(a, b, carry)
    DRIVE(carry, Maj3(a, b, carry))
    return s

def make_lfsr(width=4, taps=(3, 2)):
    """Fibonacci LFSR seeded with 1 in bit 0 (default taps give period 2^4-1)."""
    @morpho
    def lfsr():
        s = REG(np.eye(1, width, 0, dtype=np.int32)[0])
        fb = Xor(s[taps[0]:taps[0] + 1], s[taps[1]:taps[1] + 1])
        DRIVE(s, CAT(fb, s[:-1]))
        return s
    return lfsr

# Asynchronous (REG-free) feedback: a NOR-based SR latch holding its state
# purely in a combinational cycle. Requires compile_seq(..., allow_async=True).
@morpho
def sr_latch(s, r):            # s: [1]/step, r: [1]/step -> q: [1], qn: [1]
    q, qn = FORWARD(ONE), FORWARD(ONE)
    TIE(q, Nor(r, qn))
    TIE(qn, Nor(s, q))
    return q, qn


#@MARK: Cellular automata

def make_eca(rule, width):
    """Uniform elementary CA on a width-cell ring: an N-bit register bank,
    two cyclic shifts and one 3-input LUT holding the Wolfram rule."""
    Rule = LUT(3, rule, name=f'Rule{rule}')
    @morpho
    def eca():                 # -> state: [width]/step
        state = REG(np.zeros(width, dtype=np.int32))
        l = CAT(state[-1:], state[:-1])   # left neighbors (cyclic)
        r = CAT(state[1:], state[:1])     # right neighbors (cyclic)
        DRIVE(state, Rule(r, state, l))   # idx = r + 2c + 4l = Wolfram order
        return state
    eca.__name__ = f'eca_rule{rule}'
    return eca

def make_nonuniform_ca(rules):
    """Non-uniform CA: every cell on the ring applies its own Wolfram rule."""
    n = len(rules)
    luts = [LUT(3, r, name=f'Rule{r}') for r in rules]
    @morpho
    def ca():                  # -> state: [len(rules)]/step
        state = REG(np.zeros(n, dtype=np.int32))
        cells = [luts[i](state[(i + 1) % n:(i + 1) % n + 1],
                         state[i:i + 1],
                         state[(i - 1) % n:(i - 1) % n + 1]) for i in range(n)]
        DRIVE(state, CAT(*cells))
        return state
    return ca

def eca_reference(rules, state0, step_n):
    """Numpy oracle for (non-)uniform cyclic ECA. rules: int or per-cell array.
    state0: (width[, samples]). Returns (width, step_n[, samples])."""
    rules = np.asarray(rules) if np.ndim(rules) else np.int64(rules)
    if rules.ndim and state0.ndim > 1:
        rules = rules[:, None]
    frames, s = [state0], state0
    for _ in range(step_n - 1):
        l, r = np.roll(s, 1, axis=0), np.roll(s, -1, axis=0)
        s = (rules >> (4 * l + 2 * s + r)) & 1
        frames.append(s)
    return np.stack(frames, axis=1)


#@MARK: tests

def test_serial_adder(bit_n=16, sample_n=256):
    sim = compile_seq(serial_adder, (1, 1))
    rng = np.random.default_rng(42)
    a, b = rng.integers(1 << bit_n, size=(2, sample_n))
    out = sim.run(bit_n, unpack(a, bit_n)[None], unpack(b, bit_n)[None])
    assert (pack(out[0]) == (a + b) & ((1 << bit_n) - 1)).all()
    print("Serial adder matches a+b over", bit_n, "cycles!")

def test_eca(width=32, step_n=48, sample_n=8):
    rng = np.random.default_rng(42)
    state0 = rng.integers(2, size=(width, sample_n))
    for rule in (30, 90, 110):
        sim = compile_seq(make_eca(rule, width))
        trace = sim.run(step_n, state0=state0, samples=sample_n)
        assert (trace == eca_reference(rule, state0, step_n)).all(), f"Rule {rule} failed"
    print("Rules 30/90/110 match the numpy oracle!")

def test_nonuniform_ca(width=16, step_n=32, sample_n=8):
    rng = np.random.default_rng(42)
    rules = rng.choice([30, 54, 60, 90, 110, 150], size=width)
    state0 = rng.integers(2, size=(width, sample_n))
    sim = compile_seq(make_nonuniform_ca(rules))
    trace = sim.run(step_n, state0=state0, samples=sample_n)
    assert (trace == eca_reference(rules, state0, step_n)).all()
    print("Non-uniform CA matches the numpy oracle!")

def test_lfsr():
    trace = compile_seq(make_lfsr()).run(30)
    states = pack(trace)
    assert (states[15:] == states[:15]).all()      # period divides 15
    assert len(set(states[:15])) == 15 and 0 not in states  # maximal, nonzero
    print("4-bit LFSR is maximal with period 15!")

def test_sr_latch():
    sim = compile_seq(sr_latch, (1, 1), allow_async=True)
    s = np.array([[1, 0, 0, 0, 0, 1]])
    r = np.array([[0, 0, 1, 1, 0, 0]])
    q, qn = sim.run(6, s, r)
    assert (q == [1, 1, 0, 0, 0, 1]).all(), "latch failed to set/hold/reset"
    assert (qn == 1 - q).all()
    print("SR latch sets, holds and resets through pure feedback!")

def test_guards():
    try:
        compile_seq(sr_latch, (1, 1))
        assert False, "async cycle slipped through"
    except RuntimeError as e:
        assert 'allow_async' in str(e)

    @morpho
    def osc():
        q = FORWARD(ONE)
        TIE(q, Not(q))
        return q
    try:
        compile_seq(osc, allow_async=True).run(1)
        assert False, "oscillator settled?"
    except RuntimeError as e:
        assert 'settle' in str(e)

    @morpho
    def undriven():
        return REG(ZERO)
    try:
        compile_seq(undriven)
        assert False, "undriven REG slipped through"
    except RuntimeError as e:
        assert 'undriven' in str(e)
    print("All cycle/driver guard tests passed successfully!")

def test_dce():
    @morpho
    def dead_reg(x):
        d = REG(ZERO)
        DRIVE(d, Not(d))    # unused counter: should be stripped entirely
        return Not(x)
    sim = compile_seq(dead_reg, (1,))
    assert len(sim.reg_idx) == 0 and len(sim.c.ops) == 4
    print("DCE strips dead registers and keeps live drivers!")

def show_rule110(width=64, step_n=32):
    state0 = np.zeros((width, 1), dtype=np.int32)
    state0[width // 2] = 1
    trace = compile_seq(make_eca(110, width)).run(step_n, state0=state0)
    print("\nRule 110 from a single seed (1 LUT + shifts + REG bank):")
    for t in range(step_n):
        print(''.join('#' if v else '.' for v in trace[:, t]))

if __name__ == "__main__":
    test_serial_adder()
    test_eca()
    test_nonuniform_ca()
    test_lfsr()
    test_sr_latch()
    test_guards()
    test_dce()
    print("All sequential tests passed!")

    show_rule110()
    print()
    compile_seq(make_eca(110, 64)).report()
