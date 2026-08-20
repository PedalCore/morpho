# Sequential MorphoHDL — state, feedback, and things that move

MorphoHDL as published grows combinational circuits: a program describes
`y = f(x)`. This adds the smallest extension that lets it describe
`s[t+1] = F(s[t], x[t])` instead — and with it, everything that has a pulse.

Two new primitives, and one for the asynchronous corner:

```python
q = REG(init)        # allocate registers, initialised to constant bits
DRIVE(q, next)       # close a synchronous loop: q takes `next` each tick
w = FORWARD(ref)     # declare a feedback wire of a given width
TIE(w, value)        # bind it — REG-free cycles, resolved as fixed points
```

`compile_seq()` traces a program into a *cyclic* gate graph, classifies its
strongly-connected components, and refuses cycles that don't cross a register
unless `allow_async=True` is passed — so a latch is opt-in rather than an
accident.

## Try it

```bash
python tiny_morpho_seq.py           # the test suite
python examples/sequential/counter.py
open circuits-that-move.html        # interactive, sonified — no build step
```

`circuits-that-move.html` is a self-contained page: a delay line, a Johnson
counter, a serial adder that carries through *time* rather than space, an
LFSR whose taps you can drag, a live-editable elementary CA, an SR latch
demonstrating metastability on simultaneous release, and a travelling-wave
"tentacle" where register state drives muscle segments. Every demo is the
MorphoHDL program shown beside it.

## Examples

| file | what it shows |
|---|---|
| `examples/sequential/counter.py` | registers wrapped around the article's ripple adder — a number that is also a bank of divided clocks |
| `examples/sequential/delay_line.py` | remembering one signal |
| `examples/sequential/johnson_counter.py` | 2N states from N registers and one NOT |
| `examples/sequential/lfsr.py` | maximal-period pseudo-randomness from two taps |
| `examples/sequential/pulse_stretcher.py` | a timed one-shot |
| `examples/arithmetic/exp2.py` | 2^(−x) as barrel shift + interpolated LUT, verified exhaustively over all 65,536 Q8.8 inputs (388 gates) |
| `examples/arithmetic/divider.py` | restoring division (527 gates at 16/8) |

Everything is verified bit-exactly against numpy oracles; the CA examples are
checked against a reference stepper across grids, rules and horizons.

## Why bother

A serial adder is the article's ripple adder with its carry chain rotated out
of space and into time: one full adder plus one register handles any operand
width. That trade — area against latency, structure against time — is exactly
what a recursive, size-agnostic description language should be able to
express, and it needs state to say it.
