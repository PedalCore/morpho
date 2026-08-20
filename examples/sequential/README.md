# Circuits that move

Small sequential examples for the MorphoHDL dynamic-systems extension
(`tiny_morpho_seq.py`). Each is self-contained and self-testing — run any
file from the repository root or from this directory:

```bash
python3 examples/sequential/delay_line.py
```

| example | teaches | circuit |
|---|---|---|
| `delay_line.py` | remember one signal | N registers in a chain |
| `johnson_counter.py` | a ring with one inversion = a 2N-state sequencer | N registers + 1 NOT |
| `lfsr.py` | pseudo-random necklaces, maximal periods | N registers + 1 XOR |
| `counter.py` | temporal hierarchy (each bit halves the clock) | registers around the article's ripple adder |
| `pulse_stretcher.py` | hold a pulse for N ticks | delay chain + OR |

They form a progression: remember a signal → use memory for computation →
create autonomous dynamics. The serial adder (memory *for* computation)
and the elementary cellular automata (distributed dynamics) live in
`tiny_morpho_seq.py` itself; the interactive gallery for all of these is
`site/circuits-that-move.html`.
