# Morpho → hardware

`tiny_morpho_hw.py` (repository root) turns what `compile()` /
`compile_seq()` already produce — k-input LUTs, flip-flops, constants,
wires, one implicit clock — into the two formats synthesis tools eat:

```python
from tiny_morpho_hw import to_blif, to_verilog

blif    = to_blif(circuit, 'divider', output_names=('q', 'rem'))
verilog = to_verilog(circuit, 'divider', output_names=('q', 'rem'))
```

It is a transcription, not a compilation: `GATE(lut, args)` → a LUT with
that exact truth table, `REG(init)` → a flip-flop (`.latch … re clk` in
BLIF, `always @(posedge clk)` in Verilog), the synchronous commit → the
clock edge. FORWARD/TIE fixed-point circuits are rejected: the
synthesizable subset is the REG-only discipline.

## Verification, three layers

```bash
python3 examples/hardware/export_units.py
```

1. **Round trip** — `simulate_blif()` re-parses the emitted *text* and
   re-simulates it against the compiled circuit: exhaustive for the
   `2^(−x)` unit (all 65,536 Q8.8 inputs), 2,000 random divisions for the
   combinational divider, 512 divisions × 16 ticks for the streaming one.
2. **Formal equivalence** — yosys `equiv_simple` + `equiv_induct` proves
   the emitted Verilog ≡ the emitted BLIF (registers included).
3. **Synthesis + place-and-route** — the open flow
   (`brew install yosys nextpnr-ice40 icestorm`), targeting an iCE40 HX8K.
   Combinational units get a registered harness so f_max measures their
   real critical path.

## Measured results (iCE40 HX8K, nextpnr timing)

| unit | logic cells | f_max | note |
|---|---|---|---|
| `exp2neg` — wkv's 2^(−x), ROM+interp+shift | 345 | 47.1 MHz | exhaustively bit-exact before export |
| `divider` 16/8 combinational | 416 | 14.7 MHz | 16 cascaded ripple subtractors — depth is the price of one-cycle division |
| `serial_divider` streaming | **29** | **163.3 MHz** | the space→time rotation: same long division, 14× fewer cells, 11× the clock |

The last row is the article's rotation made physical: threading the
remainder through a register instead of through space turns a deep slow
circuit into a tiny fast one, and the trade is measured by a real
place-and-route tool, not estimated by our `logic_depth` metric.

All three fit in a corner of the smallest hobby FPGAs ($10 parts). The
RWKV model's entire recurrent state is 6,912 bits of register — the wkv
cell bank is small-FPGA territory with the matmuls on DSP blocks; see the
wkv-cell page for the full architecture sketch.

Caveats kept honest: timing closure is nextpnr's static analysis at the
default corner; large ROMs/matmuls should map to BRAM/DSP macros (an
exporter feature, not a language problem); the SR-latch/FORWARD corner is
deliberately outside this flow.
