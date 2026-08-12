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
| `wkv_cell` — the full RWKV channel | **5,648 (73% of HX8K)** | 9.8 MHz | one token per clock, whole step combinational; bit-exact vs the wkv-cell page's own JavaScript |

The wkv row means the complete recurrence — two exponentials, four
multipliers, saturating adds, restoring division, three state registers —
fits a $10 FPGA with room to spare, single-cycle-per-token at ~10 MHz.
That clock is the price of doing exp→multiply→divide in one combinational
pass; pipelining the step (registers between the exp, multiply, and divide
stages) trades latency for the fabric's real clock rate, and is a design
choice, not a language problem. One formal-verification honesty note:
yosys equivalence for the wkv Verilog-vs-BLIF pair is bounded at 300 s and
does not complete — SAT equivalence over multiplier cones is the classic
exponential case — so that pair rests on the text round-trip (bit-exact on
120 channels × 32 ticks); the three smaller units, built by the same
emitter, are formally proven.

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

## Predicted throughput

The per-token invoice is fixed by the model manifest: 13.1M int8 MACs,
which at one byte per weight is also 13.1 MB of weights streamed per
generated token. Three ceilings, lowest wins:

    bandwidth:  tok/s = bytes/s ÷ 13.1 MB     (rule of thumb: MB/s ÷ 13)
    compute:    tok/s = MAC/s   ÷ 13.1 M
    wkv:        tok/s = cells × f_max ÷ 2,304 (measured: 4,250/cell at 9.8 MHz)

The wkv hardware never binds — one measured cell covers 4,250 tok/s, two
orders of magnitude above realistic bottlenecks. Generation is
weight-bandwidth bound (as everywhere): ~4 tok/s from SPI flash, ~120 on
an ECP5 with 16-bit DDR3, ~300 on a Zynq, thousands with weights on-chip.
Interactive predictor + full argument: site/morpho-silicon.html.
