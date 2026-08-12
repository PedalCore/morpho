# The wkv channel, as gates

`wkv_cell.py` is the complete RWKV wkv recurrence — the cell the
[wkv-cell page](../../site/wkv-cell.html) simulates and the perplexity
sweep validated — as **one compiled Morpho circuit**: 7,201 gates and
48 registers (aa, bb, pp: three Q8.8 words, the channel's entire
recurrent state).

```bash
python3 examples/rwkv/wkv_cell.py
```

## The spec is the page

The circuit implements the page's `Q.step` exactly: Q8.8 saturating
fixed point, base-2 exponentials through the exhaustively-verified
`exp2` unit (32-entry ROM + 3-bit floor interpolation), floor-semantics
multiplies, the `den + 1` zero guard, restoring division, `pp`
initialized to −32768. The numpy reference in the file is a line-by-line
transcription of that JavaScript.

One structural liberty, provably identity-preserving: after max
subtraction one exponential of each pair is exp2(0) = 65536 exactly, and
`mulQ16(z, 65536) = z` for every z — so the circuit muxes on the
comparison instead of instantiating dead multipliers: **two** exp2 units
and **four** multipliers rather than four and six. Bit-identical, ~40%
smaller.

## Verification

| check | scope | result |
|---|---|---|
| single step vs reference | 20,006 vectors incl. rail/edge battery, dynamic + compiled | bit-exact |
| streams vs reference | 96 ticks × 320 channels through `compile_seq` | bit-exact |
| **the page's actual JavaScript** vs the same reference | 200 ticks × 10 channels, trained presets, run in a real browser | **0 mismatches in 2,000 steps** |
| float64 recurrence tracking | 256 steps, trained (w₂,u₂) presets | rms 5.7×10⁻³ |
| BLIF text round-trip | re-parsed netlist, 120 channels × 32 ticks | bit-exact |
| Verilog ≡ BLIF | yosys `equiv_simple` + `equiv_induct` | proven |

The trained presets are real (w₂, u₂) pairs from the model atlas export
(block 0/3/5 medians and slow-tail channels). The float-tracking rms sits
in the same band the collaborator's 8-arm perplexity sweep showed to be
harmless: every fixed-point format scored identical perplexity to float64.

## Synthesis

`examples/hardware/export_units.py` pushes the cell through
yosys → nextpnr-ice40: **5,648 logic cells — 73% of an iCE40 HX8K, a $10
part — placed, routed, and timed at 9.8 MHz**, one token per clock with
the entire step (exponentials, multiplies, division) in a single
combinational pass. Pipelining that pass is the obvious next trade if
clock rate matters. The model needs 384 of these channels per block —
the parallel bank is large-FPGA/ASIC territory, but the state being just
48 bits per channel means one time-multiplexed cell plus block RAM
serves a whole block on a small FPGA, exactly the architecture the
wkv-cell page sketches.
