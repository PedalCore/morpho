# morpho · snn — developmental spiking networks as interactive music

> **This fork is an experiment.** It extends the ideas of
> [MorphoHDL](https://github.com/paradigms-of-intelligence/morpho) — recursive
> structural growth — into **spiking neural networks** and **interactive
> music**: a compact developmental grammar grows a recurrent spiking network
> whose ongoing activity decides which parts of its own structure grow,
> survive and are pruned — and everything it does is audible.
>
> **▶ Play with it now: [soundlark.studio](https://soundlark.studio)** — no
> install, runs in the browser.

## The three experiments (in `snn/`)

### 1. [The Lab](https://soundlark.studio/snn/) — a developmental spiking network you can listen to

A Morpho-style recursive grammar expands into a recurrent leaky
integrate-and-fire network. Pitch **is** anatomy: developmental depth sets the
register, structural position sets the scale degree — so you hear regions
divide and climb in register, pruning thin the texture, and rare "modulator"
neurons nudge the key around an interactive circle of fifths. Stochastic
walkers roam the graph as melodic voices, and the structures they play are the
ones that survive. Scales, microtonal tunings, rubato, steering, fx.

### 2. [The Duet](https://soundlark.studio/snn/duet.html) — play with a developing brain

The human replaces the metronome as the organism's environment. Your notes
(MIDI keyboard, on-screen pads, or computer keys — locked to the current
scale/key) are encoded as spike bursts into a tonotopic sensory layer wired to
the anatomy that sounds the same degrees. There is no internal drive: it
answers **you**, in the gaps you leave, at the tempo you asked at. STDP
strengthens the pathways you play, development reorganizes around them, a
reinforce button rewards answers you like, and a live *relatedness* score
tracks whether the dialogue is converging on your material. MIDI out lets it
play your hardware.

### 3. [Attention](https://soundlark.studio/snn/attention.html) — a brain that listens back

The duet plus attention-modulated spiking, adapted (gradient-free) from
[Attention Spiking Neural Networks](https://arxiv.org/abs/2209.13929): the
region of the anatomy whose pitch material best matches your recent playing
keeps its full voice, the rest quiet down — and attention feeds survival, so
what you attend to is what develops. Measured against the plain duet:
**+12% answer relatedness at 85% fewer spikes.**

## The research

Every claim below comes from a headless, seeded experiment in
[`snn/experiments/`](snn/experiments/) (run as
`cd snn && npm run experiment:<name>`), logged chronologically in
[`snn/EXPERIMENT.md`](snn/EXPERIMENT.md) and written up in prose at
[soundlark.studio/research.html](https://soundlark.studio/research.html)
(music) and
[soundlark.studio/language.html](https://soundlark.studio/language.html)
(language). Null results are reported next to the positive ones — they are
half the point.

### Music track — what the substrate learns, and what it can't

- **Suppress-only attention works** (adapted gradient-free from
  [MA-SNN](https://arxiv.org/abs/2209.13929)): +12% answer relatedness at
  85% fewer spikes; boosting instead of suppressing destabilizes.
- **Activity-driven development is itself a morphogen**: coverage of a played
  idiom rises even without attention; the attention energy trickle adds a
  modest sharpening on top.
- **Pitch style is learnable, rhythm is not (yet)**: after training on a
  score, pitch style leans toward the trained organism (0.81 vs 0.76 twin);
  rhythm style is null — the anatomy has no place for order to live, which
  the R-STDP and motif-sequence probes confirmed independently (both null
  under controls).
- **Organisms are fully persistent**: deterministic snapshots restore
  spike-for-spike identical continuations.

### Language track — the organism as a backprop-free reservoir

A sideline that became a ladder: tiny shakespeare next-char prediction with
the spiking organism as a liquid state machine and a closed-form ridge
readout — no backprop anywhere.

| Step | Next-char accuracy |
|---|---|
| exact bigram baseline | 28.8% |
| fresh reservoir | 29.2% |
| + developmental exposure (dev+STDP while "listening") | 31.7% |
| + error-driven growth (grows only while wrong, self-limits) | 33.3% |
| 120k-neuron deep SoA brain (4 layers, 15% inhibition) | 33.0% |
| + previous-char readout context | 34.2% |
| + more fit data + 2nd previous char | 39.1% |
| developmental genome selected at 2k–8k, same full budget | 42.5% |
| + joint readout scaling + char-gated nonlinear features | **45.8%** |
| char transformer reference (with backprop) | ≈58% |

Findings along the way: the ~33% ceiling was the linear readout, not
organism capacity (an 834-neuron grown organism ties a 120k-neuron brain);
role-aware pruning takes 120k → 50k neurons with accuracy intact, while
naive pruning kills inhibition first and the network seizes; a
Mamba-style input-dependent state write transfers directionally to the
gradient-free substrate; and the honest negatives — shallow Forward-Forward
heads underperform ridge three times running, belief feedback is null, and
free-running generation is still gibberish (exposure bias made vivid).

**v13 — evolving the law, not the network:** an 11-gene genome over the
developmental wiring rule (constant length in N), evolved at 2k–8k neurons
and frozen, transfers to a held-out 120k-neuron brain it never saw:
38.1–40.7% across all six selected lineages vs the hand-designed law's
36.1–39.0%, at up to 3.3× fewer synapses — with three independent runs
converging on the same signature (inhibition-rich, feedforward-sparse,
skip-dominated, recurrence selected out). Honest null: accuracy-wise,
evolution ≈ random search over the genome space; what it demonstrably
bought is sparsity. Protocol was pre-registered before results. Under the
full readout budget the best selected genome then set the new ladder best:
**42.5%** at 120k (prior 39.1%), at 44% fewer synapses, while the hand law's
same-seed control seized.

**Docs:** [`snn/README.md`](snn/README.md) (how to run and test) ·
[`snn/EXPERIMENT.md`](snn/EXPERIMENT.md) (hypothesis, protocol, findings,
v1–v12) · the research brief and parallel C++/JUCE plugin briefs live in
[`snn/docs/`](snn/docs/).

Everything is deterministic per seed (same seed = same organism, same spikes,
same harmonic journey), tested headlessly (`cd snn && npm test`), and deployed
automatically from this branch (`snn-lab`).

---

*The original MorphoHDL README follows — the upstream project this fork
builds on, unchanged at the repository root.*

---

# MorphoHDL

**A minimalistic language for growing circuits through structural recursion.**

MorphoHDL is an experimental Hardware Description Language (HDL) and graph rewrite system built around recursive division and rewiring of cell definitions. Inspired by Parametric L-Systems and functional HDLs, MorphoHDL grows physical geometry and logical circuit structures concurrently without hardcoded bus widths.

---

## 🌟 Key Features

* **Recursive & Size-Agnostic**: Cells define rewrite rules where nodes are dynamically replaced by subcells. Bus widths are inferred and split automatically at runtime.
* **High-Performance SoA Engine**: The core compiler (`js/compiler.js`) utilizes a Struct-of-Arrays (SoA) flat memory layout for maximum cache locality and performance.
* **Interactive WebGL Explorer**: Integrated interactive viewer (`demo.html`) powered by SwissGL and Canvas 2D for real-time visualization of circuit growth, force-directed layouts, and signal propagation.
* **Rich Library of Primitives**: Includes classical boolean circuits (parallel prefix adders, multipliers, logarithmic shifters) and biological/cellular automata structures.

---

## 🚀 Getting Started

MorphoHDL runs natively in the browser with no build steps required.

### 1. Launching Locally
Start any local HTTP server from the repository root:
```bash
python3 -m http.server 8000
```

### 2. Interactive Environments
* **Interactive Explorer (`demo.html`)**: Open `http://localhost:8000/demo.html` to experiment with live circuit growth, parameter controls, and layout visualization.
* **Interactive Article (`index.html`)**: Open `http://localhost:8000/index.html` to read the interactive documentation and walk through classical boolean circuit examples.

---

## 💻 Example Usage

MorphoHDL uses a clean, dataflow-style Python syntax where bus sizes are inferred dynamically:

```python
# Define building blocks using Lookup Tables (LUTs)
Xor3 = LUT(3, 0b1001_0110)
Maj3 = LUT(3, 0b1110_1000)

# Base case (1-bit full adder)
@morpho
def full_adder(a, b, c_in):   
    sum = Xor3(a, b, c_in)
    c_out = Maj3(a, b, c_in)
    return sum, c_out

# Recursive N-bit ripple adder with 1-bit fallback
@morpho(fallback=full_adder)
def ripple_adder(a, b, c):
    a0, a1 = SPLIT(a)
    b0, b1 = SPLIT(b)
    s0, c_mid = ripple_adder(a0, b0, c)
    s1, c_out = ripple_adder(a1, b1, c_mid)
    sum = CAT(s0, s1)
    return sum, c_out
```

---

## 📁 Repository Structure

* `tiny_morpho.py`: Standalone Python reference implementation of MorphoHDL with simulation, compilation, and verification tests.
* `demo.html` / `index.html`: Web-based interactive explorer and interactive article.
* `js/`: Core browser runtime and engine:
  * `compiler.js`: Flat SoA compiler and width inference engine.
  * `viewer.js` & `layout_renderer.js`: Interactive WebGL/Canvas circuit visualizer.
  * `force_layout.js` & `hex_layout.js`: Physics and grid-based circuit layout engines.
* `graphs_engine/`: High-performance C/WASM graph layout backend.
* `scratch/`: Experimental scripts, layout benchmarks, and verification tools.

---

## 📜 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
