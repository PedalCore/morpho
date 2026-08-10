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

## 🔁 Dynamic Systems Extension (v0.2, this branch)

`tiny_morpho_seq.py` extends the Python reference implementation from pure functions `y = f(x)` to dynamical systems `s[t+1] = F(s[t], x[t])`. The circuit IR becomes a cyclic graph; cycles are classified by SCC analysis:

* **Acyclic ops** → ordinary combinational logic.
* **Cycles broken by `REG`** → legal synchronous state, simulated step-by-step.
* **Cycles without `REG`** → asynchronous fixed-point circuits, opt-in via `allow_async=True` and relaxed to a fixed point each step (oscillation is detected and reported).

Two new primitives (`REG(init)`/`DRIVE(q, next)` for registers, `FORWARD(ref)`/`TIE(fw, val)` for raw feedback) are enough to express cellular automata, bit-serial arithmetic, LFSRs, and latches with the existing Morpho mechanisms — no CA-specific language features:

```python
def make_eca(rule, width):        # e.g. Rule 110 on a ring of `width` cells
    Rule = LUT(3, rule)
    @morpho
    def eca():
        state = REG(np.zeros(width, dtype=np.int32))
        l = CAT(state[-1:], state[:-1])   # cyclic left/right neighbors
        r = CAT(state[1:], state[:1])
        DRIVE(state, Rule(r, state, l))
        return state
    return eca

trace = compile_seq(make_eca(110, 64)).run(32, state0=seed)  # (width, T) spacetime
```

The same relational structure can live in space or in time: `serial_adder` is the ripple-carry chain rotated into the time dimension — one full adder plus one register handles any operand width. Run `python3 tiny_morpho_seq.py` for the test suite (verified against numpy oracles and the combinational adders) and a Rule 110 demo.

---

## 📁 Repository Structure

* `tiny_morpho.py`: Standalone Python reference implementation of MorphoHDL with simulation, compilation, and verification tests.
* `tiny_morpho_seq.py`: Sequential/dynamical-systems extension: registers, feedback, discrete-time simulation, and cellular automata.
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
