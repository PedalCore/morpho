# M8 — oscillatory memory and retrieval diagnostics (design,
# preregistered 2026-08-28; collaborator spec)

Sources: Modern Hopfield (2008.02217) — diagnostics only; SineKAN
(2407.04149) — frequency parameterization only. Neither replaces
Longhorn (explicit decision).

## 1. Hopfield-derived retrieval diagnostics (instrument, not arch)

Hopfield retrieval = attention as one energy-minimization step; its
three regimes map onto our measured failure modes: global average ~
counter compression; metastable subset ~ entity collisions; single
fixed point ~ clean Longhorn retrieval. ADOPT the separation margin
Delta_i = x_i^T x_i - max_{j!=i} x_i^T x_j as instrumentation for
ALL future binding probes:
- correct-key margin; nearest-incorrect margin;
- retrieval entropy; effective #memories averaged;
- error vs margin curves.
Distinguishes erased association vs ambiguous keys vs blended
retrieval — the diagnosis M4/M5 probes could not make. Optional
follow-on: explicit separation loss on Longhorn keys/slot addresses.
Hopfield-as-architecture = bounded softmax KV memory (a formally
justified slots variant); NOT adopted — exponential-capacity claims
do not transfer to learned representations.

## 2. Oscillatory (complex) counters — the SineKAN translation

NOT SineKAN blocks (feature-space sines, no prox objective, no
hardware story). Instead oscillatory recurrent heads:
    s_t^(k) = rho_k e^{j omega_k} s_{t-1}^(k) + B_k x_t
= complex counters: how much AND at what phase. Connects: pitch
tracker C-statistic; MIDI rhythm/meter; DNA codon (period 3) and
nucleosome (~10.5 b) periodicity; positional info real counters
discard. Frequencies init on fixed log/dyadic (or musically/
biologically meaningful) grid; optionally learned
(SineKAN-style parameterization).

## 3. The five-arm frequency-memory probe (build next)

| arm | memory |
|---|---|
| A | real decaying counters (control) |
| B | fixed-frequency complex counters |
| C | learnable-frequency complex counters |
| D | Longhorn (control) |
| E | hybrid: Longhorn + 1/8 oscillatory heads |

Tasks: (1) periodic delayed recall; (2) MIDI rhythm/motif
continuation; (3) codon-frame / synthetic DNA periodicity;
(4) arbitrary key-value binding (NEGATIVE control).

PREREGISTERED PREDICTION (collaborator): learned oscillatory heads
win strongly on 1-3, LOSE to Longhorn on 4. If so, the architecture
conclusion is composition, not replacement: Longhorn for
associations + a small oscillatory bank for phase/periodicity —
mirroring the M7 stem-x-binding lesson that composition beats
universal blocks.

Status: design only. Build after the from-scratch baseline column
(task #56) completes. Probe scale: M5-style synthetic, Mac-runnable.
