# M10 — spikes ARE the rhythm: minimal machinery for musical time
# (design, preregistered 2026-08-29; user concept)

CONCEPT (user): drop tokenization — represent rhythm as spike trains
where the representation maps ONE-TO-ONE to musical output. A spike
at bin t IS the event at time t. Normalize pitch to one note,
velocity to on/off; find the MINIMAL machinery that replicates or
learns rhythmic structure. Read heads = aggregations over the spike
raster.

KEY UNIFICATION: beat tracking is pitch tracking at 0.5-8 Hz. The
M8/pitch complex-counter estimator (C = decayed z z*, phase ->
frequency, coherence -> confidence) applied to the onset train at
tempo frequencies IS a tempo/phase/meter tracker, near
parameter-free. The M8 oscillator bank is not analogous to meter —
it is meter.

## Data
PRIMARY (user): Groove MIDI Dataset (Magenta) — 13.6 h of real
drummers, e-kit, metronome-aligned, tempo+style annotated, human
microtiming preserved. Drums ARE rhythm: nothing to strip. Collapse
to one onset channel for the ladder (9-channel kit version later).
20 ms bins; do NOT quantize to tatum (expressiveness is signal).
Official train/val/test split. SECONDARY: ARIA onsets (transfer).

## The ladder (params ~, preregistered predictions)
L0 Bernoulli / bin-conditional Markov floor (1-10): the nothing.
L1 single LIF unit, learned decay+threshold, self-history drive
   (~3): PREDICT locks to isochronous pulse only.
L2 oscillator bank: complex counters at log-spaced tempo freqs,
   spike prob from phase, learned amplitudes/couplings (50-200):
   PREDICT tempo-following + meter (strong/weak) appear here.
L3 L2 + CRSA spike-count ladder over the raster (the "read head
   over stacks") (1-5k): PREDICT bar-level pattern repetition
   appears here (needs WHICH beats fired, not just phase).
L4 tiny full model (~50k): ceiling reference.

## Eval
Next-bin NLL + onset F1 (tolerance +/-1 bin) vs ladder; click-track
continuation renders (listenable); per-rung capability probes:
isochronous / meter / swing / bar-repetition synthetic rhythms
before ARIA (the M7 lesson: controls first, and mind the leaks —
e.g. a rate-only model fakes F1 on dense passages; report per-IOI
stratified metrics).

## Why it matters
1. "Gate count for groove": N params for pulse, M for meter, K for
   pattern — capability thresholds in the found-machines spirit.
2. Every rung is event-driven, dyadic-decay-friendly -> direct
   Morpho/iCE40 path; a hardware rhythm continuator is the program
   thesis made audible.
3. Contrast with M9 token models: do 22M-param token LMs learn
   anything about TIME that a 200-param oscillator bank doesn't?
   The answer either way is a headline.

Status: design only. Build: data prep + L0-L2 first (Mac-scale).


## v1 results (2026-08-29, GMD test split, 20 ms bins, next-bin NLL)

| rung | params | bits/bin | onset F1 (prec) |
|---|---|---|---|
| L0a rate | 1 | 0.600 | - |
| L1 single LIF | 3 | 0.598 | 0 |
| L2 oscillators only | 33 | 0.591 | 0 |
| L0b 8-tap Markov | 9 | 0.577 | 0 |
| L3 taps+counters | 417 | 0.532 | 0.12 (0.79p) |
| L3 + oscillators | 929 | 0.530 | 0.11 (0.78p) |

Findings: (1) single LIF learns nothing (predicted). (2) Short raw
history > pure phase at next-bin range. (3) 417 params of
taps+counters ~ everything; oscillators add 0.002 bits GIVEN
history — the meter prediction fails AT THIS LENS. Diagnosis
(preregistered for v2): next-20ms prediction is dominated by local
micro-structure; meter's value lives at long horizons where phase
outlasts history. v2 eval: multi-horizon prediction (t+k, hidden
intermediates, k to ~1 bar) + continuation quality; plus learned
per-oscillator decay, onset-level metrics, L4 ceiling reference.
Method note: fixed-physics dynamics + trained readouts only —
param counts are honest; first sweep was non-converged (bias/scale)
and is superseded by this one.


## v2 results (2026-08-29; collaborator-sharpened metrics/ladder)

bits/EVENT over rate baseline (GMD test, 20 ms unless noted):

| rung | trainable | state | b/ev | F1@80ms |
|---|---|---|---|---|
| traces (10 dyadic) | 11 | 10 | +0.232 | .00 |
| oscillators only | 33 | 32 | +0.158 | .00 |
| traces+osc | 43 | 42 | +0.257 | .01 |
| FIR-500 | 501 | 500 | +0.227 | .05 |
| nonlinear head | 705 | 42 | +0.394 | .05 |
| CLOCK oracle (16th-phase bins) | 17 | 0 | +0.445 | .00 |
| clock + traces | 43 | 10 | +0.630 | .18 |

FINDINGS
1. THE CLOCK IS MACHINERY, MEASURED: the transport clock alone
   (zero state) beats every self-clocked model; clock+traces are
   nearly ADDITIVE (0.445+0.232 ~ 0.630) — clock = where in the
   cycle, traces = what is being played. In a DAW the clock is
   free: 43 trainable / 10 state extracts 0.63 b/ev given it.
2. Self-clocked oscillators recover only ~35% of the clock's info —
   the measured cost of DISCOVERING pulse vs being handed it.
3. Dynamics compress: 10 traces ~ 500-tap FIR (0.232 vs 0.227).
4. GRID ABLATION (L5 per grid): 80ms +0.277 / 40ms +0.335 /
   20ms +0.394 / 10ms +0.446 — NO saturation by 10 ms; microtiming
   is signal (collaborator's 40ms-suffices guess falsified).
5. Metrical structure lives at the 16TH level, not the beat:
   onset phase concentration R_16 = 0.59 vs R_beat = 0.07
   (drummers occupy all beat slots; oracle/anchoring must target
   the concentrated level). Three oracle bugs found en route
   (per-file phase offset; 1st-harmonic-only sin/cos readout vs
   comb hazard -> phase BINS; anchor level) — recorded, fixed.

DESIGN UPDATES (user+collaborator relay, adopted):
- Contract v1: integer 20ms cells, 0/1, no tokenizer — continuously
  meaningful dynamics, discretely permitted events.
- CLOCK-CONDITIONING LADDER promoted to main axis: BPM-only ->
  +pulse phase -> +bar phase -> full position; each rung prices what
  the transport provides.
- DAW formulation (target instrument): musical-tick lattice,
  subdivision-RATIO oscillators (w_k = r_k w_master; tempo scales
  all dynamics coherently), playhead-as-read-head, swing as
  substrate deformation, loop/seek semantics explicit.
- v3: multi-channel spikes (9-piece kit) on the tick lattice —
  collapsed-to-one-channel occupancy is ~1 hit per 16th, so pattern
  information lives in WHICH drum, not whether.
- Two-grid split (metrical event + microtiming delta) later; GMD's
  score-vs-deviation annotations fit it exactly.
- STILL OWED: free-running stability as first-class metric.


## v3 results (2026-08-29): nine-voice clocked kit process

Voices: Magenta 9-class reduction, x_t in {0,1}^9, 20 ms bins,
clock-conditioned. bits/event over per-voice rate baselines:

| rung | trainable | b/ev |
|---|---|---|
| clock only | 297 | +0.49 |
| clock + own-voice traces | 387 | +1.14 |
| clock + ALL-voice traces | 1107 | +1.31 |
| + same-tick coupling (PL diagnostic) | 1188 | +4.51 |

1. The compression SURVIVES the full kit process — the
   make-or-break held: per-voice structure is far richer than the
   near-saturated any-hit channel and tiny machinery still captures
   it.
2. CROSS-VOICE MEMORY IS REAL: +0.17 b/ev for reading the whole
   kit's traces — every output voice reads all stacks.
3. COINCIDENCE DOMINATES: same-tick coupling adds +3.2 b/ev on top
   of everything (drums are chords). PL bound is diagnostic, not
   causal; v4's required change is a within-tick autoregressive
   factorization (fixed voice order, condition on already-decided
   voices).

FREE-RUN (seed 2 bars -> generate 16; clock+all-traces, 1107
params; 15 test files long enough): rate ratio 1.09 (bar 1) ->
1.23 (bar 16) — mild densification, no collapse/explosion;
generated metrical R16 = 0.39 (human 0.59) — grid-locked but
looser than human; voice-distribution JS = 0.023 bits — stable kit
balance. The learned dynamical drum machine exists at ~1.1k params.

NEXT: within-tick coupling (causal); velocity as a third stage;
tick-lattice/subdivision-ratio port (the DAW instrument); render
generated takes to audio; free-run at the coupled rung.


## v4 (2026-08-29): tick-lattice exact + velocity — collaborator's
## rhythm3_exact.py merged (built remotely, pulled in)

Tick lattice (48 steps/quarter, BPM never a feature — one process,
any host tempo), musical-time trace half-lives, causal
within-tick coupling, drift-resolved free-run:

| rung | trainable | state | b/ev | free-run |
|---|---|---|---|---|
| clock | 261 | 0 | +0.303 | stable |
| +own traces | 351 | 90 | +0.661 | stable |
| +all traces | 1071 | 90 | +0.792 | stable |
| +causal coupling | 1107 | 90 | +0.852 | stable, R16 0.378 (hum .584) |

CORRECTION THAT MATTERS: causal within-tick coupling is worth
+0.06 b/ev — the earlier +3.2 PL 'bound' was almost entirely
information flowing backwards through the conditioning, not
causally available. Preregistration discipline caught it; quote
0.06, never 3.2.

VELOCITY STAGE: per-voice linear head on the same features, 1071
params: RMSE 31.5 vs 34.1 per-voice-mean baseline (MIDI units) —
accent position is linearly learnable, phrase dynamics are not.
Renders: truth.wav (velocity-true round trip — user-verified
'sounds great'); spikeglm_vel.wav (coupled model + learned
velocities at true tempo). Ghost-note flattening was the earlier
'fury of snares' artifact — representation, not model.

NEXT: nonlinear/interaction velocity head; microtiming delta stage;
SNN rung; DAW/live port of the tick-lattice process (the natural
deployment — all state is 90 scalars + 1107 weights).


## v5 design (2026-08-29, user concept): NAVIGABLE MODE SPACE

Diagnosis: traces are FAST state; generation is grid-true but
non-committal because there is no SLOW mode variable ("this groove"
vs "fill"). Human drumming is regime-switching.

v5a — switching spike-GLM (preregistered):
1. Bar features: 9x16 slot-occupancy grids (144-d — the drum-machine
   pattern page, inspectable by eye).
2. k-means K=12 over training bars -> modes are pattern heatmaps.
3. VALIDATION AGAINST LABELS (unsupervised first): GMD marks files
   beat vs fill. PREDICT: >=1 cluster majority-fill; beat/fill
   separation visible without supervision.
4. Mode-conditioned GLM (+K one-hot features). PREDICT: b/ev gain
   over the 'all' rung (0.792) — the mode carries pattern identity
   the traces cannot hold.
5. Generation = NAVIGATION: user/DAW chooses the bar-mode sequence
   (e.g. 7 bars groove cluster, 1 bar fill cluster). Render demo.
Program tie-in: mode = M4 slot (which regime is active), traces =
fast dynamics within it. Later: learned transitions (tiny Markov),
continuous style axes (v5b).


## v5a results (2026-08-29)

- 12 modes over 17,898 training bars; modes = 9x16 pattern
  centroids (inspectable). PREDICTION FAILURE recorded: no
  majority-fill cluster (fills are 2% of bars; k-means organizes by
  density/style). Mode 0 (sparse) is 4.5x fill-enriched (0.09) and
  serves as the break mode; a TRUE fill vocabulary needs the labels.
- Mode conditioning: +1.317 -> +1.398 b/ev (+0.08, 108 params).
  Value is CONTROL, not prediction.
- OPTIMIZATION FINDING: global row shuffle + feature
  standardization lifts the same 1071-param clock+traces model from
  +0.792 (rhythm3_exact per-file fitting) to +1.317 b/ev — rung
  comparisons must be re-run under the stronger optimizer before
  quoting.
- NAVIGATION DEMO: navigate.wav — funk seed, mode path
  7 groove / 1 sparse / 7 groove / 1 sparse, hand-steered.
NEXT: labeled fill submodes; learned bar-boundary transitions;
velocity in navigated renders; continuous style axes (v5b).
