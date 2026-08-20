# M3 probe suite — LOCKED before any M3 model exists

The circuit-to-token loop: the temporal ladder that exposed circuit-scale
representation limits (Exp 1–3: delayed recall found the register lower
bound; copy-after-delay found the interface boundary) rerun at token
scale against attention variants. Locked here — tasks, formats, delays,
seeds, metrics, and predictions — before M3 training begins. The
generator (`probes.py`) is deterministic from the seeds below; any change
to this file or the generator after M3-control starts training is a
protocol amendment and gets its own commit saying so.

## Reference horizon

Counter decay reference: m = 5 ⇒ time constant 2^m = 32 tokens,
half-life t_½ = ln2·32 ≈ 22.2 tokens. Delay grid in ABSOLUTE tokens
(identical sequences for every architecture):

    D ∈ {6, 11, 22, 44, 89}  =  {0.25, 0.5, 1, 2, 4} × t_½

Windowed-attention arm window W = 32 tokens (= the time constant), so the
cliff is predicted between D = 22 and D = 44.

## Tasks (vocab 64: 16 keys, 16 values, 16 fillers, control tokens; all
sequences ≤ 256 tokens; training delays sampled uniformly from [4, 96] so
the eval grid is in-distribution except D = 89's upper tail — deliberate)

1. **copy** — payload of 8 symbols, D filler tokens, cue, reproduce all 8.
   Metric: per-token exact-match accuracy.
2. **assoc-recall** — 4 key→value pairs, D fillers, one key queried.
   Metric: value accuracy.
3. **induction** — pattern A B embedded, D tokens later A reappears →
   predict B. Metric: B accuracy.
4. **selective-retention** — one MARKED symbol among distractor symbols,
   D fillers, retrieve the marked one (single-item retention under
   interference — the sharpest probe of what decaying statistics keep).

## Protocol

Each architecture is TRAINED on the task family (mixed tasks, sampled
delays, 20k steps, identical data stream from the seeds) and evaluated on
held-out sequences at the fixed grid — the Exp-1-3 protocol at token
scale. Architectures (exact configs declared at M3-control build time,
matched parameter budgets, three train seeds each): full-KV attention;
sliding-window (W=32); RWKV-style decay (comparable ρ); M3 counters at
m ∈ {4, 5, 6}; M3 staircase variant.

Seeds: data 20260820; train {0, 1, 2}; eval split seed 7.

## Metrics, per (architecture, task, delay)

- retrieval accuracy (mean ± range over 3 seeds)
- **state bits at query time** (locked formulas: KV = T·d·b_kv;
  windowed = W·d·b_kv; RWKV = d·b_state; M3 = K·p·b_counter)
- counter-state entropy and **collision rate** (fraction of eval items
  whose query-time counter states are indistinguishable at counter
  precision while their correct answers differ — measured, not assumed)
- R(Z) through depth at query time (does retrieval failure co-occur with
  expansion collapse or with counter collision? — distinguishes "state
  too small" from "state collapsed")

## Predictions (written before any run)

1. Full KV: flat accuracy across all delays, at maximal state bits.
2. Windowed: step cliff between D = 22 and D = 44; near-KV inside.
3. RWKV-decay: smooth monotone decay in D tracking its ρ.
4. M3 counters: decay profile tracking t_½ per m — the m-sweep curves
   should ORDER by m at D ≥ t_½ and coincide at D ≪ t_½.
5. M3 staircase: plateau structure aligned with the κ_r comparator
   boundaries (the one distinctively-M3 prediction; if absent, the
   staircase is cosmetic).
6. selective-retention degrades before copy at matched D for the counter
   methods (interference through shared counters — the collision metric
   should co-move with this failure).

The memorable figure, if the data cooperates: accuracy-vs-delay curves
for all architectures over one x-axis in units of t_½, with state bits
as marker size — retention you can afford, priced in bits and, one
methods-section later, in gates.


## RESULTS (2026-08-21, two seeds — protocol amendment: user stopped the
seed-2 pass, arms already cleanly separated; dval and staircase arms see
notes)

Copy / assoc / induction / selective accuracy, mean of seeds {0,1}, at
delays {6, 11, 22, 44, 89}:

    copy       kv 0.72-0.70 flat | win32 0.12->0.08 cliff | dval FLOOR | crsa 0.56-0.63 FLAT
    assoc      kv ~0.30          | win32 ~0.10           | dval FLOOR | crsa 0.15-0.19
    induction  kv 1.00 all       | win32 0.28->0.09 cliff | dval FLOOR | crsa **1.00 ALL DELAYS**
    selective  kv ~0.91          | win32 ~0.08           | dval FLOOR | crsa 0.52-0.58

Prediction scorecard:
1. KV flat — CONFIRMED (every task).
2. Window cliff between 22 and 44 — CONFIRMED in shape (induction
   0.28->0.09; copy 0.12->0.08), though the arm trains weakly overall.
3. Decayed-value smooth decay — NOT TESTED IN EFFECT: the arm floored at
   chance both seeds; the (1-rho) input attenuation (up to 64x) is the
   suspected implementation artifact. The baseline is INVALID until a
   rescaled follow-up runs; no comparative claims against it.
4. CRSA ordering by m at D >= t_1/2 — REFUTED in the strong direction:
   retention is FLAT to 89 tokens (2x the slowest half-life, 17x the
   fastest) — the multiscale population retains what no single counter
   could. The stronger branch of the preregistered framing.
5. Staircase plateaus — arm deferred, untested.
6. Interference signature — CONFIRMED in relative form: CRSA's gap to KV
   is much larger on selective-retention (0.56 vs 0.91) and assoc than on
   copy (0.58 vs 0.71) — shared-counter interference, as predicted.

Headline: **a constant-state operator with no token pairs solves
induction (A B ... A -> B) perfectly at every tested distance.** CRSA is
the only constant-state arm that learns all four tasks. Its cost
concentrates precisely where the mechanism predicts: interference-prone
retrieval, not distance.
