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
