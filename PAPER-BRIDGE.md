# Growing a threshold inside a trained network: function-preserving
# sparsification of a delta-memory language model

*Status: v1.0 (2026-08-27). One seed. All numbers from
`whitebox/runs/train-distill.log` and the analysis commands quoted
inline. Companion program docs: M5.md (the delta-memory operator),
M6.md (the synthesis program this experiment belongs to).*

---

## 1. The problem this solves

The whitebox program's target architecture has two components with
independent pedigrees:

- a **diagonal delta memory** (Longhorn-form; M5.md) as the sequence
  mixer — an online ridge-regression update we can read, calibrate,
  and cost in hardware (78 gates/element ladder-quantized), and
- a **sparse dictionary feature block** (CRATE-form ISTA step) as the
  per-token nonlinearity — the interpretability bet that features
  should be a small set of named atoms, not a dense vector.

Separately, both work. Composed and trained from scratch, they fail —
reproducibly and completely:

| scratch arm | config | val ppl @500 | fate |
|---|---|---|---|
| ReLU-ISTA, overcomplete | dict_expand 2 | 288 | censored (stop rule) |
| ReLU-ISTA, complete | dict_expand 1 | 413 | censored (stop rule) |
| linear dictionary (no threshold) | m2_identity | **9.97 @3k** | healthy |

The threshold is the poison, isolated to one bit: the identical
architecture with the prox removed trains to the best number the
dictionary family had ever posted. Two hypotheses for *why*:

1. **Calibration**: thresholds are mis-scaled at init, gradients die
   before organization. REFUTED — we measured the init pass-rate of
   the prox at 46%, i.e. well-scaled; units were alive and firing.
2. **Dynamics**: the joint training trajectory of (threshold,
   delta-memory) from random init never reaches an organized region —
   the failure is a property of the *path*, not the *destination*.

If (2), there is an obvious treatment with a precedent: the
Transformer-to-Mamba distillation literature shows that architectures
unreachable by scratch training can be reached by **function-preserving
initialization from a trained donor plus staged conversion**. The
experiment below is the smallest possible instance of that idea: the
donor differs from the target by exactly one property (the threshold),
and the conversion has an exact identity at its starting point.

Preregistered interpretations (M6.md, written before the run):

- Student aligns and survives phase B → the scratch failure was
  **reachability**; bridges are the standard tool from here on.
- Student aligns frozen but collapses when unfrozen → genuine
  **co-adaptation conflict** between threshold and delta memory.
- Student cannot align even at τ≈0 → bug in the harness (halt).

## 2. The exact model

**Teacher** — `whitebox/runs/screen-m5diag-ista/ckpt.pt` (9.97 val ppl
at 3k steps, TinyStories, 4k BPE vocab, ctx 256):

- `CausalCRATEM2`: 12 blocks, d = 448, 16 heads (p = 28), tied
  embedding/head, learned positions, input LayerNorm. 13.9M params
  (trainer count).
- Block, in order (BlockM2, variant b, `m2_identity=True`):
  1. `x = z + LonghornMem(z)` — per head, state S ∈ R^{28×28},
     update `S ← (1 − e k²) ⊙ S + e v kᵀ` with per-head learned rate
     `e = σ(w_e·z)`, update magnitude clamped ≤ 0.9; read `y = S q`;
     q/k per-head LayerNorms; output LayerNorm then W_o. Chunked
     scan, CHUNK = 16.
  2. One unrolled ISTA step against an orthogonally-initialized
     learned dictionary D ∈ R^{448×448}: `u = LN(x)`,
     `r = u − z Dᵀ`, `v = z + 0.1 (r D)`. With `m2_identity` both
     prox operators of the full M2b design are absent — this is the
     "linear dictionary": all sparse-coding machinery, no
     nonlinearity anywhere in the network.

**Student** — the same network, weights copied, plus one `PairedProx`
appended to each block's output:

```
phi_tau(u) = [ ReLU(u − tau) ; ReLU(−u − tau) ]   in R^896
decode:      a @ D_p,   D_p in R^{896×448},  init [I; −I]
thresholds:  tau = exp(log_tau), 896 per block, init exp(−20) ≈ 0
```

At τ = 0, `ReLU(u) − ReLU(−u) = u`: the student IS the teacher,
exactly. Verified at init: max per-layer relative error **1.2e-13**.
Added params: 402,304/block × 12 = 4.83M → student ≈ 18.7M.

## 3. Protocol

**Phase A — bridge (1500 steps).** Freeze everything except prox
parameters (D_p, log_tau). Ramp τ toward TAU_MAX = 0.10 on a gated
schedule: the ramp only advances while every layer's relative error
against the frozen teacher stays under TOL = 0.05; it halts and lets
realignment catch up otherwise. Alignment loss per layer: relative
MSE + cosine distance (magnitude matters — the delta memory's write
values are downstream of these activations). AdamW 3e-4 on prox
params only.

**Phase B — joint (3000 steps).** Unfreeze all. Loss = CE + w·align,
with w = 0.5 decaying linearly to zero at step 1500. LR cosine
1.5e-4 → 1.5e-5. The alignment term is scaffolding, deliberately
removed halfway: the second half is pure language modeling through
the threshold — exactly the gradient regime that killed the scratch
runs.

## 4. What happened

**Phase A** (prints every 150 steps; align = summed per-layer loss,
maxrel = worst layer's relative error, act = fraction of the 896
atoms active):

| step | align | maxrel | tau | act |
|---|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.000 | 0.500 |
| 150 | 0.019 | 0.0019 | 0.0125 | 0.495 |
| 300 | 0.040 | 0.0032 | 0.025 | 0.489 |
| 450 | 0.074 | 0.0058 | 0.0375 | 0.483 |
| 600 | 0.136 | 0.0115 | 0.050 | 0.478 |
| 750 | 0.179 | 0.0143 | 0.0625 | 0.472 |
| 900 | 0.236 | 0.0184 | 0.075 | 0.466 |
| 1050 | 0.308 | 0.0243 | 0.0875 | 0.460 |
| 1200 | 0.372 | 0.0290 | **0.100** | 0.454 |
| 1350 | 0.341 | 0.0262 | 0.100 | 0.456 |
| 1499 | 0.316 | **0.0239** | 0.100 | 0.458 |

The ramp reached full threshold without a single gate failure —
maxrel peaked at 0.029, never approaching the 0.05 gate — and
alignment *improved* for 300 further steps once τ stopped moving.
End of phase A: a genuinely sparse network (45.8% activity) that
computes the teacher's function to within 2.4% everywhere.

**Phase B** (prints every 250 steps; val_ppl on 25 held-out batches):

| step | val_ppl | act | maxrel (vs teacher) |
|---|---|---|---|
| 0 | 15.831 | 0.457 | 0.025 |
| 250 | 10.267 | 0.460 | 0.044 |
| 500 | 10.312 | 0.460 | 0.050 |
| 750 | **9.952** | 0.461 | 0.055 |
| 1000 | 9.400 | 0.461 | 0.063 |
| 1250 | 9.389 | 0.461 | 0.071 |
| 1500 | 9.406 | 0.461 | 0.098 |
| 1750 | 9.193 | 0.460 | 0.126 |
| 2000 | 8.828 | 0.460 | 0.132 |
| 2250 | 8.811 | 0.460 | 0.147 |
| 2500 | **8.434** | 0.460 | 0.146 |
| 2750 | 8.774 | 0.460 | 0.152 |
| 2999 | 8.724 | 0.460 | 0.162 |

Reading the table:

- **Crossed the teacher (9.97) at step 750**, while the alignment
  anchor was still partially engaged — improving *despite* being
  pulled toward the teacher's function.
- **The anchor released at step 1500 and nothing collapsed.** The
  preregistered failure mode had its moment and declined it; the
  first fully-unanchored print *improved* by 0.21.
- **Activity is pinned at 0.460 throughout.** The model never won by
  de-sparsifying; τ stayed at its target. The rising maxrel is
  departure *above* the teacher, not drift.
- Final band 8.4–8.8. A fresh 25-batch eval draw on the final
  checkpoint gave **8.18**, confirming the late prints wobble by
  ±0.2–0.3 from eval sampling; quote "8.5 ± 0.3", never the best
  print. (A 100+-batch eval is owed for a tight headline number.)

Reference points on the same backbone, same 3k-step schedule class:
linear dictionary 9.97, MLP champion 8.48, this sparse student
8.4–8.8. Of the 1.49-ppl "nonlinearity dividend" between the linear
dictionary and the dense MLP, sparse coding claimed essentially all
of it.

## 5. Verdict

**The scratch failure was reachability, not incompatibility.** Pure
cross-entropy through a hard threshold — the exact gradient pathway
that produced 288-and-climbing perplexity from random init, twice —
steadily *improves* the model when the trajectory starts at an
organized function. The threshold and the delta memory have no
intrinsic conflict; random-init joint training just cannot find the
region where they cooperate.

The artifact is as important as the number:
`whitebox/runs/distill-prox/ckpt.pt` is a working sparse-dictionary
language model on the delta-memory backbone — diagonal regression
memory, dictionary residual steps, paired-atom soft threshold, and
nothing else. No MLP, no softmax, no attention. Every activation is
either a memory read with known semantics or one of 896 nameable
atoms per layer of which ~46% fire per token. This is the
interpretability substrate CRATE promised, on the best-understood
mixer we have.

## 6. Honesty notes

1. **Not schedule-matched.** The student consumed the teacher's 3k
   steps + 1.5k alignment + 3k joint. The MLP champion's own 7.5k
   checkpoint reads 6.85 — at equal total compute the dense MLP still
   wins. The claim is "sparse coding reaches dense-MLP quality via a
   bridge", not "prox beats MLP".
2. **Parameter count went UP** (13.9M → 18.7M; +4.83M of prox). The
   student outweighs the 13.4M champion it matches. See §7.1 for the
   11k-param version this motivates — and the measurement showing it
   is not free.
3. **One seed**, one teacher, one τ target. The gated ramp never
   stressed its own machinery (no halt was ever triggered); TOL and
   the ramp shape are untested degrees of freedom.
4. **Eval noise** dominates the last few prints (8.43/8.77/8.72/8.18
   across draws). Any external quote needs the big-eval first.
5. **Post-hoc D-snapping fails.** The learned decode matrices sit
   close to their [I; −I] init (off-diag RMS ≈ 0.004, diagonals
   1.02–1.07, worst entry 0.17) — but snapping them exactly to
   [I; −I] explodes val_ppl 8.18 → 40.4. The small drift is
   load-bearing; twelve layers of ~5% rescaling plus broad 0.004-rms
   crosstalk compound into the function. Near-identity in norm ≠
   dispensable.

## 7. Future experiments, in priority order

1. **Frozen-D bridge (the 11k-param prox).** Re-run phases A/B with
   D_p frozen at [I; −I]; only log_tau (10,752 params total) learns,
   plus the backbone in phase B. If it reaches the 8.x band, the
   entire sparsification costs 11k params and the parameter-parity
   asterisk of §6.2 disappears. §6.5 shows this must be trained-in,
   not rounded-in. Cheap: same runtime as this experiment.
2. **Atom analysis.** The point of the exercise. For each layer's
   896 atoms: firing rate distribution (dead atoms? hubs?),
   token/positional selectivity, cross-layer atom correlation, and
   whether the paired structure (+i vs −i) learned antonymic or
   unrelated roles. Deliverable: a named-atom table for at least one
   layer, or the honest finding that atoms are polysemantic at this
   scale.
3. **Big-eval** (100+ batches) on the final checkpoint → the number
   any external page quotes.
4. **Schedule-matched control**: MLP champion trained under
   teacher-3k + 6.5k continuation vs this student — prices the
   bridge fairly at equal total steps.
5. **Repeat seed** of the whole bridge (different data order and
   prox init) — the reachability claim currently rests on n=1.
6. **τ beyond 0.10**: extend the ramp (0.15, 0.20, ...) in phase-A
   style on the trained student and trace the activity-vs-ppl curve.
   At what sparsity does quality actually break? 46% is where we
   stopped, not a measured frontier.
7. **Overcomplete via bridge**: the scratch failure's original arm
   (dict_expand 2, 896 atoms pre-pairing → 1792 paired). Bridge from
   the same teacher with D_p ∈ R^{1792×448} init [I;−I;0]-padded? No
   exact identity exists for the extra atoms — design question:
   grow them from zero rows (exact identity preserved) and let
   phase B recruit them. Tests whether overcompleteness itself was
   ever the problem or just another unreachable-from-scratch region.
8. **Bridge × spiking ladder** (M6 program convergence): the same
   gated-ramp machinery applied to the M2 ternary quantization rungs
   (q first, then q+k) on the delta memory — the protocol transfers
   verbatim; "uncalibrated 4→2→1" remains banned.
9. **Longer bridges**: MLP champion → prox student (no identity
   construction; distill via phase-A alignment on block outputs
   only), and the 6.04 transformer → Longhorn student. Each tests
   the method's range beyond the one-property case.

## 8. Reproduction

```
# bridge (both phases, ~20h wall on shared M-series MPS):
python3 -m whitebox.distill_prox            # defaults: 1500 + 3000
# teacher expected at whitebox/runs/screen-m5diag-ista/ckpt.pt
# student lands at whitebox/runs/distill-prox/ckpt.pt
# log: whitebox/runs/train-distill.log
```

The init-identity assertion (must print ≤ 1e-6; observed 1.2e-13)
runs before any training and is the harness's own bug gate
(preregistered interpretation 3).
