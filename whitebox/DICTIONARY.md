# The overcomplete dictionary fork (formalization v2 — corrected)

Motivation: published CRATE never beat attention transformers; this line
inherits that ceiling. The mathematical point (sharper than "capacity"):
**an overcomplete dictionary with the prox disabled is still a linear
map** — the hypothesis is overcomplete NONLINEAR sparse coding, not
dictionary size. Wording correction on record: this is NOT the program's
first nonlinear arm (M0's ReLU/ISTA and the spike-prox models were
nonlinear); it is **the first CRSA arm with an active, transformer-scale
overcomplete nonlinear feature block**.

## Canonical block (block-local form — the factorial's design)

Post-CRSA representation x_ℓ ∈ R^d; block-local code a_ℓ ∈ R^q, q = r·d,
r ∈ {1, 2, 4}; dictionary D_ℓ ∈ R^{d×q}. Local objective: nonnegative
sparse coding, a* = argmin_{a≥0} ½‖x − D a‖² + λ‖a‖₁. One exact ISTA
iteration from a⁽⁰⁾ = 0 (cleanest first implementation):

    a⁽¹⁾ = ReLU( η D^T x − η λ 1 )
    x̂    = D a⁽¹⁾
    z'   = x + γ (x̂ − x)          γ = 1: sparse reconstruction directly;
                                   0 < γ < 1: relaxed update (stability)

CRSA and its counters stay d-wide; only the block-local feature bank
expands. Honest divergence from CRATE: the sparse object is an internal
latent; the inter-block state is its d-dim reconstruction — CRATE's
eqs. 14–17 pass the code itself onward and do NOT transfer unchanged.
(The persistent cross-layer-code variant already implemented and
screening as `screen-dict4` is a defensible alternative design — possibly
stronger — but it is a DIFFERENT interpretation; the factorial uses the
block-local form.)

## Overcomplete geometry (corrected)

For q > d, D^T D ≈ I_q is RANK-IMPOSSIBLE — never assess it. The premise
becomes a tight-frame condition: D D^T ≈ c·I_d (unit columns ⇒ c = q/d),
plus low mutual coherence. ISTA stability: 0 < η < 2/‖D‖₂². Logged per
layer, every eval: η‖D‖₂² (must stay < 2), ‖DDᵀ − cI_d‖_F, mutual
coherence, reconstruction error, sparsity, **dead-atom fraction and
activation-frequency distribution** (aggregate sparsity cannot show
whether 1,792 atoms function as a much smaller dictionary).

## The crucial factorial (fixed d = 448; allocation comparison ONLY after)

| arm | dictionary | prox | isolates |
|---|---|---|---|
| F1 | q = d | identity | current control |
| F2 | q = d | active | nonlinearity alone |
| F3 | q = 4d | identity | factorization/params alone — D(Dᵀx) is STILL LINEAR |
| F4 | q = 4d | active | the full overcomplete sparse coder |

If only F4 improves, the gain is thresholded feature regions — not a
larger matrix. Parameters are NOT matched across arms by design
(mechanism first); the allocation question (d=672,q=d vs d=448,q=4d)
comes after, and eventual reporting is PPL at matched tokens, PPL per
FLOP, and hardware-normalized cost — the current comparison is
parameter-matched only (the persistent 4d stream also adds dictionary
ops and inter-block bandwidth).

## Plateau protocol (supersedes screen-verdicts; applies to every arm)

Screens eliminate catastrophic configurations only. Checkpoints retained
for every healthy configuration. Credible arms train on full-horizon
schedules and stop on a VALIDATION PLATEAU, not a step count: <0.5%
relative improvement over 2,000–3,000 steps, after at least one LR
reduction; report the best validation checkpoint. Both numbers always:
PPL at matched tokens AND lowest plateau PPL. Standing consequence:
lower-LR and extra-heads are screen-negative, NOT proven incapable of a
better asymptote; the d=672 baseline (19.62 → 17.66 over steps
3,000→4,000) was nowhere near plateau.

## Screen results (persistent-code variant) + a schedule confound

screen-dict4 (persistent-code design, d448/4d, 14.0M): gap to the d672
baseline CONVERGED monotonically +6.0 → +3.1 → +1.8 → +1.3 → +1.0
(steps 500–2500), finished 21.24 vs 19.62 — the campaign's only
converging trajectory; healthy throughout (sparsity 0.58, no collapse).
CONFOUND, recorded before further reading: screens carry 3,000-step
cosines (end-of-schedule LR boost at the finish) while the baseline's
19.62 is a MID-5,500-schedule reading — LR differs at matched steps,
cutting both ways. Trajectory-gap comparisons across differing schedules
are indicative only; the plateau protocol is the resolution. Verdict:
retained for plateau evaluation.

## The MLP control (the decisive three-way)

BlockMLP: h' = x + W2·GELU(W1·LN(x)), W1: d→4d untied — established,
dense, NOT white-box. At fixed d it carries ~2× the tied dictionary's
feature params (8d² vs 4d²): compare at fixed d AND fixed total params.
Three-way at d=448, plus outcome table (agreed):

    softmax + 4d MLP   |  CRSA + 4d MLP   |  CRSA + 4d overcomplete prox

| result | interpretation |
|---|---|
| MLP and dictionary both improve | missing capacity was nonlinear feature expansion |
| MLP improves, dictionary does not | CRATE's tied sparse coder is the bottleneck |
| dictionary matches MLP | white-box sparse coding provides transformer-grade capacity |
| dictionary beats MLP | very strong result for the derived architecture |
| neither improves | look back toward CRSA, optimization, or another bottleneck |

If CRSA+MLP matches or beats softmax+MLP, CRSA stands independently of
CRATE's sparse-coding limitations. Trade named: MLP eases optimization
and scaling; dictionary keeps the exact local objective, tied maps,
sparse activations, and the spike-hardware route.

## FACTORIAL RESULTS (2026-08-21, fixed d=448, 3,000-step screens)

| | identity prox | active prox |
|---|---|---|
| q = d (6.8M) | 22.32 | 20.60 |
| q = 4d (14.0M) | 22.40 | **19.75** |

Formal decomposition in validation NATS (agreed): L1..L4 = 3.105 /
3.025 / 3.109 / 2.983. Prox at q=d: −0.080 nats. Width with identity:
+0.004 (zero). Width with active prox: −0.042. Interaction: −0.046
nats — THE number: the active prox makes the additional atoms useful.
PPL terms: prox alone +7.7%, width-after-prox +4.1%, F4 over F1 +11.5%.
Conclusion wording (agreed): *overcompleteness has no value as a linear
factorization; it becomes useful only when the proximal threshold
partitions the input space into selectively activated feature regions.* The sharpest-conclusion wording below now has
measured cells. Additional verdicts: the block-local a₀=0 form (19.75)
beats the persistent-code variant (21.24) — canonical design confirmed,
plateau run switched to it; F4 vs d672 baseline is NUMERICAL PARITY under different cosine
horizons (19.75 vs 19.62) — not a victory; the factorial proves
short-budget synergy, not a lower asymptotic floor. Block-local vs
persistent (19.75 vs 21.24; same seed/schedule): repeatedly re-encoding
each layer beats carrying a wide code whose atom coordinates change
between layer dictionaries. Screen-scope caveats
apply as everywhere (3k reads, plateau protocol governs).

## The eventual architecture ladder (plateau-protocol comparisons)

plain CRSA | CRSA + overcomplete prox | CRSA + conventional MLP |
causal TSSA + the same MLP | softmax + the same MLP.
Three-way outcome map (agreed): CRSA beats TSSA ⇒ dyadic
forgetting/multiscale adds capability; matches ⇒ CRSA preserves TSSA
quality while gaining bounded counters + spike compatibility + hardware
simplicity; TSSA beats CRSA ⇒ forgetting/event compression has a
measurable quality cost; both beat softmax ⇒ token-statistics mixing is
the underlying advantage; softmax wins ⇒ constant-statistics attention
trades retrieval capacity for state efficiency.
FIRST SCREEN RESULT: CRSA+MLP = **16.79 @3000** — best screen number of
the campaign (F4 19.75, d672 baseline 19.62); softmax+MLP running; CRSA-UNIFORM+MLP queued
(naming rule: uniform prefix measure with CRSA's router/price/tied-agg
is NOT published TSSA — it lacks soft membership, temperature, bias,
untied out; the strongest ladder = CRSA-leaky / CRSA-uniform / literal
TSSA Algorithm 2 verified numerically equivalent on shared weights —
the literal arm is OWED). Param caveat on 16.79: the MLP model carries
~10M more params than the 14M dictionary arm — strong trainability +
feature-bottleneck support, NOT proof that MLP beats dictionary — separating
nonlinear capacity, sparse-coding value, CRSA's decayed statistics, and
token-statistics attention itself.

## The strongest potential conclusion (pre-written)

*CRSA's temporal operator was not the scaling bottleneck; quality
required an overcomplete, proximally gated feature dictionary that
restores nonlinear sparse feature expansion between recurrent mixing
steps.* A coherent architecture: CRSA does constant-state temporal
mixing; the overcomplete prox does nonlinear feature expansion.

## STAA-SNN triage (arXiv:2503.02689 — vision SNN, BPTT+surrogate; not
causal-LM, not forward-only, not objective-derived; its attention embeds
convs/sigmoid/ReLU/LN, so no cleaner spike-attention for CRSA)

Transfers: (1) adaptive permeability → eventual learned SELECTION among
dyadic decays (retains shift-only hardware; never arbitrary float ρ);
(2) stochastic bypass (TSRD) → randomized identity/prox routing during
spike conversion — a stochastic generalization of our shadow protocol;
(3) their non-monotonic internal-width optimum supports ratio ablations
(r ∈ {1,2,4}) but says nothing about q=4d optimality here. Their energy
numbers are op-count models with assumed pJ values; placed-and-routed
measurement remains the stronger standard.
