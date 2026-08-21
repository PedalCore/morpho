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
