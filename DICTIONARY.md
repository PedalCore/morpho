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
SCREEN RESULTS: CRSA+MLP = 16.79 | **softmax+MLP = 12.54** — the
outcome map's fifth row fires: *constant-statistics attention trades
retrieval capacity for state efficiency*. With a transformer-grade
feature block, pairwise attention wins by 4.25 ppl (34% rel); the
M3-control result (CRSA 13.78 > softmax 14.22) is hereby REINTERPRETED
as a feature-starved-regime comparison — neither operator could exploit
retrieval capacity without a real MLP. Consistent with the probe
suite's selective-retention gap (0.56 vs 0.91): the cost was always
identity retrieval, and the MLP unlocks softmax's use of it. CRSA's
standing claims narrow accordingly: state efficiency (384 counters vs
growing KV), hardware simplicity, derivation fidelity — at a now-
measured quality cost in feature-rich regimes. softmax+MLP running; CRSA-UNIFORM+MLP queued
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


## MLP three-way — COMPLETE (23.7M, 3,000-step screens, one seed)

softmax+MLP **12.54** | CRSA-leaky+MLP 16.79 | CRSA-uniform+MLP 24.57.

Row 5 fires: constant-statistics attention trades retrieval capacity for
state efficiency (softmax wins by 4.25 with a real feature block;
M3-control's CRSA-beats-softmax reinterpreted as a feature-starved-regime
comparison — the probes' selective-retention gap predicted this). Row 1
fires: the dyadic ladder is worth **+7.8 ppl over uniform prefix
statistics** — the campaign's largest single effect; uniform all-history
statistics dilute as context accumulates, the leaky ladder stays live.
Ladder ordering: softmax > CRSA-leaky >> CRSA-uniform. Caveats standing:
CRSA-uniform ≠ published TSSA (literal Algorithm-2 arm with numerical
equivalence owed); CRSA+MLP's 16.79 carries the +10M-params caveat vs
the dictionary arm; one seed; screen scope.


## Matched-20k results (overnight; NOT plateaus — both still descending
~1%/1.5k steps at schedule end, so true floors are lower)

CRSA + overcomplete dictionary (14.0M): **8.40** | plain CRSA d=672
(13.8M): 9.72. The dictionary wins by 1.32 at matched horizon — the
converging screen trajectory cashed out (parity at 3k, decisive at 20k):
slower to organize, lower floor. Screen-vs-20k gaps (19.75→8.40,
19.62→9.72) empirically validate every screen-scope qualification in
this file. Comparisons to the 5,500-step baseline table (RWKV 6.42 etc.)
are NOT licensed across budgets; within-pair matched-horizon comparison
only. One seed; longer schedules needed for true plateau per protocol.


## Literal TSSA arm — COMPLETE (26.1M, 3,000-step screen, one seed)

The owed arm is paid. Implementation is the published causal form (ToST
arXiv:2412.17810, eqs. 10/28/31 + C.1): soft head membership pi =
softmax_K over (1/2eta)||U_k^T z||^2 with learnable temperature eta and
learnable per-position bias b_{k,j}; causally accumulated
membership-weighted second moments normalized by membership counts
n_{j,k}; diagonal scaling 1/(1+s); untied learnable output W (the
overparameterized form "used in practice"). Verified BEFORE launch per
the naming rule: vectorized form numerically equivalent to a per-token
loop transcription of eq. 28 (max diff 6e-8) and exactly causal —
causality alone proves nothing; equivalence proves it is TSSA.

FULL LADDER (matched d=448 + identical 4d MLP, 3,000 steps, one seed):

    softmax 12.54  |  CRSA-leaky 16.79  |  CRSA-uniform 24.57  |
    literal TSSA 25.43

Matched-step trajectory: literal TSSA tracked CRSA-uniform in lockstep
the entire run (500: 46.0 vs 46.8; 1000: 38.4 vs 38.1; 2000: 29.0 vs
28.2; final 25.4 vs 24.6) and finished +0.86 behind it — within screen
noise of equal, decisively behind the leaky arm.

ATTRIBUTION SHARPENED: the campaign's largest effect (+7.8 ppl,
CRSA-leaky over CRSA-uniform) is now attributable to DECAYED VS
ALL-HISTORY STATISTICS, not to routing machinery. The published extras
(soft membership, temperature, position bias, untied W, +2.4M params)
recover none of the gap — both all-history arms dilute as context
accumulates regardless of how tokens are routed to heads. CRSA's dyadic
forgetting is the load-bearing modification to its ancestor, on this
corpus, at this scale, one seed, screen scope. Param caveat (corrected wording): the
literal arm carries +2.4M over the other MLP arms (untied W is the
published form's own choice) and still trails — this shows the deficit
is NOT caused by having fewer raw parameters; it does NOT prove
capacity is irrelevant, because parameter placement and optimization
still matter.


## Ladder in cross-entropy nats/token (agreed convention) + the close

CRSA-leaky vs CRSA-uniform: **-0.381**. CRSA-leaky vs literal TSSA:
**-0.415**. TSSA vs CRSA-uniform: +0.034 (same performance class at
one seed). Softmax vs CRSA-leaky: -0.292.

Result vs mechanism, kept distinct: the CONTROLLED RESULT isolates
exponential temporal weighting as the load-bearing CRSA modification.
The PROPOSED MECHANISM — uniform all-history second moments become
increasingly stale/diluted while dyadic exponential measures remain
responsive — is a hypothesis, testable later by measuring adaptation
after a controlled distribution change or by sweeping context length.

Defensible conclusions (agreed): (1) softmax retains a substantial
retrieval advantage; (2) dyadic exponential forgetting improves CRSA
by 7.78 ppl over its otherwise matched uniform-prefix ablation;
(3) literal TSSA clusters with CRSA-uniform despite +2.4M params;
(4) the 0.86 gap between the all-history arms supports no claim at one
seed; (5) CRSA's contribution beyond TSSA is now measured. The 6e-8
equation-level equivalence licenses reporting this as a literal
causal-TSSA reproduction, not a uniform-CRSA proxy.

PAPER-READY STATEMENT: *At matched width, feature block, data, seed,
and 3,000-step schedule, dyadic-leaky CRSA improves by 0.381
nats/token over its uniform-measure ablation and by 0.415 nats/token
over a numerically verified reproduction of causal TSSA, despite TSSA
carrying 2.4M additional parameters. This identifies multiscale
exponential temporal weighting — not routing simplification or
parameter count — as the principal source of CRSA's advantage over
cumulative token-statistics attention in this setting.*

PAGE-LEVEL SENTENCE (agreed): *On a one-seed, 3,000-step TinyStories
screen with identical MLP feature blocks, softmax led at 12.54 PPL,
dyadic-leaky CRSA reached 16.79, and both all-history statistics
operators trailed at 24.57-25.43. CRSA therefore recovers substantial
capability over its direct cumulative-statistics ancestor while
preserving bounded recurrent state, but does not eliminate the
retrieval advantage of full attention.* An honest trade frontier — a
stronger close than claiming CRSA universally replaced attention.


## Extension run (floor hunt) — STOPPED EARLY BY DECISION (censored)

plateau-dict-ext20k: warm restart of the 8.40 checkpoint at peak LR
1.5e-4 (one LR reduction, per protocol). Warm-restart bump to 8.68,
recovered, crossed below the parent at step ~4,500, and descended
steadily to **8.077 at step 8,500 of 20,000** when the run was stopped
by decision to redirect compute to the M4 probe ladder. CENSORED
RESULT: not a plateau — still descending ~0.045/1k steps at stop; the
schedule-limited floor estimate was ~7.8-8.0. Rationale recorded: the
first 20k bought 19.75->8.40, this segment bought 0.32 in 8.5k —
optimization is nearly exhausted; the remaining gap to the retrieval
class is architectural (M4). Best constant-state checkpoint is now
plateau-dict-ext20k/ckpt.pt (8.08, 28.5k total steps). Generation
qualitatively improved: complete story arcs with endings now occur
(one drift-free full story observed); name drift persists in other
samples (Billy->Sarah).


## CRATE-alpha (arXiv:2405.20299, NeurIPS 2024) — independent convergence

Yang et al. scale CRATE for vision by identifying the ISTA block (not
attention) as the bottleneck; vanilla CRATE gains only +0.5% B->L.
Their three modifications: (1) overcomplete dictionary, C=4, prox,
TWO prox steps from A0=0 (+5.3%) — INDEPENDENTLY CONVERGENT with this
fork's q=4d active-prox design (our factorial additionally isolates
that overcompleteness pays ONLY through the prox — a decomposition
their paper does not make); (2) DECOUPLED dictionary — encode with D,
decode with a different learned D-hat (+2.0%) — NOT in our block
(ours ties encode/decode); (3) residual around the sparse block
(+0.7%) — already our gamma-mixed block-local form. Results: 76.5
B/32 (vanilla 68.5), 83.2 B/8, 85.1 L/8, 72.3 zero-shot Huge.

Language (their Table 4, OpenWebText/nanoGPT): CRATE-alpha-base 120M
CE 3.14 vs GPT-2-base 124M CE 2.85 — they fixed the feature half and
STILL trail a standard transformer at matched size, with tied MSSA
untouched. Consistent with this program's diagnosis: the CRATE line's
second deficit is retrieval/binding in the tied attention (our
query-map + slots work). The two programs are complementary halves.

CANDIDATE UPGRADES QUEUED (cheap screens when the LM ladder resumes):
decoupled D-hat decode; two-step prox (already on the deferred list,
now externally evidenced). Both compose with the dictionary model.


## Neuron-interpretability dividend (arXiv:2410.16443, CPAL 2025 oral)

Bai & Ma: CRATE-architecture language models show up to 103% relative
improvement in NEURON-LEVEL interpretability over post-hoc dictionary
methods (SAE route), consistent across layers — in-architecture sparse
coding makes neurons activate distinctively on relevant tokens.
Program relevance: external, citable support for the third leg of the
MLP-vs-dictionary trade (alongside the exact local objective and the
spike/hardware route): the dense MLP control's ppl advantage forfeits
measured interpretability that post-hoc methods only partially
recover. No architectural implications (attention untouched there);
the binding program is unaffected.
