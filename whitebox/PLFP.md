# PLFP — Parallel Local Forward Perturbation (formalization)

The M3-local/forward axis made precise: a scan-parallel, forward-only,
three-factor rule that optimizes explicit local rate-reduction and
predictive objectives **through exact hard spike events** — no backward
pass, no cross-layer gradients, no STE anywhere in the training
computation. Narrower and more defensible than "Forward-Forward for
transformers" (FF gives the local-objective precedent, not the strict
hard-spike no-reverse implementation; CLAPP gives the causal predictive
plasticity precedent).

## 1. Block-local causal objective

Block ℓ predicts its own future code from clean forward-pass activity
(targets are observed local activity — the stop-gradient is conceptual;
there is no backward graph):

    p_{ℓ,t} = P_ℓ z_{ℓ,t}
    L^pred_ℓ = −log [ exp(p_{ℓ,t}·z_{ℓ,t+1}/T) /
                      Σ_{u∈{t+1}∪N_t} exp(p_{ℓ,t}·z_{ℓ,u}/T) ]

with N_t negatives shuffled from other sequences. Plus the assigned
white-box terms:

    L_ℓ = L^pred_ℓ + λ_c·ΔR^c_{ℓ,attn} + λ_r·½‖x − Da‖² + λ_h·(s̄ − r*)²

Two measurable jobs per block: predict future local activity; perform
the assigned compression/sparse-coding role. **λ_c is the price dial:
sweeping it measures the capability cost of objective faithfulness** —
the fourth outcome of the gap-attribution ladder, now with a knob.

## 2. Credit from forward evaluations only

v = pre-prox input. K Rademacher/Hadamard probes ξ_k, antithetic hard
evaluations (real spike transitions, not a surrogate):

    z^{k,±} = Q_τ(v ± ε ξ_k)
    d_k = [L_ℓ(z^{k,+}) − L_ℓ(z^{k,−})] / 2ε
    ĝ_v = (1/K) Σ_k d_k ξ_k

Unbiased for the gradient of the locally smoothed hard-spike objective
under isotropic directions; variance grows with perturbed dimension, so
**perturb node/router groups (dimension d or head width p), never
weights (d²)**.

**ε from measured margins — the forward-only analogue of the M2
calibration lesson.** Q is piecewise constant: probes that don't cross a
threshold return zero credit. ε is set per channel from the observed
margin distribution |v − nearest grid boundary| — and the machinery that
measures those distributions already exists (`calibrate.py` captures
exactly these pre-prox distributions; the M2-spike dead-zone failure is
the same trap in gradient form). K = 8–16 orthogonal probes per head to
start.

## 3. Three-factor synaptic rules

    W_ji ← W_ji − η ĝ_{v_j} x_i          (pre event × post perturbation
                                          identity × head-local loss diff)
    τ_j  ← τ_j + η_τ ĝ_{v_j} + η_h (s̄_j − r*_j)      (credit + homeostasis)
    D    ← D + η_D r a^T, columns renormalized       (EXACT frozen-code
                                          descent on ½‖x−Da‖², matching
                                          the prox interpretation)
    ΔU_Oja = η_o (x y^T − U y y^T), y = U^T x        (explicitly restores
                                          the subspace premise global LM
                                          training destroyed)

No weight transpose, no downstream error, no stored backward activation.
Hardware note: every operand is native — x_i integer/ternary events,
ξ_j ± 1, d_k one scalar per group, homeostasis a counter. This is an
on-chip-learning-shaped rule, priceable later by the same gate pipeline.

## 4. Parallelism inventory

- ±ε branches: one doubled batch; K probes: one more batch dimension.
- Heads/router groups: independent local losses, parallel.
- Blocks: update concurrently on detached clean activations, or pipeline
  across microbatches — forward traversal stays (block 6 needs block 5's
  activation); what disappears is backward locking.
- Timesteps: M3 counters via affine prefix scan (c_t = ρc_{t−1} + e is
  associative). **Exact only for the linear / residual-bits recurrence:
  the floor-based integer decay breaks affine associativity.** The
  retained-residual-bits variant already specified in M3.md is precisely
  the scan-compatible rational implementation — train with it in
  parallel, deploy either, measure the drift.

## 5. The four-arm ladder (architecture and local objective held fixed)

| arm | credit | isolates |
|---|---|---|
| Local-BP | exact within-block gradient | local-objective upper bound |
| Local-JVP | forward-mode directional derivative | estimator w/o hard-spike discontinuity |
| Hard-Perturb | antithetic node perturbation | genuinely forward-only hard-spike learning |
| Analytic-local | Oja + Hebbian dictionary + homeostasis | cheapest hardware rule |

This separates "local objectives fail" from "the forward estimator is
too noisy" — the confound the earlier M3-local sketch couldn't resolve.
Success metric per arm: gap to Local-BP at matched steps, plus the
standard kit (ΔR^c aligned, R(Z) envelope, Gram orthonormality — does
Oja hold the premise?, probe-suite retrieval).

## 6. The claim being assembled

*A scan-parallel, forward-only three-factor rule that optimizes explicit
local rate-reduction and predictive objectives through exact hard spike
events* — with the faithfulness price measured by λ_c, the subspace
premise actively maintained by Oja, and the rule's own hardware cost
counted by construction.

Sequencing unchanged: PLFP runs after M3-control validates the operator;
operator and learning rule never change in the same experiment.
