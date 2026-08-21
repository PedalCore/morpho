# The overcomplete dictionary fork (formalization before implementation)

Motivation, stated plainly: **published CRATE never beat attention
transformers, and this line inherits that ceiling.** The most conspicuous
capacity difference is the sparsification block — CRATE's d×d dictionary
against the transformer's 4d MLP. Sharpened by an observation from our
own wiring: the identity-control blocks contain **no elementwise
nonlinearity whatsoever** (the prox is identity; all nonlinearity is
LayerNorm plus the attention gate). The fork therefore tests two things
at once, deliberately: dictionary WIDTH (4d atoms) and an ACTIVE prox
(soft-threshold on the wide code). "Capacity" here = width × nonlinearity.

Screens rank early trajectories only (standing caveat); the eventual
winner gets a train-to-plateau run — where ppl bottoms out is the real
comparison, per protocol note below.

## The five interface decisions (each made explicitly)

**D1 — What is the layer state?** Chosen: a STATE PAIR. The d-stream z
feeds attention (Kp = d counter policy intact, all attention results
carry over); a wide code stream a ∈ R^{4d} carries the sparse code
across layers. Rejected: state = wide code everywhere (touches every
interface, quadruples the state-bits claim); state = reconstruction with
code reset per block (loses the unroll trick that won M2-control its
1.15 ppl).

**D2 — Unroll point.** Chosen: ISTA unrolls from the PREVIOUS LAYER'S
wide code (a₀ = a_{ℓ−1}; a₀ = 0 before block 0) — the faithful analog of
the winning reorder. Cost: a 4d activation stream at train/inference
time; zero parameters.

**D3 — What the prox quantizes (future spike arm).** The wide code a.
Coverage accounting REOPENS: D consumes codes; Dᵀ consumes the float
residual (or ternary, M2b-style); attention's U consumes the DENSE
decode z' = D a — so U-coverage is lost relative to M2's 100% unless a
second prox quantizes the decode (option recorded, deferred). No spike
claims carry over unexamined.

**D4 — Dictionary init and step.** D ∈ R^{d×4d}, column-normalized
random init (orthogonality is impossible overcomplete; Gram statistics
logged with the existing tooling). Step size η stays the learnable
scalar, init 0.1 (the load-bearing-scale lesson).

**D5 — The prox itself (this screen).** Soft-threshold ReLU on the wide
code — an ACTIVE nonlinearity, unlike the identity control. λ = 0.1 as
M0. (Spike prox substitutes here later, via the calibration protocol.)

## Block equations (the chosen design)

    x  = z + attn(z)                       CRSA on the d-stream, unchanged
    u  = LN(x)
    a' = relu( a + η·Dᵀ(u − D a) − ηλ )    one ISTA step, D ∈ d×4d,
                                           unrolled from the previous code
    z' = D a'                              decode; the d-stream continues

Parameters per block: U d² + D 4d² = 5d² (vs 2d² today).
**Matched-scale config: d = 448, L = 12, K = 8 (p = 56): 5d²·12 = 12.0M
+ tied embedding 1.8M ≈ 14.0M.** Counters Kp = 448 = d per policy.

## Screen protocol

3,000 steps, scratch, spikelm recipe, vs the width baseline's 19.62
(same read: early trajectory only). Outcome map: clearly better ⇒
extended run toward plateau (the "where does ppl bottom out" question is
the decisive one — larger/more-nonlinear models may descend slower and
bottom lower; screens cannot see that); comparable ⇒ plateau runs for
BOTH this and baseline before judging; worse ⇒ the ceiling is not (only)
the dictionary — training horizon, recurrent optimization, normalization,
tied-block capacity remain the live alternatives.
