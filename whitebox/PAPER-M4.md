# Owner-Routed Slot Memory: Arbitrary Key–Value Binding at Constant State
### DRAFT v0.1 (2026-08-23) — all single-seed results marked; replication running

## Abstract

Constant-state sequence models — recurrent architectures whose memory
does not grow with context length — are attractive for streaming and
hardware deployment but have an unmeasured deficit: can they form
*arbitrary key–value bindings*, remembering which attribute belongs to
which entity? We answer this for a family of counter-based
statistics-attention models (CRSA, derived from the CRATE/ToST line).
First, we show the deficit is real and architectural: on a
preregistered binding-swap probe suite, every tied-projection mechanism
— counters, uniform statistics, windowed and full softmax attention
with shared Q=K=V — performs at stored-set chance, while the same
models hold perfect induction; a counterfactual suite on trained
14M-parameter language models finds no binding either, exposing a
primacy heuristic that masquerades as partial binding unless query
position is counterbalanced. Second, we localize the missing degree of
freedom by minimum untying at equal parameter count: giving attention a
separate *query* map yields near-perfect binding (0.44 → 1.00) where a
separate *value* map does not, and the query correction compresses to
rank d/8–d/4. Third, we introduce a constant-state repair:
M content-addressable slots with decaying statistics, written by
*owner routing* — each token is filed under the preceding token's key,
so associations co-locate. One seed: binding 0.94 with induction and
selective recall at 1.00, degradation governed by slot count rather
than elapsed time (capacity 0.615 → 0.97 at 16 stored facts when M
doubles from 8 to 16), at M(d+1) values per layer independent of
sequence length. We report the full ladder including instructive
failures — a shared representation basis loses a measured gradient
tug-of-war (cos = −0.46), and three mechanisms fail by cold-start
unless initialized at working structure — and the preregistered gates
that separate what is shown from what is not.

## 1. Introduction

A transformer's KV cache grows with every token; a counter-based
recurrent model carries a fixed set of statistics forever. The trade
is usually described in terms of perplexity, but perplexity averages
thousands of easy predictions and can hide a narrow failure. This
paper is about one such failure — the inability to remember *who owns
what* — and its repair without giving up constant state.

Contributions:
1. **A preregistered binding instrument** (matched swap pairs; axes
   separating time, load, interference, and locality; controls that
   caught two masquerades — a chance-level "ceiling" arm and a
   position heuristic imitating binding).
2. **Mechanistic localization by minimum untying at equal
   parameters**: the query map, not the value map, is the binding
   bottleneck in tied-projection attention; the correction is
   low-rank (rank 16–32 at d=128 retains 97–100%).
3. **Owner-routed slot memory**: a constant-state associative
   mechanism (one seed: 0.94 binding, 1.00 induction/selective,
   capacity ∝ slot count), plus the measured negative results that
   shaped it.
4. **A two-scale consistency result**: the probe-scale deficit
   predicts the LM-scale counterfactual behavior of 5–26M models,
   including the generation failure mode (referent drift).

## 2. Background

**CRSA.** Causal Rate-Statistics Attention replaces pairwise attention
with per-coordinate decaying activity counters c ← ρc + h², prices
output by 1/(1+c), and carries a fixed dyadic ladder of half-lives.
It descends from the Token Statistics Transformer (ToST,
arXiv:2412.17810): in prior work we showed multiscale exponential
forgetting is worth 0.38–0.42 nats/token over cumulative statistics
(including a numerically verified literal TSSA reproduction), while
softmax retains a 0.29 nats/token retrieval advantage. Counters hold
perfect induction at all tested delays; their measured weakness is
selective retention under distractors.

**The CRATE line and tied projections.** CRATE derives attention with
one shared projection per head (Q=K=V) from a compression objective.
Recent scaling work (CRATE-α, arXiv:2405.20299; CRATE-LM,
arXiv:2410.16443) substantially repairs its sparse-coding half —
independently converging with our overcomplete-dictionary fork — but
leaves tied attention untouched. Our results identify what that
omission costs.

## 3. The deficit, measured at two scales

**Probe scale.** The binding-swap suite (Sec. 4) puts every
tied-projection arm at stored-set chance across all cells — including
full-window MSSA (the intended ceiling), which forced a protocol
amendment: a true untied-QKV ceiling, which then swept the grid at
1.00. Re-reading our earlier locked suite confirms the associative
task was at stored-set chance for *every* arm all along.

**LM scale.** A counterfactual suite on seven trained checkpoints
(5.2M–26M; counters, softmax, dictionary variants) measures
candidate-restricted accuracy and swap sensitivity S on paired
prompts differing only in which color belongs to which name.
No model exceeds chance in any cell. The best model (8.08 ppl on
TinyStories) shows the largest |S| — but with sign flipping by query
position (+0.225 first-mentioned / −0.228 last-mentioned): a
*primacy heuristic*, not binding, detectable only because query
position was counterbalanced. This model writes fluent stories whose
characteristic failure is referent drift — the deficit rendered as
prose.

## 4. The instrument

Matched twin pairs: identical names, attributes, gap, and distractor
positions; only the assignment permutation differs, and the query
targets a swapped position so twins' answers must differ. Axes swept
independently: delay via quiet fillers (pure time), stored-fact count
(collision), distractor tokens at fixed total gap (interference),
query distance around the cache window (locality). Names and
attributes randomized per example. Preregistered gates: preserve
induction ≈ 1.00; recover ≥ half the selective-recall gap; binding
improvement growing with entity count; out-of-window benefit for
long-term claims; a declared state budget.

Two controls earned their keep: the tied "ceiling" at chance (Sec. 3)
and, in the offset-varied extension, an oracle-exploitation control
that measured grammar leakage (a fixed previous-token router extracts
0.62–0.75 on some cells by systematic cross-filing without ownership),
resetting the pass bar above measured exploitation rather than naive
chance.

## 5. Localizing the mechanism: minimum untying

At matched depth, width, data, and schedule, with equal parameter
count between arms:

| variant | binding |
|---|---|
| Q=K=V (tied) | chance |
| Q=K shared, V separate | 0.44, degrading with facts |
| **Q separate, K=V shared** | **1.00** |
| Q, K, V all separate | 1.00 (ceiling; +params) |

The bottleneck is *matching a retrieval context to a stored context*,
not representing returned content: "Tom has the red ball" and "Later,
Tom picks up his…" express one entity through different contexts, and
one projection cannot both encode mentions and formulate lookups.
SVD-truncating Δ = W_q − W_kv at evaluation: rank 16 retains 0.971,
rank 32 → 0.999 (d=128). The repair is a small query-side
preconditioner over one tied memory basis — at d=448, L=12:
172–344k parameters versus 9.6M for full QKV.

A windowed (16-token) untied-QKV arm fails to form the binding
circuit even under a reachability-constrained curriculum, while
full-history untied attention succeeds — the tested local operator
cannot learn the circuit, motivating a mechanism that is full-history
*by content* rather than by position.

## 6. Owner-routed slot memory

**Design.** Per layer: M slots with learned keys; write path
k = v = U_slot^T x (one basis, one cached vector per token); read path
q = W_q^T x (the proven bottleneck matrix). Writes route by softmax
over slot keys; slots keep decaying statistics V_j ← ρ_j V_j + a_j v
on a dyadic ladder; reads match q against slot keys. State M(d+1)
values per layer — constant in sequence length (~194 KB with counters
at d=448, L=12, versus a KV cache growing without bound).

**The defect that mattered (v1 → v2).** Routing each token by its own
content puts "red" in red's drawer, never Tom's: ownership is
structurally unrepresentable, and all v1 arms sat at chance with alive
routing (entropy 1.01/2.08, all slots occupied). Routing each write by
the *preceding token's* key — the slot form of attention's two-hop
composition — takes binding from chance to 0.94 in one change.

**Result (one seed, replication running).** Induction 1.00 (all
delays), selective recall 1.00 (all delays), binding 0.94: delay-flat
0.955–0.985 across 8–96 tokens; facts 1.00/0.95/0.91/0.615 at
2/4/8/16 — strong through M, cliff beyond; distractor-robust
0.925–0.995; identical in and out of any positional window. Doubling
M to 16 dissolves the cliff (0.97 at 16 facts); M=4 failed to form
the circuit at all (one seed, flagged, excluded from the capacity
claim). Failure moves from a time limit to a capacity/interference
limit.

**Basis ladder.** Slots sharing the counters' U fail (binding-loss
gradient on U opposes the locked-task gradient, cos = −0.455 at layer
0, with the locked gradient 2.6× larger); frozen shared basis also
fails binding but yields perfect selective recall — as does the own
basis. Slots require their own representation basis; whether
U_slot = U + low-rank Δ suffices is open.

## 7. Negative results and the cold-start pattern

Signed moments (restoring the sign h² discards) collapsed induction;
a zero-initialized gated branch never recruited (γ ≈ 0.05 after 20k
steps — at γ=0 the branch gradient is exactly zero); an open-gate
initialization (γ=0.1) injected interference without buying the
circuit; a learned owner selector (two micro-attentions choosing
write address and content) never bootstrapped, scoring below even the
hard-wired rule. Three independent mechanisms failed by cold start
and succeeded (slots v2) or are predicted to succeed (selector v2)
when initialized *at* working structure and allowed to deviate. We
offer this as a design rule for composing memory mechanisms:
function-preserving initialization is not enough; circuits must be
reachable from the initialization by gradient.

## 8. What is and is not shown

Shown (at probe scale, one seed unless noted): constant-state
induction + selective recall + arbitrary binding simultaneously;
capacity tracking slot count between M=8 and 16; query-map
sufficiency and low-rank structure (equal-param controlled); the
two-scale consistency of the deficit. Not shown: that language
training discovers write policies (the probe supplies
owner-adjacency; our learned selector has not yet passed); delay
flatness beyond 96 tokens; spike/hardware implementation of soft
routing (a calibrated hardening protocol is specified); LM-scale
quality of the integrated model (screen in progress); replication
(seeds running). The M=4 anomaly and the grammar-leakage measurement
are reported rather than smoothed.

## 9. Related work

ToST (token statistics attention; our operator's ancestor — we add
multiscale forgetting and now binding); CRATE / CRATE-α / CRATE-LM
(white-box line; feature-half repairs independently convergent with
ours; tied attention retained throughout, which our results identify
as the second deficit); slot/memory networks and fast-weight
programmers (related goals; our contribution is the measured
localization — query-role separation, owner routing, capacity ∝ M —
under preregistered controls at constant state); SAE-based
interpretability (post-hoc sparsity vs. our in-architecture route).

## 10. Conclusion

You do not have to choose between constant state and binding. A small
bank of owner-routed, content-addressed slots — sized by a measured
capacity law, read through a low-rank query correction, alongside
counters that own induction and a sparse dictionary that owns feature
capacity — provides all three measured memory capabilities at a few
hundred kilobytes of state, independent of sequence length. The open
problem is no longer the mechanism but the write policy: learning,
from language itself, who owns what.
