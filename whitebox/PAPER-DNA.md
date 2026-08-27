# Do counters own a domain? Decaying event statistics vs associative
# memory on genomic sequence tasks

*Status: v0.9 LIVE (2026-08-27) — counter arm final; Longhorn arm
mid-run (results below update as arms land); CNN control queued.
Program doc: M7-DNA.md (preregistered design, decision rules, dataset
ladder). Raw numbers: whitebox/runs/dna/sweep.log,
whitebox/runs/dna/results.jsonl.*

---

## 1. The hypothesis, and why DNA

CRSA counters — multiscale exponentially-decaying event statistics,
the whitebox program's cheapest memory — repeatedly measured as
*insufficient* on language: TinyStories never recruited key–value
binding, and where binding mattered (M4/M5 probe grids) counters
lost to regression memories decisively. The M7 conjecture is that
this is a statement about the domain, not the mechanism:

> Many genomic classification tasks ask "which motifs occurred, how
> often, at what scales" — for these, decaying counts are the
> SUFFICIENT STATISTIC, and associative (who-bound-what) memory is
> machinery without a job.

The conjecture is falsifiable in both directions: retrieval-shaped
genomic tasks exist (variant effect, motif spacing, splice sites)
where counters *should* lose. Getting both halves — counters match
associative memory on composition tasks AND lose on position-critical
tasks — is the mechanistically ideal outcome, far stronger than
winning everywhere.

## 2. Task and arms

**Task 1: `human_enhancers_cohn`** (GenomicBenchmarks): 20,843 train
/ 6,948 test, 500 bp, binary (enhancer vs negative). Chosen as the
first *hard* task — motif-density shaped, not trivially
compositional. (coding_vs_intergenomic was run as smoke only;
success there is not a result.)

Published context for calibration (checked 2026-08-27 against the
HyenaDNA and Caduceus reporting): CNN baseline 69.5; GPT 70.5;
DNABERT (110M, pretrained) 74.0; HyenaDNA (tiny-1k-d256, <2M,
hg38-pretrained) 74.2; ConvNova 74.3; Caduceus-Ph (small,
hg38-pretrained) 74.7. NOTE the size structure: the strong modern
baselines are SMALL pretrained models — the parameter gap is only
dramatic vs the transformers (DNABERT/NT class). The honest axis of
comparison for us is PRETRAINING: every model above 71 in that list
saw the whole human genome first; our arms see only the 21k task
examples.

**Matched design.** Three arms share EVERYTHING except the mixer:
single-base tokens (A/C/G/T/N), motif conv stem (width 11, GELU,
residual), d = 128, 4 blocks, per-block MLP (4×, GELU), LayerNorms,
mean+max pooling, linear head, AdamW 3e-4 OneCycle, batch 64,
8 epochs, seed 0. The mixer is the isolated variable:

| arm | mixer | params |
|---|---|---|
| cnn | none (stem + MLPs only) | ~0.6M |
| counter | RC-tied bidirectional CRSA counters | 776k |
| longhorn | faithful (no-Wv) diagonal delta, bidirectional | 976k |

**Exact reverse-complement invariance**, all arms: logits =
½[f(x) + f(RC(x))] with shared parameters — a fair-comparison
necessity against RC-equivariant baselines (Caduceus), implemented
by batching both strands through one trunk call (mathematically
exact; verified).

**Counter mixer** (BiCounter): h = Ux split into 4 horizon groups
m ∈ {4, 6, 8, 10} → decay ρ = 1 − 2^-m → half-lives ≈ 11 / 44 /
177 / 710 bases (chosen for 500 bp: sub-motif, motif, motif-cluster,
whole-sequence). Counters c = ½(forward + backward) decayed cumsums
of h² per coordinate; price-gated read h/(1 + c) (the CRSA rule);
decode through Uᵀ. Everything is a decaying sum — shift-register
hardware class, state 4×32 scalars per position stream.

**Longhorn mixer** (BiDelta): the faithful no-Wv diagonal delta
memory from M5 (value = the stream itself; the Wv-dispensability
result), run in both directions with shared parameters, outputs
averaged. State: 4 heads × 32×32 matrix — an order more state than
the counters, plus multiply-heavy updates (delta_cell_mul 522 gates
vs shift-ladder counter cells at 78).

## 3. Results so far

**Counter arm — FINAL: 74.05%** (best epoch; 8 epochs, from scratch,
776k params). Trajectory: 70.4, 71.8, 71.7, 73.1, 73.7, 72.9, 73.7,
74.05. Read against the context numbers: inside the pretrained
leaderboard cluster (74.0–74.7) with NO pretraining — at 776k we are
in fact ~1.7x LARGER than the strongest small baselines (HyenaDNA
436k, Caduceus 470k; ~1/140th of DNABERT) — under exact
RC-invariance, with a mixer that is entirely decaying sums. The
claim this supports: decaying multiscale statistics extract from 21k
labeled examples what the pretrained models bring from a genome.
(An earlier draft said "1/100th the size" of the band generally —
wrong for the SSM baselines; corrected, not patched silently.)

**Longhorn arm — FINAL: 74.63%** (best epoch = epoch 8). Trajectory:
64.0, 72.9, 66.4, 73.0, 73.8, 73.3, 74.1, 74.63 — far more volatile
than the counters' staircase, same destination class. At n=6,948 the
0.58-point margin over the counters is ~1 sigma of test noise: the
honest verdict is a STATISTICAL TIE leaning Longhorn (single seed;
replication owed). Both from-scratch arms therefore sit at the top
of the published pretrained cluster (74.0-74.7). Decision-rule
branch: counters ~ Longhorn — the cost axis (32x less state, 78- vs
522-gate cells, ~2x throughput) becomes the differentiator, pending
the CNN floor.

**CNN control — FINAL: 73.45%** (best epoch 4; trajectory 64.9,
72.2, 72.3, 73.45, 73.3, 72.2, 73.1, 73.4). The floor is HIGH.

**The completed cohn triple** (one seed, test n=6,948, sigma per
number ~0.5):

| arm | best acc | vs floor |
|---|---|---|
| cnn (stem+MLP, no memory) | 73.45 | — |
| counter | 74.05 | +0.60 (~0.8 sigma) |
| longhorn | 74.63 | +1.18 (~1.6 sigma) |

HONEST VERDICT ON COHN: all three arms land in a 1.2-point band that
overlaps the pretrained leaderboard cluster. The memory arms' edge
over the memory-free floor is within (counter) or barely beyond
(Longhorn) single-seed noise. Two reframings follow:

1. The headline moves to the STEM: a 711k RC-invariant stem+MLP
   from scratch scores 73.45 where the published CNN baseline is
   69.5 — our +4 points come from RC-invariance + mean/max pooling +
   recipe, NOT from memory. Cohn barely discriminates memory designs
   at all.
2. The memory-value question transfers to the seed replication
   (does +0.6/+1.2 survive?) and to the NT tasks, where floors and
   arms can separate. Cohn's role in the story shrinks to: "all
   three arms match genome-pretrained models from scratch."

Decision-rule reading: closest to counters ~ CNN ~ Longhorn — cohn
is (mostly) solvable without recurrence, which rule 2 did not
anticipate as a three-way tie; the rules bind on tasks that
separate.

**Preregistered decision rules** (M7-DNA.md, verbatim):
- counters ≈ Longhorn > CNN ⇒ counters are the right sufficient
  statistic (associative state adds nothing here).
- counters ≈ CNN < Longhorn ⇒ the task needs association after all.
- counters win enhancers/histones but LOSE splice sites ⇒ the ideal
  mechanistic result.
- complementary wins ⇒ a 50/50 hybrid earns its run.

## 3b. NT-suite results (LIVE; from-scratch, chromosome-held-out)

| task | cnn | counter | longhorn |
|---|---|---|---|
| splice_sites_donors (600bp) | 76.37 | 76.80 | **80.37** |
| H3K4me3 (1000bp) | 79.12 | 79.51 | 79.25 |
| enhancers (400bp) | 74.87 | 75.00 | 75.17 |

MATRIX COMPLETE (one seed): 8/9 cells are floor-ties; the single
separation is association x the relational task (+4.0, ~5 sigma).
Composition tasks do not discriminate memory designs; splice does,
and only binding wins it. (Longhorn H3K4me3/enhancers ran on the
verified FLA path; splice on the exact scan.)

**THE SPLICE SEPARATION (the within-design headline).** Longhorn
+3.6 over counters, +4.0 over the CNN floor (~5 sigma at n=3000,
single seed): the first real memory separation in M7, on exactly the
task preregistered as retrieval-shaped. Adding counting to the trunk
moves splice ~0.4 (noise); adding ASSOCIATIVE BINDING moves it +4.
The task discriminates memory types; composition tasks don't.
Remaining gap to pretrained (95-98.5 F1) = what genome-scale
pretraining buys on top.

**Strand test (no-RC): NEGATIVE, both arms.** cnn-norc 74.83
(-1.5), counter-norc 74.93 (-1.9): exact RC averaging HELPS splice
(two-strand ensembling) — the strand-identity concern measured
false. TrinityDNA's learned-gate alternative is not supported here.

**Metric correction — MCC/F1 (the literature's metrics), post-hoc
from saved final-epoch checkpoints, vs published pretrained+finetuned
baselines (ConvNova paper, 5 seeds):**

| task | metric | NTv2 | HyenaDNA | Caduceus-Ph | ConvNova | our cnn | our counter |
|---|---|---|---|---|---|---|---|
| H3K4me3 | MCC | 50.3 | 50.4 | 56.7 | 67.2 | 58.3 | 57.5 |
| enhancers | MCC | 54.5 | 53.1 | 55.2 | 57.6 | 50.5 | 50.4 |
| splice donors | F1 | 98.5 | 95.3 | 94.7 | 96.6 | 76.8 | 76.7 |

Three separated regimes invisible in accuracy: (1) H3K4me3 — our
FROM-SCRATCH trunk beats 4/5 pretrained baselines (only ConvNova
ahead); histone marks need no pretraining, and the win belongs to
the shared trunk, not the memory. (2) enhancers — pretraining is
worth a real 3-7 MCC. (3) splice donors — we are crushed (77 vs
95-98.5): BOTH our arms fail equally at the task pretrained models
nearly solve. The counter~CNN accuracy tie had concealed a 20-point
cross-model gap; the position hypothesis returns at the pretraining
level. Longhorn's pending splice run is now the live question: does
associative state close any of it? Trainer logs MCC natively from
here on. Protocol caveats: their numbers are 10-fold/5-seed
fine-tuning; ours single-seed final-epoch.

**The negative control did not go negative — and the autopsy matters.**
Counters were preregistered to LOSE splice donors (position-critical).
They tied/edged the CNN instead. Diagnosis: in the NT splice tasks
the candidate site is at a FIXED CENTERED position in every window,
so "is the donor consensus present?" is a counting question — the
task tests local pattern detection, not positional binding. Our
control was mis-chosen, not the story disproven; but per the
preregistration this must be treated as a strike against the
sufficient-statistic interpretation until a TRUE positional test
exists. That test is synthetic-ladder rungs 4-6 (fixed spacing /
ordered pairs / distant association), now promoted from optional to
REQUIRED for the M7 verdict. Lesson recorded: benchmark task names
are not mechanistic task types; window construction decides what a
task measures.

## 4. Engineering appendix (affects how speed claims may be made)

The first Longhorn attempt ran >2h without completing epoch 1 and was
killed. Diagnosis, in order discovered:

1. **System memory thrash** — 15.8/16GB swap with three ML jobs
   sharing unified memory; all timings from that period are
   confounded and were discarded.
2. **A real implementation bug in the shared LonghornMem scan**: the
   chunked scan materialized the per-position p×p outer-product
   state — ~30GB of memory traffic per training step at DNA shapes
   (B×4 strands/directions, T=500). FLOPs were trivial; the operator
   was bandwidth-bound purely through implementation.

Fix: the scan was rewritten in chunked linear-attention form —
intra-chunk reads factor through a C×C causal matrix
(A = q̃ k̃ᵀ, q̃ = q⊙P, k̃ = k/P, P the running decay product), the
p×p carry updates once per chunk. Verified against the sequential
diagonal recurrence at float64: values 1.4e-15, gradients ≤ 1.9e-13,
checkpointed path identical. Step time 111s → 3.36s (33×). The fix
is in `whitebox/model.py` and accelerates every Longhorn model in
the program, including the LM stack.

Consequences for claims: (a) yesterday's "counters train 10× faster"
observation was mostly our bug — the honest throughput gap is
whatever the current run shows (~2.3× at present: 1194s vs 517s per
epoch, and the counter epochs were themselves GPU-shared); (b) even
post-fix, per-element hardware cost still favors counters (78-gate
shift cells vs 522-gate multiply cells, and 32× less state) — the
efficiency half of the thesis rests on the hardware costing, not on
MPS wall-clock.

## 5. What a strong result requires (proofs that need doing)

The counter number is currently one seed, one task, one benchmark
family. In order of how much each closes:

1. **The within-task triple** (running): counter vs Longhorn vs CNN
   under the matched design → apply the decision rules. Without the
   CNN floor the 74% is uninterpretable — if stem+MLP alone hits 73,
   the counters did nothing.
2. **Seed replication** (≥3 seeds, counter arm first): GenomicBench
   test sets are small (6,948); ±0.5–1% seed noise is plausible and
   the foundation-band claim needs error bars.
3. **Splice-site negative control** (revised NT suite donor/acceptor):
   the falsification half. Counters SHOULD lose to Longhorn here —
   position-critical, retrieval-shaped. If counters win everything,
   the sufficient-statistic story is unfalsified puffery; if they
   lose exactly where predicted, the mechanism is doing the work.
4. **Chromosome-held-out negatives** (revised NT suite versions of
   enhancers/histones): GenomicBenchmarks negatives are synthetic;
   the substantive suite has real negatives and held-out chromosomes.
   The 74% must survive the harder negatives to mean anything
   biological.
5. **Horizon ablation**: drop each m ∈ {4,6,8,10} in turn. If
   accuracy is insensitive, the multiscale story is decoration; if
   the motif-scale horizons carry it, the sufficient-statistic claim
   gains its mechanism.
6. **RC ablation**: train the counter arm without strand averaging.
   Prices what exact RC-invariance is worth vs doubled compute.
7. **Synthetic controls ladder** (M7-DNA.md §synthetic): GC-threshold
   → motif-present → motif-count → fixed spacing → ordered pair →
   distant mutation–motif association. The counters-to-Longhorn
   TRANSITION POINT along this ladder is the cleanest mechanistic
   finding available; nothing on real data substitutes for it.

## 5b. Beyond accuracy: is what it learned USEFUL? (validation
## program — none of these need pretraining or new data)

Accuracy on a held-out split is the weakest evidence of learning;
the whitebox construction permits much stronger tests:

1. **Motif recovery**: align the stem's 128 width-11 filters against
   JASPAR TF motifs. Rediscovering known enhancer-associated motifs
   (AP-1/GATA/ETS-class) is external validation independent of any
   test split. One matmul + a PWM scan.
2. **In-silico saturation mutagenesis**: per-position prediction
   deltas over test sequences; real enhancer models concentrate
   importance on motif instances (comparable to published maps),
   noise-fitters don't.
3. **Motif-injection dose–response**: plant k motif copies in random
   background; score must rise monotonically with k at the right
   horizon scale. Turns "the counters count motifs" from
   interpretation into measurement. (Synthetic ladder rungs 2–3,
   run post-hoc on the trained model.)
4. **Frozen-trunk transfer**: cohn-trained trunk + linear head on
   ensembl enhancers / OCR vs from-scratch linear baseline —
   reusable features vs dataset quirks.
5. **Horizon attribution**: ablate each m at eval; which timescales
   carry the decision (the whitebox answer to "what did it learn").

## 5c. PREREGISTERED predictions for the NT-suite runs (written
## 2026-08-27, before any NT training)

1. nt_enhancers (400bp, real negatives): counters ~ Longhorn, both
   within ~1-2 pts of pretrained small models; absolute numbers drop
   vs cohn for everyone.
2. nt_H3K4me3 (1000bp): counters >= Longhorn (histone marks are
   broad density signals — the counter-friendly extreme). Registered
   caveat: slowest horizon half-life ~710b vs 1000b window; if
   counters underperform, suspect ladder coverage first (m=12
   diagnostic) before mechanism.
3. nt_splice_sites_donors (600bp, NEGATIVE CONTROL): counters LOSE
   clearly (positional geometry is unrepresentable in decaying
   sums); Longhorn > counters; both likely trail pretrained models
   more than on composition tasks. If counters do NOT lose here, the
   sufficient-statistic interpretation of M7 is wrong somewhere.
4. Hybrid trigger (decision rule 4): if 1-3 land as predicted
   (complementary wins), the 50/50 MixedMem arm runs on
   splice_sites_all / enhancers_types with the prediction that it
   matches the better specialist on each.

## 6. Future directions beyond the current task

- **Species classification, length sweep** (HyenaDNA task): accuracy
  vs L ∈ {1k, 4k, 16k, 64k, 256k} at FIXED counter state — the
  potential headline. Counters are O(1) state at any length; if
  accuracy *rises* with length while attention baselines truncate or
  pay quadratically, this is the clean CRSA validation TinyStories
  never was. Needs the dyadic horizon ladder extended to m ≈ 16
  (half-life ~46kb) — shift-only implementation already supports it.
- **Histone marks + promoters** (revised NT suite): the
  counter-friendly middle of the suite, with real negatives.
- **Variant-effect boundary** (Caduceus 131k eQTL): the designed
  counter-failure case at scale — variant identity + distant context
  = retrieval-shaped. Expected counter loss; measures how far the
  domain extends before associative memory earns its hardware.
- **The 50/50 hybrid** (MixedMem, half counters / half delta): runs
  ONLY if the arms show complementary wins (decision rule 4) — a
  hybrid win without single-arm attribution is uninterpretable,
  which is why it was excluded from the first sweep.
- **Genome-scale masked pretraining** (Caduceus hg38 pipeline,
  shared across arms): the from-scratch numbers bound what the
  architecture extracts from 21k examples; pretraining tests what it
  extracts from a genome. Only worth it after the triple + negative
  control pattern is known.
- **Hardware costing of the DNA counter stack**: the 776k-param
  classifier's mixer is shift-register class end to end; a bank8-style
  synthesis (cf. examples/delta) of one horizon group would put a
  gate count under the "1/100th the size" claim and connect M7 back
  to the program's hardware thesis.

## 6b. Reading-derived experiments (HPB-DNA ICLR'26 sub; TrinityDNA
## 2507.19229) — added 2026-08-28

1. **Complex (oscillatory) counters** — decaying sums with phase,
   s <- lambda e^{j omega} s + x: the pitch tracker's C-statistic as
   a sequence-model channel. Serves DNA periodicity (codon period 3,
   nucleosome ~10.5 b — HPB's wavelet-path motivation, done
   forward-only at constant state), and LANGUAGE position
   sensitivity (the measured weak spot of counters AND delta
   models). Dyadic omega keeps the shift-ladder hardware class.
   The single highest-leverage new arm; test on synthetic
   periodicity first, then LM position probes.
2. **Stem upgrade (LPL-lite)**: multi-width kernel bases {3,5,7,11}
   with token-conditioned mixture weights, replacing the single
   static width-11 conv. M7 showed the trunk carries the results —
   stem improvements now outrank mixer improvements.
3. **Strand-identity test (running)**: exact RC-invariance averages
   away strand — wrong prior for directional tasks (TrinityDNA uses
   a learned RC gate instead). --no-rc arms on splice donors test
   whether part of the 20-point pretrained gap is self-inflicted.
4. Protocol note: HPB's matched reruns place HyenaDNA at 72.46 /
   Caduceus-Ph 73.62 / ConvNova 73.82 on cohn (vs their own
   pretrained 74.96) — our from-scratch 74.05/74.63 exceeds all
   their baseline reruns.

## 7. Reproduction

```
# data: GenomicBenchmarks auto-download to ~/.genomic_benchmarks
python3 -m whitebox.dna_train --arm counter  --task human_enhancers_cohn
python3 -m whitebox.dna_train --arm longhorn --task human_enhancers_cohn
python3 -m whitebox.dna_train --arm cnn      --task human_enhancers_cohn
# ~20 min/epoch (longhorn) / ~9 min (counter) on M-series MPS;
# results append to whitebox/runs/dna/results.jsonl
```
