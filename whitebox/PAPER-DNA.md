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

**Longhorn arm — running.** Epoch 1: 64.0% (vs the counters' 70.4%
opening). The counters gained 3.7 points over their remaining
schedule; Longhorn needs a qualitatively steeper curve to reach
74. Updates land here as epochs print.

**CNN control — queued** (runs immediately after Longhorn). This arm
prices the stem+MLP floor: how much of 74% is motif detection alone,
no recurrence at all?

**Preregistered decision rules** (M7-DNA.md, verbatim):
- counters ≈ Longhorn > CNN ⇒ counters are the right sufficient
  statistic (associative state adds nothing here).
- counters ≈ CNN < Longhorn ⇒ the task needs association after all.
- counters win enhancers/histones but LOSE splice sites ⇒ the ideal
  mechanistic result.
- complementary wins ⇒ a 50/50 hybrid earns its run.

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

## 7. Reproduction

```
# data: GenomicBenchmarks auto-download to ~/.genomic_benchmarks
python3 -m whitebox.dna_train --arm counter  --task human_enhancers_cohn
python3 -m whitebox.dna_train --arm longhorn --task human_enhancers_cohn
python3 -m whitebox.dna_train --arm cnn      --task human_enhancers_cohn
# ~20 min/epoch (longhorn) / ~9 min (counter) on M-series MPS;
# results append to whitebox/runs/dna/results.jsonl
```
