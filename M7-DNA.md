# M7 — genomics: do counters own a domain? (formalized 2026-08-26)

HYPOTHESIS: many genomic tasks ask "which motifs occurred, how often,
at what scales" — CRSA's decaying event statistics as SUFFICIENT
STATISTICS — not "which value belonged to this key." DNA may be a
better domain for pure counters than language ever was. Not all tasks:
variant effects / motif spacing / promoter-enhancer interactions are
retrieval-shaped (the measured CRSA weakness) — those are the
boundary controls, not the headline.

## Datasets (ordered)

1 GenomicBenchmarks (8 tasks, auto-download, 200-500 b, cheap) — entry.
  Counter-friendly: coding_vs_intergenomic, human_or_worm, enhancers
  (cohn/ensembl), ocr, regulatory. Position-sensitive: non-TATA
  promoters. Tiny (high variance): mouse enhancers (1,210 seqs).
2 Revised Nucleotide Transformer suite (chromosome-held-out; real
  negatives) — the substantive suite. Start: enhancers, H3K4me1/3,
  H3K27ac, promoters. NEGATIVE CONTROL: donor/acceptor splice sites
  (position-critical — pooled statistics SHOULD struggle).
3 HyenaDNA species classification — the headline: length sweep
  L in {1k, 4k, 16k, 64k, 256k}; accuracy rising with length at FIXED
  recurrent state = the clean CRSA validation TinyStories never was.
4 Caduceus 131k eQTL/VEP — boundary/negative control (variant identity
  + distant context = retrieval-shaped); later: counters vs mixed vs
  Longhorn vs Mamba separation prediction.

## DNA-specific architecture changes

- RC-EQUIVARIANCE: h(x) = 1/2 [F(x) + RC^-1 F(RC(x))], SHARED params;
  averaged logits = exact reverse-complement invariance (2x compute,
  1x params). Fair-comparison necessity vs Caduceus-class models.
- BIDIRECTIONality for classification (no causal constraint).
- LOCAL MOTIF STEM before counters: conv width ~7-15 b + event
  nonlinearity — counters cannot count motifs nothing detects. SAME
  stem across all arms (recurrence is the isolated variable).
- DNA HORIZONS: dyadic m in {4, 6, 8, 10} -> half-lives ~11/44/177/710
  bases for <=1kb tasks; for 131k later: m up to ~16. Shift-only
  implementation retained.

## Synthetic genomic controls (run before real data)

1 GC-content threshold | 2 motif-present | 3 motif-count threshold |
4 two motifs fixed spacing | 5 ordered motif pair | 6 mutated base
associated with distant motif. EXPECTED TRANSITION: counters solve
1-3; conv+counters maybe 4; 5-6 favor Longhorn/mixed. The transition
POINT is the finding.

## First experiment (collaborator decision: PURE COUNTERS first —
## a hybrid win would not attribute; counters give the clean answer)

Model: single-base tokens -> bidirectional motif conv -> RC-tied
BiCRSA counters (m 4/6/8/10) -> MLP -> mean+max pool -> classifier.
First real task: human_enhancers_cohn (28k seqs, 500 b, motif-density
shaped, not trivially compositional). coding_vs_intergenomic first as
SMOKE ONLY (should be easy; success there is not a result).

Minimal matched arms (same stem/MLP/width/depth/pool/schedule):
| arm | question |
|---|---|
| local CNN + pooling | does recurrence help at all? |
| RC-BiCRSA counters | the statistical-memory hypothesis |
| faithful no-Wv Longhorn | is associative state useful here? |
NO hybrid until these exist.

DECISION RULES: counters ~ Longhorn > CNN => counters are the right
sufficient statistic. counters ~ CNN < Longhorn => task needs
association. counters win enhancer/histone but LOSE splice sites =>
the ideal mechanistic result. complementary wins => 50/50 hybrid
earns its run.

STRONG-RESULT TARGET: *RC-equivariant multiscale counters match
Mamba-class models on composition/motif tasks at far lower state and
hardware cost; associative heads required only where position/variant
binding matters.* Even species-classification-only validates counters
as efficient sufficient statistics, not failed attention.

Substantive phase later: shared hg38 masked pretraining (Caduceus
pipeline), repeated seeds (small sets are noisy).


## First result — human_enhancers_cohn, counter arm (one seed)

**RC-BiCRSA counters: 74.05% test accuracy** (8 epochs, from scratch,
776k params, exact RC-invariance). Context (verified 2026-08-27):
CNN 69.5, GPT 70.5, DNABERT(110M) 74.0, HyenaDNA-tiny(<2M) 74.2,
ConvNova 74.3, Caduceus-Ph 74.7 — all models above 71 are
hg38-PRETRAINED; the strong modern ones are small. Honest claim:
counters match the pretrained cluster from scratch at comparable
size (1/150th of DNABERT only). [Correction: an earlier version of
this entry said "~1/100th size" of the band generally — wrong for
HyenaDNA/Caduceus, which are tiny.]
Trajectory: 70.4 / 71.8 / 71.7 / 73.1 / 73.7 / 72.9 / 73.7 / 74.05.
Longhorn arm running; CNN control last; splice-site negative control
and repeat seeds queued as the follow-ons that make the claim airtight.
