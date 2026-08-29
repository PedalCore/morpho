# M10 — spikes ARE the rhythm: minimal machinery for musical time
# (design, preregistered 2026-08-29; user concept)

CONCEPT (user): drop tokenization — represent rhythm as spike trains
where the representation maps ONE-TO-ONE to musical output. A spike
at bin t IS the event at time t. Normalize pitch to one note,
velocity to on/off; find the MINIMAL machinery that replicates or
learns rhythmic structure. Read heads = aggregations over the spike
raster.

KEY UNIFICATION: beat tracking is pitch tracking at 0.5-8 Hz. The
M8/pitch complex-counter estimator (C = decayed z z*, phase ->
frequency, coherence -> confidence) applied to the onset train at
tempo frequencies IS a tempo/phase/meter tracker, near
parameter-free. The M8 oscillator bank is not analogous to meter —
it is meter.

## Data
ARIA unique subset -> onset trains: all notes -> one channel,
velocity stripped, 20 ms bins (fine grid PRESERVES swing/expressive
timing — do not quantize to tatum; expressiveness is signal).
Train/val split by performance.

## The ladder (params ~, preregistered predictions)
L0 Bernoulli / bin-conditional Markov floor (1-10): the nothing.
L1 single LIF unit, learned decay+threshold, self-history drive
   (~3): PREDICT locks to isochronous pulse only.
L2 oscillator bank: complex counters at log-spaced tempo freqs,
   spike prob from phase, learned amplitudes/couplings (50-200):
   PREDICT tempo-following + meter (strong/weak) appear here.
L3 L2 + CRSA spike-count ladder over the raster (the "read head
   over stacks") (1-5k): PREDICT bar-level pattern repetition
   appears here (needs WHICH beats fired, not just phase).
L4 tiny full model (~50k): ceiling reference.

## Eval
Next-bin NLL + onset F1 (tolerance +/-1 bin) vs ladder; click-track
continuation renders (listenable); per-rung capability probes:
isochronous / meter / swing / bar-repetition synthetic rhythms
before ARIA (the M7 lesson: controls first, and mind the leaks —
e.g. a rate-only model fakes F1 on dense passages; report per-IOI
stratified metrics).

## Why it matters
1. "Gate count for groove": N params for pulse, M for meter, K for
   pattern — capability thresholds in the found-machines spirit.
2. Every rung is event-driven, dyadic-decay-friendly -> direct
   Morpho/iCE40 path; a hardware rhythm continuator is the program
   thesis made audible.
3. Contrast with M9 token models: do 22M-param token LMs learn
   anything about TIME that a 200-param oscillator bank doesn't?
   The answer either way is a headline.

Status: design only. Build: data prep + L0-L2 first (Mac-scale).
