# M12 — the HEU collaboration: replication, synthesis, and the
# front-end-mix experiment (2026-09-01)

External work: "Expressivity Limits of Single-Envelope Excitable
Dynamics for Event Commitment" (bakobiibizo, hi-sci-collab project
can-recurrent-temporal-structure-form-stable-representations).

## Replication (filed: github.com/PedalCore/heu-replication)
Blind reimplementation, paper only. PARTIAL REPRODUCTION:
mechanism confirmed (two-voice > single-voice; gap-region
separation 752x, their order; parameter-saturation fingerprint
2/4 vs 2/8 exactly as described); headline ratios not reproduced
(1.5x vs 7.7x/21x; ablation 1.0x vs 2.75x) pending five filed spec
ambiguities (units; voice-input assignment; window semantics; gap
floor mechanism; uncoupled bounds). Own bug caught en route: per-
second vs per-ms units reading (target diagnostics exposed it).

## Synthesis — where the HEU sits for us
- vs LIF: LIF resets (destroys level at every spike; cannot
  represent sustain); HEU keeps the envelope meaningful between
  spikes — envelope follower AND event detector in one unit, with
  compressor-vocabulary parameters. Complement, not competitor.
- THE LAW, fifth instance: one envelope cannot be post-decay AND
  pre-sustain = one LIF cannot drum (M10) = fast traces cannot
  hold a mode (v5) = one state cannot be mid-phrase (piano) = one
  pooled statistic cannot bind (DNA). Conflicting temporal
  configurations demand independent state variables; parameter
  search cannot substitute for state. Their N=1 vs N=2 result also
  retro-justifies our multiscale trace LADDERS.
- Event commitment costs TWO envelopes — a minimal-state threshold
  in our gate-count idiom; worth an actual LC costing (HEU vs LIF
  vs Schmitt+ladder) in the Morpho flow.
- Naming feedback delivered (measured discoverability cost: our
  own search failed under "Hodgkin-Huxley"): suggested Excitable
  Envelope Unit / Event Commitment Unit.

## Preregistered next experiment: THEIR FRONT END, OUR MEMORY
Pipeline: audio -> our coherence-gated filterbank envelopes ->
TWO-voice HEU commitment layer (two-voice from day one, per their
fragmentation warning) -> spike raster -> our clock-conditioned
GLM + consolidation slot memory -> stability metrics
(whitebox/stability_metrics.py).
Measures: (1) front-end fidelity: committed events vs ground-truth
MIDI onsets (GMD audio or synthesized), F1 at +/-20/40ms +
fragmentation rate (re-detections per true note; their headline
failure mode); (2) representation stability: do HEU-committed
events, fed to gradient-free consolidation, form stable
representations (their project's central question) — retrieval
margin, consolidation-weighted recall, disturbance recovery.
PREDICT: two-voice commitment beats single-voice on fragmentation
by a large factor (their 510x gap-region result should manifest as
sustain-fragmentation reduction); slot stability comparable to
clean-MIDI baseline if commitment F1@40ms > ~0.8.
Status: design only; build after their spec clarifications or
independently on synthesized audio.


## Measure 1, v1 results (2026-09-01; synthesized GMD audio,
## broadband envelope, 12 files, tol 40ms)

| commitment arm | F1 | prec | rec | fragmentation |
|---|---|---|---|---|
| single (defaults) | 0.292 | .32 | .28 | 0.58 extra/true |
| single (fast-recovery) | 0.290 | .34 | .26 | 0.51 |
| two-voice | 0.304 | .27 | .36 | 0.97 |

Two findings, both honest: (1) the BROADBAND front end is the
bottleneck (all arms weak on dense drum audio) — per-band flux is
required, consistent with the program's standing front-end law and
their per-key CQT design; commitment cannot rescue a poor
envelope. (2) The two-voice fragmentation PREDICTION FAILED at
this operating point: recall up, fragmentation UP (attack-biased
voice sustains through cymbal wash and re-crosses threshold).
Caveat recorded: their two-voice claim concerned envelope FITTING;
its interaction with threshold+refractory commitment is our
extension. v2: per-band envelopes -> per-band commitment, then
re-test the two-voice prediction where the front end is no longer
the binding constraint.


## Measure 1, v2 (per-band): prediction fails twice, diagnosis moves

| arm | F1 | rec | fragmentation |
|---|---|---|---|
| single broadband | 0.292 | .28 | 0.58 |
| two-voice broadband | 0.304 | .36 | 0.97 |
| single per-band (6) | 0.302 | .47 | 1.65 |
| two-voice per-band | 0.293 | .56 | 2.30 |

Per-band buys recall, explodes fragmentation (uniform 0.8
thresholds: cymbal wash re-triggers high bands; 10ms cross-band
merge insufficient for band group-delay spread). TWO-VOICE
FRAGMENTS MORE AT BOTH OPERATING POINTS — our extension prediction
is twice-failed. Fair scope note: their pipeline fed PITCH
LIKELIHOOD to per-key units + an FSM post-processor, never raw
flux to bare commitment; our failed extension measures the
distance between HEU-as-principle and HEU-as-deployed-system.
v3 (if pursued): per-band threshold calibration (CMA on train
split) + FSM-style post-processing — i.e., converging on what
mature onset detectors already do, which is itself a finding.


## Measure 2 (2026-09-01): COMPLETE — commitment adds
## representational value beyond detection accuracy

Single-channel bar-grid consolidation (16 slots, our WTA/TTG store)
+ shared stability metrics, 12 GMD files:

| event source | retrieval margin | consol recall@1 |
|---|---|---|
| ground-truth MIDI (dense) | +0.064 | 0.25 |
| truth @ HEU-matched density | +0.104 | 0.34 |
| HEU-committed (F1 0.29!) | +0.166 | 0.41 |

Density control separates the effect: sparsification helps
(saturated grids are inseparable), and HEU commitment beats
density-matched RANDOM sparsification — it selects accent-
structured events (energy-peak commitment; accents recur bar to
bar). POSITIVE finding for their thesis at the declared seam: an
imperfect excitable encoder yields MORE consolidatable temporal
structure than perfect detection. Caveats: n=12, 16-d single-
channel grids, one arm, modest absolute margins vs our 9-voice
memory (+0.252). M12 declared scope now fully tested: replication
(partial, filed), measure 1 (front-end/commitment, our extension
twice-failed), measure 2 (stability, positive with control).


## Hearables + legitimacy baseline + the first full loop (2026-09-01)

BASELINE (12 files, 40ms): flux+peak-pick F1 0.415 / frag 1.42 vs
HEU 0.292 / 0.58. VERDICT, precisely: HEU is NOT a better onset
detector than the standard method (never claim it); it is a
2.4x-lower-fragmentation, accent-selective COMMITTER — and measure
2 showed that is the property consolidation rewards. Different
virtues, both measured.
Renders (runs/m12): hear_committed.wav / hear_baseline.wav (92bpm
hiphop groove + clicks — the difference is audible), and
CALLRESPONSE.WAV: 8 bars real audio -> HEU commitment -> clocked
GLM -> 8 generated bars. First audio-in/response-out artifact of
the program. Known flaw: response denser than call (177 vs 103
events — single-channel free-run drift; the 9-voice + coupling
stack is the fix when this graduates from demo to instrument).
Not blocked on the author: clarifications gate only the
replication ratio re-run and v3 calibration.
