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
