# Pitch tracking as online complex regression — the program's
# operators, one channel wide

*Status: v1 (2026-08-27). Prototype + benchmark in
whitebox/pitch_track.py (CPU, numpy). Derivation and design due to
the collaborator; this page records the build, the measured results,
and the honest failures. Origin: a Faust Hilbert/unwrap tracker
(after Puckette's "Patch for Guitar") that intermittently parks at
its frequency clamps.*

## 1. The estimator

Predict the next analytic sample instead of differentiating phase:
z_t ~ a z_{t-1} under decayed least squares. Sufficient statistics
are decaying complex counters:

    C_t = l C_{t-1} + z_t conj(z_{t-1})
    P_t = l P_{t-1} + |z_{t-1}|^2,   Q_t = l Q_{t-1} + |z_t|^2
    f = Fs/(2pi) arg(C),   coherence = |C|/sqrt(PQ)

One estimator, three of the program's objects at once: a decaying
counter (CRSA), a scalar complex Longhorn (online ridge regression,
forward-only, constant state), and an explicit prediction objective
whose residual is a confidence signal. Amplitude weighting is
intrinsic (|z z*| ~ A^2): the amplitude-null phase jumps that break
unwrap trackers contribute ~nothing by construction.

## 2. v1 failure, recorded (the front-end lesson again)

A bare single-channel version (no filterbank, dyadic coherence
fusion) was measured first:

- perfect on clean sines, AM-nulls, vibrato (0.0-0.4c median);
- 983c high on an 8-partial tone; octave-plus errors at snr <= 10dB.

Autopsy: without band-limiting, arg(C) measures the SPECTRAL
CENTROID of the whole analytic signal — a sum of phasors at
different partial frequencies averages above f0. Same shape as the
M7 stem lesson: the memory stage cannot rescue a missing front end.
The collaborator's spec had the filterbank first; v1 skipped it and
paid exactly the predicted price.

## 3. v2 — filterbank -> per-band counters -> harmonic scoring

32 log-spaced 2nd-order bandpass bands (50-2400 Hz) -> per-band
C/P/Q at 16 ms half-life -> per-hop (1 ms) harmonic template score
over an 8-cent f0 grid (8 harmonics, 1/k weights, 35c Gaussian
tolerance, bands weighted by coherence*sqrt(energy)) -> refine f0 as
the weight-averaged f_band/k of contributing bands. The scoring
stage is the "sparse harmonic dictionary" of the polyphonic design,
untrained. VOICING: harmonic-score contrast (peak/mean over the f0
grid) — noise scores flat; within-band coherence CANNOT work
(narrowband-filtered noise is locally coherent; measured 99.9%
false-voiced before the fix, 1.4% after).

Measured (48kHz, |cents| median / 95% / octave-error rate):

| case | med | 95% | oct% |
|---|---|---|---|
| sine 55 / 220 / 880 Hz clean | 0.0-0.2 | <2 | 0 |
| sine 220 Hz, SNR 20 / 10 / 0 dB | 0.2 / 0.5 / 1.7 | <5 | 0 |
| 330 Hz AM through zero (4 Hz) | 0.4 | 1.2 | 0 |
| 165 Hz, 8 partials (1/k) | 0.8 | 1.4 | 0 |
| white noise | voiced 1.4% (correctly rejected) | | |
| 440+659 Hz two-tone | 697 | 697 | 100 |

Reference: the unwrap baseline (the Faust structure) at 0 dB SNR is
~3600c (parked at a clamp), 112c spikes at AM-nulls, and 99.8%
boundary-dwell under noise. The two-tone row is the preregistered
polyphony failure: a single estimator locks between notes — that is
the slot stage's job, not a bug in this one.

## 4. Why the original Faust tracker gets stuck (answered)

Measured and mechanistic, matching the collaborator's diagnosis:
(1) amplitude nulls make phase arbitrary — one bad sample becomes a
huge frequency innovation; (2) the min/max clamp sits INSIDE the
feedback loop, so a railed measurement feeds the smoother and stays
railed (unwrap baseline: 99.8% boundary dwell under noise);
(3) the estimated pitch retunes the prefilter SVFs whose phase
response then shifts, and the tracker reads its own filter motion as
input motion — a positive-feedback lock. Minimal fixes for the
Faust patch, in order of value: replace phase-diff/unwrap with
z_t z*_{t-1} accumulation; clamp only the published output; gate
updates on coherence (freeze, don't integrate noise); slow the
prefilter cutoff far below the pitch rate or use a small fixed
filterbank so the observation path cannot move.

## 5. Roadmap

1. Polyphony: M slots over the harmonic-score surface (peak pick +
   inhibition, per-slot C/P/Q state) — note identity through
   crossings; the M4 binding machinery's cleanest audio use.
2. Streaming port: causal FIR quadrature pair (the offline
   scipy.hilbert here is anticausal); latency budget ~5-10 ms.
3. Hardware: everything is leaky accumulation + one atan2 per hop +
   a small score table — shift-ladder decays apply (dyadic l);
   candidate for the Morpho pipeline next to the wkv cell.
4. Dyadic multi-horizon per band (fast attack heads / slow noise
   rejection) with score-level fusion — currently one 16 ms horizon.
5. Faust back-port of the minimal fixes (§4) for the original patch.
6. Real-signal evaluation: speech (voiced/unvoiced accuracy vs
   laryngograph ground truth, e.g. Keele/PTDB), guitar with pick
   attacks, vibrato cello; cents-error CDFs vs pYIN/CREPE baselines.
