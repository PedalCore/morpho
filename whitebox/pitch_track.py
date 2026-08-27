"""Hilbert predictive pitch tracker — complex counters vs phase-unwrap.

The estimator (collaborator derivation, PITCH.md): predict the next
analytic sample z_t ~ a z_{t-1} under decayed least squares. The
sufficient statistics are decaying complex counters:

    C_t = l C_{t-1} + z_t conj(z_{t-1})     (cross)
    P_t = l P_{t-1} + |z_{t-1}|^2           (energy, lagged)
    Q_t = l Q_{t-1} + |z_t|^2               (energy)
    f_t = Fs/(2 pi) arg(C_t)                (frequency: phase of C)
    coh = |C_t| / sqrt(P_t Q_t)             (phase coherence in [0,1])

This is a scalar complex Longhorn (online regression, forward-only,
constant state); the dyadic multi-horizon bank is CRSA. Amplitude
weighting is intrinsic: |z z*| ~ A^2, so nulls contribute ~nothing —
the mechanism that breaks unwrap-based trackers (arbitrary phase
jumps at nulls) is squelched by construction.

Baseline for comparison: classic instantaneous frequency
(diff of unwrapped analytic phase) + one-pole smoothing + output
clamp — the structure of the Faust tracker under test.

python3 -m whitebox.pitch_track      # runs the benchmark table
"""

import numpy as np
from scipy.signal import hilbert, lfilter

SR = 48000
FMIN, FMAX = 55.0, 1760.0
HALF_LIVES_MS = (2.0, 8.0, 32.0, 128.0)     # dyadic-ish CRSA horizons
COH_GATE = 0.6


def _decay(hl_ms):
    return 0.5 ** (1.0 / (hl_ms * SR / 1000.0))


def _leaky(lam, u):
    return lfilter([1.0], [1.0, -lam], u)


def track_counters(z):
    """Multi-horizon complex-counter tracker.
    Returns (freq_hz, coherence, amplitude)."""
    u = np.empty_like(z)
    u[0] = 0
    u[1:] = z[1:] * np.conj(z[:-1])          # per-sample phase increment
    e = np.abs(z) ** 2
    el = np.empty_like(e)
    el[0] = 0
    el[1:] = e[:-1]

    vec = np.zeros(len(z), dtype=complex)    # coherence-weighted fusion
    best = np.zeros(len(z))
    for hl in HALF_LIVES_MS:
        lam = _decay(hl)
        C = _leaky(lam, u)
        P = _leaky(lam, el)
        Q = _leaky(lam, e)
        coh = np.abs(C) / (np.sqrt(P * Q) + 1e-12)
        mag = np.abs(C) + 1e-30
        vec += (C / mag) * coh               # unit phasor, coherence weight
        best = np.maximum(best, coh)
    f = np.angle(vec) * SR / (2 * np.pi)

    # confidence gate: freeze (forward-fill) where coherence is low
    ok = (best >= COH_GATE) & (f > 0)
    idx = np.where(ok, np.arange(len(f)), 0)
    np.maximum.accumulate(idx, out=idx)
    f = f[idx]
    return np.clip(f, FMIN, FMAX), best, np.abs(z)


def track_unwrap(z, tau_ms=10.0):
    """Baseline: unwrapped-phase difference + one-pole + output clamp."""
    ph = np.unwrap(np.angle(z))
    f = np.empty(len(z))
    f[0] = 0
    f[1:] = np.diff(ph) * SR / (2 * np.pi)
    lam = _decay(tau_ms)
    f = lfilter([1 - lam], [1.0, -lam], f)
    return np.clip(f, FMIN, FMAX)


def cents(f, f0):
    return 1200.0 * np.log2(np.maximum(f, 1e-6) / f0)


def _steady(sig, f0, skip_ms=200):
    s = int(skip_ms * SR / 1000)
    z = hilbert(sig)
    fc, coh, _ = track_counters(z)
    fu = track_unwrap(z)
    return cents(fc[s:], f0), cents(fu[s:], f0)


def bench():
    rng = np.random.default_rng(0)
    rows = []

    def add(name, cc, cu):
        rows.append((name,
                     np.median(np.abs(cc)), np.percentile(np.abs(cc), 95),
                     np.mean(np.abs(cc) > 600) * 100,
                     np.median(np.abs(cu)), np.percentile(np.abs(cu), 95),
                     np.mean(np.abs(cu) > 600) * 100))

    t = np.arange(int(1.0 * SR)) / SR
    for f0 in (55.0, 220.0, 880.0, 1760.0):
        sig = np.sin(2 * np.pi * f0 * t)
        add(f'sine {f0:.0f}Hz clean', *_steady(sig, f0))
    for snr in (20.0, 10.0, 0.0):
        f0 = 220.0
        sig = np.sin(2 * np.pi * f0 * t)
        n = rng.standard_normal(len(t)) * 10 ** (-snr / 20) / np.sqrt(2)
        add(f'sine 220Hz snr{snr:.0f}dB', *_steady(sig + n, f0))

    # tremolo through zero (amplitude nulls — the unwrap killer)
    f0 = 330.0
    am = np.sin(2 * np.pi * 4.0 * t)         # nulls at 8/s
    sig = am * np.sin(2 * np.pi * f0 * t)
    add('330Hz AM-null 4Hz', *_steady(sig, f0))

    # vibrato tracking: 6 Hz, +/-50 cents
    f0 = 440.0
    fi = f0 * 2 ** (50 / 1200 * np.sin(2 * np.pi * 6 * t))
    sig = np.sin(2 * np.pi * np.cumsum(fi) / SR)
    s = int(0.2 * SR)
    z = hilbert(sig)
    fc, _, _ = track_counters(z)
    fu = track_unwrap(z)
    add('440Hz vib 6Hz +/-50c',
        cents(fc[s:], fi[s:]), cents(fu[s:], fi[s:]))

    # harmonic tone (sawtooth-ish, 8 partials 1/k)
    f0 = 165.0
    sig = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 9))
    add('165Hz 8-partial', *_steady(sig, f0))

    # two equal tones a fifth apart (polyphonic stress; "truth" = lower)
    sig = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 659.25 * t))
    add('440+659 two-tone', *_steady(sig, 440.0))

    print(f'{"case":24s} | counter med/95/oct% | unwrap med/95/oct%')
    for r in rows:
        print(f'{r[0]:24s} | {r[1]:7.1f} {r[2]:8.1f} {r[3]:5.1f} | '
              f'{r[4]:7.1f} {r[5]:8.1f} {r[6]:5.1f}')

    # step reacquisition: 500 -> 800 Hz at t=1s
    t2 = np.arange(int(2.0 * SR)) / SR
    fi = np.where(t2 < 1.0, 500.0, 800.0)
    sig = np.sin(2 * np.pi * np.cumsum(fi) / SR)
    z = hilbert(sig)
    fc, _, _ = track_counters(z)
    fu = track_unwrap(z)
    for name, f in (('counter', fc), ('unwrap', fu)):
        after = f[SR:]
        w = np.where(np.abs(cents(after, 800.0)) < 50)[0]
        ms = w[0] / SR * 1000 if len(w) else float('inf')
        print(f'reacquire 500->800Hz {name}: {ms:.1f} ms')

    # boundary dwell under pure noise (should NOT park at 55/1760)
    z = hilbert(rng.standard_normal(int(2.0 * SR)) * 0.1)
    fc, coh, _ = track_counters(z)
    fu = track_unwrap(z)
    for name, f in (('counter', fc), ('unwrap', fu)):
        dwell = np.mean((f <= FMIN * 1.02) | (f >= FMAX * 0.98)) * 100
        print(f'noise boundary-dwell {name}: {dwell:.1f}%')
    print(f'noise mean coherence (counter): {coh.mean():.3f} '
          f'(gate {COH_GATE} -> output frozen, flagged unvoiced)')


if __name__ == '__main__':
    bench()


# ---------------------------------------------------------------------------
# v2 — fixed analytic filterbank -> per-band complex counters -> harmonic
# scoring (untrained "sparse harmonic dictionary" stage) -> f0, voicing.
# ---------------------------------------------------------------------------
from scipy.signal import butter, sosfilt

N_BANDS = 32
BAND_LO, BAND_HI = 50.0, 2400.0
HOP = 48                                     # 1 ms decision rate
F0_GRID = 55.0 * 2 ** (np.arange(0, 4801, 8) / 1200.0)   # 55-880Hz, 8c
N_HARM = 8
VOICED_COH = 0.75
VOICED_CONTRAST = 4.0   # harmonic-score peak/mean; noise is flat


def _bank():
    edges = np.geomspace(BAND_LO, BAND_HI, N_BANDS + 1)
    return [(np.sqrt(lo * hi), butter(2, [lo, hi], 'bandpass',
                                      fs=SR, output='sos'))
            for lo, hi in zip(edges[:-1], edges[1:])]


_BANK = _bank()


def track_bank(x, hl_ms=16.0):
    """Returns per-hop (f0_hz, voiced, coherence) via harmonic scoring."""
    lam = _decay(hl_ms)
    fb, cohb, engb = [], [], []
    for fc, sos in _BANK:
        z = hilbert(sosfilt(sos, x))
        u = np.empty_like(z); u[0] = 0
        u[1:] = z[1:] * np.conj(z[:-1])
        e = np.abs(z) ** 2
        el = np.empty_like(e); el[0] = 0; el[1:] = e[:-1]
        C = _leaky(lam, u); P = _leaky(lam, el); Q = _leaky(lam, e)
        fb.append(np.angle(C) * SR / (2 * np.pi))
        cohb.append(np.abs(C) / (np.sqrt(P * Q) + 1e-12))
        engb.append(Q)
    fb = np.array(fb)[:, ::HOP]              # (B, T/HOP)
    cohb = np.array(cohb)[:, ::HOP]
    engb = np.array(engb)[:, ::HOP]
    w = cohb * np.sqrt(engb)                 # band weight
    T = fb.shape[1]
    f0s = np.zeros(T); voi = np.zeros(T, bool); conf = np.zeros(T)
    logg = np.log2(F0_GRID)
    for t in range(T):
        fbt, wt = fb[:, t], w[:, t]
        good = (wt > 1e-6) & (fbt > BAND_LO * 0.5)
        if good.sum() < 1:
            continue
        score = np.zeros(len(F0_GRID))
        for k in range(1, N_HARM + 1):
            d = 1200 * np.abs(np.log2(np.maximum(fbt[good, None], 1e-3)
                                      / (k * F0_GRID[None, :])))
            score += (wt[good, None] / k * np.exp(-(d / 35.0) ** 2)).sum(0)
        i = int(np.argmax(score))
        contrast = score[i] / (score.mean() + 1e-12)
        # refine: weighted mean of contributing band freqs / k
        num = den = 0.0
        for k in range(1, N_HARM + 1):
            d = 1200 * np.abs(np.log2(np.maximum(fbt[good], 1e-3)
                                      / (k * F0_GRID[i])))
            m = d < 50
            num += (wt[good][m] / k * fbt[good][m] / k).sum()
            den += (wt[good][m] / k).sum()
        f0s[t] = num / den if den > 0 else F0_GRID[i]
        conf[t] = contrast
        voi[t] = contrast > VOICED_CONTRAST
    # hold-last-voiced
    idx = np.where(voi, np.arange(T), 0)
    np.maximum.accumulate(idx, out=idx)
    return f0s[idx], voi, conf


def bench2():
    rng = np.random.default_rng(0)
    t = np.arange(int(1.0 * SR)) / SR
    skip = int(0.3 * SR / HOP)

    def run(sig, f0, name):
        f, voi, _ = track_bank(sig)
        c = cents(f[skip:], f0)
        print(f'{name:24s} | {np.median(np.abs(c)):7.1f} '
              f'{np.percentile(np.abs(c), 95):8.1f} '
              f'{np.mean(np.abs(c) > 600) * 100:5.1f} | voiced '
              f'{voi[skip:].mean() * 100:5.1f}%')

    print(f'{"case (v2 bank)":24s} | med/95/oct-err% | voiced')
    for f0 in (55.0, 220.0, 880.0):
        run(np.sin(2 * np.pi * f0 * t), f0, f'sine {f0:.0f}Hz clean')
    for snr in (20.0, 10.0, 0.0):
        f0 = 220.0
        n = rng.standard_normal(len(t)) * 10 ** (-snr / 20) / np.sqrt(2)
        run(np.sin(2 * np.pi * f0 * t) + n, f0, f'sine 220 snr{snr:.0f}dB')
    am = np.sin(2 * np.pi * 4.0 * t)
    run(am * np.sin(2 * np.pi * 330 * t), 330.0, '330Hz AM-null 4Hz')
    sig = sum(np.sin(2 * np.pi * k * 165 * t) / k for k in range(1, 9))
    run(sig, 165.0, '165Hz 8-partial')
    sig = np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 659.25 * t)
    run(sig, 440.0, '440+659 two-tone')
    # noise: voiced flag should be ~0
    f, voi, conf = track_bank(rng.standard_normal(len(t)) * 0.1)
    print(f'{"noise voiced-rate":24s} | {voi.mean() * 100:.1f}% '
          f'(mean coh {conf.mean():.2f})')


if __name__ == '__main__':
    pass
