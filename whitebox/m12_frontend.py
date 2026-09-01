"""M12 measure 1 — HEU commitment on synthesized GMD audio.

GMD MIDI -> fluidsynth audio -> onset-band energy envelope (1ms) ->
single-voice vs TWO-voice HEU commitment -> spike events vs
ground-truth MIDI onsets: F1 (+/-20/40ms) + FRAGMENTATION rate
(extra detections per true onset — their headline failure mode).
HEU params: paper defaults (p=0) with light CMA fit of p_total &
p_balance on 5 train files; two-voice = recovery-biased + attack-
biased pair as in their target construction.

python3 -m whitebox.m12_frontend
"""

import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path.home() / 'coding' / 'heu-replication'))

from heu import coupled_params, simulate                    # noqa
from whitebox.rhythm3_exact import GMD                       # noqa

SF = ('/private/tmp/claude-501/-Users-marycarrigan-coding-morpho-0-2/'
      '54729931-579f-4181-b299-4b6eb253be2c/scratchpad/'
      'MuseScore_General.sf3')
OUT = pathlib.Path('whitebox/runs/m12')
SR = 22050


def midi_onsets(path):
    import mido
    mid = mido.MidiFile(path)
    t, ons = 0.0, []
    for m in mid:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0:
            ons.append(t)
    return np.array(ons)


def audio_envelope(path):
    """Render midi -> audio -> 1ms onset-energy envelope."""
    wav = OUT / (path.stem + '.wav')
    if not wav.exists():
        subprocess.run(['fluidsynth', '-ni', '-F', str(wav),
                        '-r', str(SR), SF, str(path)],
                       capture_output=True)
    import wave
    w = wave.open(str(wav))
    x = np.frombuffer(w.readframes(w.getnframes()),
                      np.int16).astype(np.float64)
    if w.getnchannels() == 2:
        x = x.reshape(-1, 2).mean(1)
    x /= (np.abs(x).max() + 1e-9)
    hop = SR // 1000                          # 1 ms
    n = len(x) // hop
    e = np.sqrt(np.convolve(x * x, np.ones(4 * hop) / (4 * hop),
                            'same'))[::hop][:n]
    flux = np.maximum(0, np.diff(e, prepend=e[0]))   # onset energy
    return flux / (np.quantile(flux, 0.99) + 1e-9)


def commit(drive, pvec_list):
    """HEU commitment: spikes at threshold crossings w/ refractory.
    Multi-voice: max of envelopes."""
    envs = [simulate(drive, *coupled_params(p))[0] for p in pvec_list]
    env = np.max(envs, 0)
    spikes, refr = [], 0.0
    for t in range(len(env)):
        if refr > 0:
            refr -= 1
            continue
        if env[t] >= 0.8:
            spikes.append(t)
            refr = 20                          # ms, paper reference
    return np.array(spikes, float), env


def score(spikes_ms, truth_s, tol_ms):
    truth = truth_s * 1000
    if len(spikes_ms) == 0:
        return 0, 0, 0, 0
    used = np.zeros(len(truth), bool)
    tp = 0
    for s in spikes_ms:
        d = np.abs(truth - s)
        i = int(np.argmin(d))
        if d[i] <= tol_ms and not used[i]:
            used[i] = True
            tp += 1
    prec = tp / len(spikes_ms)
    rec = tp / max(len(truth), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    frag = (len(spikes_ms) - tp) / max(len(truth), 1)
    return f1, prec, rec, frag


BANDS = [(40, 120), (120, 400), (400, 1200), (1200, 3500),
         (3500, 7000), (7000, 10500)]


def band_envelopes(path):
    """Per-band 1ms onset-flux envelopes (v2 front end)."""
    from scipy.signal import butter, sosfilt
    wav = OUT / (path.stem + '.wav')
    if not wav.exists():
        subprocess.run(['fluidsynth', '-ni', '-F', str(wav),
                        '-r', str(SR), SF, str(path)],
                       capture_output=True)
    import wave
    w = wave.open(str(wav))
    x = np.frombuffer(w.readframes(w.getnframes()),
                      np.int16).astype(np.float64)
    if w.getnchannels() == 2:
        x = x.reshape(-1, 2).mean(1)
    x /= (np.abs(x).max() + 1e-9)
    hop = SR // 1000
    outs = []
    for lo, hi in BANDS:
        sos = butter(2, [lo, hi], 'bandpass', fs=SR, output='sos')
        xb = sosfilt(sos, x)
        e = np.sqrt(np.convolve(xb * xb, np.ones(4 * hop) / (4 * hop),
                                'same'))[::hop]
        flux = np.maximum(0, np.diff(e, prepend=e[0]))
        outs.append(flux / (np.quantile(flux, 0.99) + 1e-9))
    n = min(len(o) for o in outs)
    return [o[:n] for o in outs]


def merge_events(spike_lists, win=10):
    """Union across bands; events within win ms collapse to one."""
    allsp = np.sort(np.concatenate([s for s in spike_lists
                                    if len(s)]))
    if len(allsp) == 0:
        return allsp
    merged = [allsp[0]]
    for s in allsp[1:]:
        if s - merged[-1] > win:
            merged.append(s)
    return np.array(merged)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import csv
    rows = [r for r in csv.DictReader(open(GMD / 'info.csv'))
            if r['beat_type'] == 'beat'
            and r['time_signature'] == '4-4'
            and 20 < float(r['duration']) < 60]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    files = rows[:12]
    ARMS = {
        'single(default)': [[0, 0, 0, 0]],
        'single(fast-rec)': [[0, -0.5, 0, -1.0]],
        'TWO-voice': [[0, -0.5, 0, -1.0], [0, 2.0, 0, 0.5]],
    }
    print(f'{len(files)} files; tolerance 40 ms', flush=True)
    agg = {a: [] for a in ARMS}
    for r in files:
        p = GMD / r['midi_filename']
        truth = midi_onsets(p)
        drive = audio_envelope(p) * 1.5
        bands = band_envelopes(p)
        for arm, pv in ARMS.items():
            sp, _ = commit(drive, pv)
            f1, prec, rec, frag = score(sp, truth, 40)
            agg[arm].append((f1, prec, rec, frag))
            spb = merge_events([commit(b * 1.5, pv)[0]
                                for b in bands])
            f1, prec, rec, frag = score(spb, truth, 40)
            agg.setdefault(arm + ' +perband', []).append(
                (f1, prec, rec, frag))
    for arm, v in agg.items():
        v = np.array(v)
        print(f'{arm:18s} F1 {v[:,0].mean():.3f}  prec '
              f'{v[:,1].mean():.3f}  rec {v[:,2].mean():.3f}  '
              f'fragmentation {v[:,3].mean():.2f} extra/true',
              flush=True)


if __name__ == '__main__':
    main()
