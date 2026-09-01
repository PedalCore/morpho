"""M12 measure 2 — do HEU-committed events form stable
representations under gradient-free consolidation? (their project's
central question, at the seam of the two systems).

Per file: ground-truth MIDI onsets AND HEU-committed events (best-
precision arm: single-voice broadband) -> 20ms single-channel
rasters -> bar grids (16 slots via true bpm) -> MotifStore-style
consolidation -> stability metrics. Compare truth vs committed.

python3 -m whitebox.m12_stability
"""

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path.home() / 'coding' /
                       'heu-replication'))

from whitebox.m12_frontend import (GMD, audio_envelope, commit,     # noqa
                                   midi_onsets)
from whitebox.stability_metrics import (consolidation_weighted_recall,  # noqa
                                        retrieval_margin)

BIN = 0.020


class Store:
    """MotifStore logic, dimension-agnostic (16-d bar grids)."""

    def __init__(self, d, m=16):
        self.W = np.zeros((m, d))
        self.g = np.zeros(m, int)
        self.theta = np.full(m, 0.35)

    def store(self, x):
        if x.sum() == 0:
            return
        n = np.linalg.norm(self.W, axis=1) * (np.linalg.norm(x) + 1e-9)
        s = (self.W @ x) / (n + 1e-9)
        i = int(np.argmax(s))
        delta = float(((self.W[i] - x) ** 2).mean())
        if self.g[i] == 0 or delta <= self.theta[i]:
            self.W[i] += 0.35 * (x - self.W[i])
            self.g[i] += 1
            self.theta[i] *= (1 - 1 / (4.0 + self.g[i]))
        else:
            j = int(np.argmin(self.g))
            self.W[j] = x.copy()
            self.g[j] = 1


def bar_grids_1ch(events_s, bpm, total_s):
    bar = 4 * 60 / bpm
    nb = int(total_s / bar)
    G = np.zeros((nb, 16))
    for t in events_s:
        b = int(t / bar)
        if b < nb:
            G[b, min(15, int((t % bar) / bar * 16))] = 1
    return G


def run_source(name, events_by_file):
    margins, recalls = [], []
    rng = np.random.default_rng(0)
    for (events, bpm, dur) in events_by_file:
        G = bar_grids_1ch(events, bpm, dur)
        st = Store(16)
        for g in G:
            st.store(g)
        live = st.g > 0
        if live.sum() < 3:
            continue
        P, wts = st.W[live], st.g[live]
        idx = rng.choice(len(P), size=min(8, len(P)), replace=False)
        cues = P[idx] * (rng.random(P[idx].shape) > 0.3)
        m, _ = retrieval_margin(P, cues, idx)
        r = consolidation_weighted_recall(P, wts, cues, idx)
        margins.append(np.median(m))
        recalls.append(r)
    print(f'{name:22s} margin median {np.median(margins):+.3f}  '
          f'consol recall@1 {np.mean(recalls):.2f}  '
          f'({len(margins)} files)', flush=True)


def main():
    rows = [r for r in csv.DictReader(open(GMD / 'info.csv'))
            if r['beat_type'] == 'beat'
            and r['time_signature'] == '4-4'
            and 20 < float(r['duration']) < 60]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    truth_src, heu_src = [], []
    for r in rows[:12]:
        p = GMD / r['midi_filename']
        bpm, dur = float(r['bpm']), float(r['duration'])
        truth = midi_onsets(p)
        drive = audio_envelope(p) * 1.5
        sp_ms, _ = commit(drive, [[0, 0, 0, 0]])
        truth_src.append((truth, bpm, dur))
        heu_src.append((sp_ms / 1000.0, bpm, dur))
    run_source('ground-truth MIDI', truth_src)
    run_source('HEU-committed (F1 .29)', heu_src)
    # DENSITY CONTROL: truth subsampled to the HEU event count
    sub_src = []
    r2 = np.random.default_rng(1)
    for (tr, bpm, dur), (he, _, _) in zip(truth_src, heu_src):
        k = min(len(he), len(tr))
        sub = np.sort(r2.choice(tr, size=k, replace=False))
        sub_src.append((sub, bpm, dur))
    run_source('truth @ matched density', sub_src)


if __name__ == '__main__':
    main()
