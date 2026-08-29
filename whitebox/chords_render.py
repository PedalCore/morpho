"""M11 — audible progressions: render (a) a REAL performance's
chord lattice and (b) a Markov-2 SAMPLED progression through the
same block-chord decoder. The pair isolates model vs abstraction.

python3 -m whitebox.chords_render
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.chords_asap import (ASAP, KQ, MERGE_COS,             # noqa
                                  beat_chroma)

OUT = pathlib.Path('whitebox/runs/piano')
BAR_S = 2.4                      # render tempo: 100 bpm, 4/4


def build_lattice():
    ann = json.load(open(ASAP / 'asap_annotations.json'))
    files = sorted(ann.keys())
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    Q, per_file = [], []
    for f in files:
        ch = beat_chroma(ASAP / f, ann[f]['performance_downbeats'])
        if ch is None:
            continue
        spans, cur, dur = [], ch[0], 1
        for i in range(1, len(ch)):
            if float(cur @ ch[i]) > MERGE_COS:
                cur = cur * dur + ch[i]
                cur /= np.linalg.norm(cur) + 1e-9
                dur += 1
            else:
                spans.append(cur)
                cur, dur = ch[i], 1
        spans.append(cur)
        row = []
        for c in spans:
            r = int(np.argmax(c))
            Q.append(np.roll(c, -r))
            row.append((len(Q) - 1, r))
        per_file.append(row)
    Q = np.array(Q)
    from scipy.cluster.vq import kmeans2
    np.random.seed(0)
    cent, asn = kmeans2(Q, KQ, minit='++', seed=0)
    return cent, asn, per_file


def chord_pitches(cent_q, root):
    pcs = [p for p in np.argsort(cent_q)[::-1][:4]
           if cent_q[p] > 0.08]
    notes = sorted(((root + p) % 12) + 60 for p in pcs)
    return [notes[0] - 12] + notes            # bass double


def render(sym_seq, cent, path):
    import mido
    mid = mido.MidiFile()
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tpb = mid.ticks_per_beat
    bar_ticks = int(BAR_S / 0.5 * tpb)
    root = 0
    ev = []
    for b, sym in enumerate(sym_seq):
        iv, q = sym // KQ, sym % KQ
        root = (root + iv) % 12
        for n in chord_pitches(cent[q], root):
            ev.append((b * bar_ticks, 1, n))
            ev.append((b * bar_ticks + bar_ticks - 10, 0, n))
    ev.sort()
    prev = 0
    for tt, on, n in ev:
        tr.append(mido.Message('note_on' if on else 'note_off',
                               note=int(n), velocity=80 if on else 0,
                               time=int(tt - prev)))
        prev = tt
    mid.save(path)


def main():
    cent, asn, per_file = build_lattice()
    V = 12 * KQ

    def to_syms(row):
        s, prev = [], None
        for qi, r in row:
            iv = 0 if prev is None else (r - prev) % 12
            s.append(iv * KQ + int(asn[qi]))
            prev = r
        return s
    ntr = int(0.9 * len(per_file))
    m1, m2 = {}, {}
    for row in per_file[:ntr]:
        s = to_syms(row)
        for i in range(2, len(s)):
            m1.setdefault((s[i-1],), np.zeros(V))[s[i]] += 1
            m2.setdefault((s[i-2], s[i-1]), np.zeros(V))[s[i]] += 1
    # (a) real lattice excerpt
    real = to_syms(max(per_file[ntr:], key=len))[:16]
    render(real, cent, OUT / 'prog_real.mid')
    # (b) markov-2 sample seeded from the real excerpt's first two
    rng = np.random.default_rng(2)
    s = real[:2]
    for _ in range(14):
        c = m2.get((s[-2], s[-1]))
        if c is None or c.sum() < 3:
            c = m1.get((s[-1],), np.ones(V))
        p = c / c.sum()
        s.append(int(rng.choice(V, p=p)))
    render(s, cent, OUT / 'prog_markov2.mid')
    print('rendered prog_real.mid / prog_markov2.mid (16 bars each)',
          flush=True)


if __name__ == '__main__':
    main()
