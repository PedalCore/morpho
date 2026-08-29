"""M11 rung a+b v2 — BEAT-ALIGNED chord lattice on ASAP.

Chroma aggregated per annotated beat, merged while harmonically
stable (cos > 0.9): harmonic rhythm on the true beat lattice.
Root-normalized quality clustering + progression ladder, directly
comparable to the beat-blind ARIA numbers (5.63 b/chord markov-1).

python3 -m whitebox.chords_asap
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ASAP = pathlib.Path.home() / 'datasets' / 'asap'
KQ = 8
MERGE_COS = 0.9


def beat_chroma(path, beats):
    import mido
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    t, ev = 0.0, []
    for m in mid:
        t += m.time
        if m.type == 'note_on' and m.velocity > 0:
            ev.append((t, m.note % 12))
    if len(ev) < 30 or len(beats) < 12:
        return None
    times = np.array([e[0] for e in ev])
    pcs = np.array([e[1] for e in ev])
    ch = np.zeros((len(beats) - 1, 12))
    idx = np.searchsorted(times, beats)
    for b in range(len(beats) - 1):
        for e in range(idx[b], idx[b + 1]):
            ch[b, pcs[e]] += 1
    keep = ch.sum(1) > 0
    ch = ch[keep]
    return ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)


def main():
    ann = json.load(open(ASAP / 'asap_annotations.json'))
    files = sorted(ann.keys())
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    Q, per_file = [], []
    for f in files:
        ch = beat_chroma(ASAP / f, ann[f]['performance_downbeats'])
        if ch is None:
            continue
        # merge harmonically-stable consecutive beats
        spans = []
        cur, dur = ch[0], 1
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
    ntr_f = int(0.9 * len(per_file))
    print(f'{len(per_file)} performances, {len(Q)} beat-spans',
          flush=True)
    from scipy.cluster.vq import kmeans2
    np.random.seed(0)
    cent, asn = kmeans2(Q, KQ, minit='++', seed=0)
    for k in range(KQ):
        top = np.argsort(cent[k])[::-1][:4]
        w = cent[k][top] / (cent[k].sum() + 1e-9)
        desc = ' '.join(f'+{t}({w[i]:.2f})' for i, t in enumerate(top))
        print(f'  q{k}: {desc}  [{(asn == k).sum()}]', flush=True)

    V = 12 * KQ

    def to_syms(rows):
        out = []
        for row in rows:
            s, prev = [], None
            for qi, r in row:
                iv = 0 if prev is None else (r - prev) % 12
                s.append(iv * KQ + int(asn[qi]))
                prev = r
            if len(s) > 4:
                out.append(s)
        return out
    tr = to_syms(per_file[:ntr_f])
    te = to_syms(per_file[ntr_f:])
    eps = 0.5
    uni = np.zeros(V)
    m1, m2 = {}, {}
    for s in tr:
        for i in range(2, len(s)):
            uni[s[i]] += 1
            m1.setdefault((s[i-1],), np.zeros(V))[s[i]] += 1
            m2.setdefault((s[i-2], s[i-1]), np.zeros(V))[s[i]] += 1

    def bits(counts, ctx_of):
        tot, n = 0.0, 0
        for s in te:
            for i in range(2, len(s)):
                c = counts.get(ctx_of(s, i))
                p = ((c[s[i]] + eps) / (c.sum() + eps * V)
                     if c is not None else 1.0 / V)
                tot -= np.log2(p)
                n += 1
        return tot / n
    print(f'bits/chord: uniform {np.log2(V):.2f} | unigram '
          f'{bits({(): uni}, lambda s, i: ()):.2f} | markov-1 '
          f'{bits(m1, lambda s, i: (s[i-1],)):.2f} | markov-2 '
          f'{bits(m2, lambda s, i: (s[i-2], s[i-1])):.2f}',
          flush=True)


if __name__ == '__main__':
    main()
