"""M11 rung a+b — chord lattice from ARIA + progression baselines.

a: segment by harmonic change (chroma cosine), root-normalize each
   span's chroma, k-means the QUALITIES (transposition-invariant);
   vocabulary = (quality k, root r). Inspect centroids: triads?
b: progression models over (interval, quality) symbols:
   unigram / Markov-1 / Markov-2 / decayed-trace logistic.
   Metric: bits/chord.

python3 -m whitebox.chords
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.piano_ladder import BIN, NK, OUT, P0                 # noqa

KQ = 8                       # quality clusters
HOP_S, MERGE_COS = 0.5, 0.85
NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A',
         'Bb', 'B']


def spans_of(x):
    """Harmonic spans: (chroma, dur) via hop-chroma + merge."""
    hop = int(HOP_S / BIN)
    pcmap = np.array([(k + P0) % 12 for k in range(NK)])
    n = len(x) // hop
    if n < 2:
        return []
    ch = np.zeros((n, 12))
    for i in range(n):
        w = x[i*hop:(i+1)*hop]
        for k in range(NK):
            ch[i, pcmap[k]] += w[:, k].sum()
    ch = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    spans = []
    cur, dur = ch[0], 1
    for i in range(1, n):
        if float(cur @ ch[i]) > MERGE_COS:
            cur = (cur * dur + ch[i]) / (dur + 1)
            cur /= np.linalg.norm(cur) + 1e-9
            dur += 1
        else:
            spans.append((cur, dur))
            cur, dur = ch[i], 1
    spans.append((cur, dur))
    return spans


def root_norm(c):
    r = int(np.argmax(c))
    return np.roll(c, -r), r


def main():
    from scipy.cluster.vq import kmeans2
    store = np.load(OUT / 'aria88.npy', allow_pickle=True).item()
    seqs_q, roots, seqs = [], [], {}
    Q, R, F = [], [], []
    for split in ('train', 'test'):
        per_file = []
        for x in store[split]:
            sp = spans_of(x)
            row = []
            for c, dur in sp:
                q, r = root_norm(c)
                Q.append(q)
                row.append((len(Q) - 1, r))
            per_file.append(row)
        seqs[split] = per_file
    Q = np.array(Q)
    np.random.seed(0)
    cent, asn = kmeans2(Q, KQ, minit='++', seed=0)
    print(f'{len(Q)} spans; quality centroids (top pcs rel. root):')
    for k in range(KQ):
        top = np.argsort(cent[k])[::-1][:4]
        w = cent[k][top] / (cent[k].sum() + 1e-9)
        desc = ' '.join(f'+{t}({w[i]:.2f})' for i, t in enumerate(top))
        print(f'  q{k}: {desc}  [{(asn==k).sum()} spans]')

    # symbol sequences: (root interval mod 12, quality)
    def to_syms(per_file):
        out = []
        for row in per_file:
            s = []
            prev_r = None
            for qi, r in row:
                iv = 0 if prev_r is None else (r - prev_r) % 12
                s.append(iv * KQ + int(asn[qi]))
                prev_r = r
            if len(s) > 4:
                out.append(s)
        return out
    tr = to_syms(seqs['train'])
    te = to_syms(seqs['test'])
    V = 12 * KQ
    ntr = sum(len(s) for s in tr)
    print(f'vocab {V}, {ntr} train / {sum(len(s) for s in te)} test '
          f'chords, median span {np.median([d for _, d in []] or [0])}')

    eps = 0.5

    def bits(counts, ctx_of):
        tot, n = 0.0, 0
        for s in te:
            for i in range(2, len(s)):
                c = counts.get(ctx_of(s, i), None)
                p = ((c[s[i]] + eps) / (c.sum() + eps * V)
                     if c is not None else 1.0 / V)
                tot -= np.log2(p)
                n += 1
        return tot / n

    uni = np.zeros(V)
    m1, m2 = {}, {}
    for s in tr:
        for i in range(2, len(s)):
            uni[s[i]] += 1
            m1.setdefault((s[i-1],), np.zeros(V))[s[i]] += 1
            m2.setdefault((s[i-2], s[i-1]), np.zeros(V))[s[i]] += 1
    b_uni = bits({(): uni}, lambda s, i: ())
    b_m1 = bits(m1, lambda s, i: (s[i-1],))
    b_m2 = bits(m2, lambda s, i: (s[i-2], s[i-1]))
    print(f'bits/chord: uniform {np.log2(V):.2f} | unigram '
          f'{b_uni:.2f} | markov-1 {b_m1:.2f} | markov-2 {b_m2:.2f}')
    np.savez(OUT / 'chords.npz', cent=cent)


if __name__ == '__main__':
    main()
