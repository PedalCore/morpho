"""M10-P — piano spike ladder on ARIA (free-clock; M10-RHYTHM.md).

88 onset streams, 20 ms bins. Rungs: per-key traces (880 state) vs
12 PITCH-CLASS traces (120 state) vs both vs +copy-prev(2s window).
Metric: bits/event over per-key rates.

python3 -m whitebox.piano_ladder --prep --limit 1500
python3 -m whitebox.piano_ladder --sweep
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ARIA = pathlib.Path.home() / 'datasets' / 'aria'
OUT = pathlib.Path('whitebox/runs/piano')
BIN = 0.020
NK, P0 = 88, 21
HL = [0.04, 0.16, 0.64, 2.56, 10.24]          # 5 half-lives (s)


def prep(limit):
    import mido
    files = sorted(ARIA.rglob('*.mid'))[:limit]
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    items = []
    for f in files:
        try:
            mid = mido.MidiFile(f)
        except Exception:
            continue
        t, ev = 0.0, []
        for m in mid:
            t += m.time
            if m.type == 'note_on' and m.velocity > 0 \
                    and P0 <= m.note < P0 + NK:
                ev.append((t, m.note - P0))
        if len(ev) < 100 or t > 300:
            continue
        n = int(ev[-1][0] / BIN) + 2
        x = np.zeros((n, NK), np.uint8)
        for tt, k in ev:
            x[int(tt / BIN), k] = 1
        items.append(x)
        if len(items) % 200 == 0:
            print(f'{len(items)} files', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ntr = int(0.9 * len(items))
    np.save(OUT / 'aria88.npy',
            np.array({'train': items[:ntr], 'test': items[ntr:]},
                     dtype=object))
    print(f'{ntr} train / {len(items)-ntr} test files', flush=True)


def _traces(sig, hls):
    cols = []
    for hl in hls:
        lam = 0.5 ** (BIN / hl)
        c = lfilter([1 - lam], [1, -lam], sig.astype(np.float64))
        cols.append(np.concatenate([np.zeros((1, sig.shape[1])),
                                    c[:-1]]))
    return np.concatenate(cols, 1).astype(np.float32)


def feats(x, kind):
    F = []
    if 'key' in kind:
        F.append(_traces(x, HL))               # 88*5 = 440 cols
    if 'pc' in kind:
        pc = np.zeros((len(x), 12), np.uint8)
        for k in range(NK):
            pc[:, (k + P0) % 12] |= x[:, k]
        F.append(_traces(pc, HL))              # 60 cols
    if 'copy' in kind:
        w = int(2.0 / BIN)
        cp = np.zeros_like(x, np.float32)
        cp[w:] = x[:-w]
        F.append(cp)                           # crude 2s-ago echo
    return np.concatenate(F, 1) if F else \
        np.zeros((len(x), 0), np.float32)


def sweep():
    import torch
    store = np.load(OUT / 'aria88.npy', allow_pickle=True).item()

    def build(split, kind):
        X = [feats(x, kind)[::2] for x in store[split]]   # thin rows
        Y = [x[::2] for x in store[split]]
        return (np.concatenate(X),
                np.concatenate(Y).astype(np.float32))

    _, Ytr0 = build('train', ())
    r0 = np.clip(Ytr0.mean(0), 1e-6, 1 - 1e-6)
    eps = 1e-7
    _, Yte0 = build('test', ())
    P0b = np.broadcast_to(r0, Yte0.shape)
    nll0 = -(Yte0 * np.log2(P0b + eps) +
             (1 - Yte0) * np.log2(1 - P0b + eps)).sum()
    nev = Yte0.sum()
    print(f'events/test {nev/1e3:.0f}k  mean rate {r0.mean():.4f}',
          flush=True)
    for kind, state in ((('pc',), 60), (('key',), 440),
                        (('key', 'pc'), 500),
                        (('key', 'pc', 'copy'), 500)):
        Xtr, Ytr = build('train', kind)
        Xte, Yte = build('test', kind)
        torch.manual_seed(0)
        net = torch.nn.Linear(Xtr.shape[1], NK)
        with torch.no_grad():
            net.bias.copy_(torch.from_numpy(
                np.log(r0 / (1 - r0)).astype(np.float32)))
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xt = torch.from_numpy((Xtr - mu) / sd)
        Yt = torch.from_numpy(Ytr)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        for ep in range(4):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 32768):
                ii = perm[i:i+32768]
                loss = torch.nn.functional.\
                    binary_cross_entropy_with_logits(net(Xt[ii]),
                                                     Yt[ii])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            P = torch.sigmoid(net(torch.from_numpy(
                (Xte - mu) / sd))).numpy()
        nll = -(Yte * np.log2(P + eps) +
                (1 - Yte) * np.log2(1 - P + eps)).sum()
        b = (nll0 - nll) / nev
        npar = sum(p.numel() for p in net.parameters())
        print(f'{"+".join(kind):12s} state {state:4d} '
              f'params {npar:6d} | {b:+.3f} b/ev', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--limit', type=int, default=400)
    a = ap.parse_args()
    if a.prep:
        prep(a.limit)
    if a.sweep:
        sweep()


if __name__ == '__main__':
    main()
