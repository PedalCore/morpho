"""M10 v2 — collaborator-sharpened ladder (see M10-RHYTHM.md).

Metrics: bits/EVENT over the rate baseline (not bits/bin); event F1
at +/-20/40/80 ms; oracle decomposition (true-BPM / bar-phase);
complexity ledger (trainable, state, ops/step). Free-running eval in
rhythm3 (next).

python3 -m whitebox.rhythm2          # runs the whole v2 ladder
"""

import csv
import json
import pathlib
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

GMD = pathlib.Path.home() / 'datasets' / 'groove'
OUT = pathlib.Path('whitebox/runs/rhythm')
BIN = 0.020
TRACE_HL = [0.02 * 2 ** i for i in range(10)]     # 20ms .. 10.24s
PERIODS = np.geomspace(0.25, 4.0, 16)
OSC_RHO = 0.9995
FIR_TAPS = 500


def load_split(split):
    """Returns list of (spike array, bpm, beats_per_bar)."""
    rows = list(csv.DictReader(open(GMD / 'info.csv')))
    data = np.load(OUT / 'gmd.npz', allow_pickle=True)[split]
    metas = [r for r in rows if r['split'] == split]
    out = []
    i = 0
    import mido
    for r in metas:
        try:
            mid = mido.MidiFile(GMD / r['midi_filename'])
        except Exception:
            continue
        n_on = sum(1 for m in mid if m.type == 'note_on'
                   and m.velocity > 0)
        if n_on < 8:
            continue                      # same filter as prep()
        x = data[i]
        i += 1
        bpb = int(r['time_signature'].split('-')[0])
        out.append((x, float(r['bpm']), bpb))
    assert i == len(data), f'alignment: {i} vs {len(data)}'
    return out


def trace_feats(x):
    xf = x.astype(np.float64)
    cols = []
    for hl in TRACE_HL:
        lam = 0.5 ** (BIN / hl)
        c = lfilter([1.0], [1.0, -lam], xf)
        cols.append(np.concatenate([[0.0], c[:-1]]))
    return np.stack(cols, 1).astype(np.float32)


def osc_feats(x):
    xf = x.astype(complex)
    cols = []
    for T in PERIODS:
        rho = 0.5 ** (BIN / (4 * T))     # integrate ~4 periods, not 40s
        pole = rho * np.exp(1j * 2 * np.pi * BIN / T)
        z = lfilter([1.0], [1.0, -pole], xf)
        z = np.concatenate([[0.0], z[:-1]])
        cols.append(z.real)
        cols.append(z.imag)
    return np.stack(cols, 1).astype(np.float32)


def fir_feats(x):
    xf = np.concatenate([np.zeros(FIR_TAPS), x.astype(np.float32)])
    W = np.lib.stride_tricks.sliding_window_view(xf, FIR_TAPS)[:-1]
    return W.astype(np.float32)                   # row t: x_{t-500..t-1}


def oracle_feats(x, bpm, bpb):
    t = np.arange(len(x)) * BIN
    beat = 2 * np.pi * t * bpm / 60.0
    # anchor at the CONCENTRATED metrical level: 16th notes (beat
    # phase is near-uniform — drummers occupy all 16th slots; R_16
    # ~0.6 vs R_beat ~0.07, measured)
    on = np.where(x > 0)[0]
    s16 = 4 * beat
    phi16 = np.angle(np.exp(1j * s16[on]).sum()) if len(on) else 0.0
    s16 = s16 - phi16
    beat = beat - phi16 / 4
    bar = beat / bpb
    # phase-BIN features: piecewise-constant hazard over the cycle
    # (a linear sin/cos readout = 1st harmonic only — useless for a
    # comb-shaped hazard; 16 bins express the comb)
    def bins(theta, k=16):
        idx = np.minimum(((theta / (2 * np.pi)) % 1.0 * k)
                         .astype(int), k - 1)
        oh = np.zeros((len(theta), k), np.float32)
        oh[np.arange(len(theta)), idx] = 1.0
        return oh
    return np.concatenate([bins(s16), bins(bar)], 1)


def assemble(split, kind):
    Xs, ys = [], []
    for x, bpm, bpb in load_split(split):
        if kind == 'trace':
            F = trace_feats(x)
        elif kind == 'osc':
            F = osc_feats(x)
        elif kind == 'trace+osc':
            F = np.concatenate([trace_feats(x), osc_feats(x)], 1)
        elif kind == 'fir':
            F = fir_feats(x)
        elif kind == 'oracle-beat':
            F = oracle_feats(x, bpm, bpb)[:, :16]
        elif kind == 'oracle-bar':
            F = oracle_feats(x, bpm, bpb)
        elif kind == 'oracle+trace':
            F = np.concatenate([trace_feats(x),
                                oracle_feats(x, bpm, bpb)], 1)
        Xs.append(F)
        ys.append(x)
    return np.concatenate(Xs), np.concatenate(ys)


def readout(Xtr, ytr, Xte, hidden=0, epochs=10, seed=0, sub=None):
    import torch
    torch.manual_seed(seed)
    if sub:
        idx = np.random.default_rng(0).choice(len(Xtr), sub,
                                              replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    b0 = float(np.log(ytr.mean() / (1 - ytr.mean())))
    if hidden:
        net = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1))
        torch.nn.init.constant_(net[-1].bias, b0)
    else:
        net = torch.nn.Linear(Xtr.shape[1], 1)
        torch.nn.init.constant_(net.bias, b0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(
        ytr.astype(np.float32))
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 65536):
            ii = perm[i:i+65536]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(Xt[ii]).squeeze(-1), yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xte)).squeeze(-1)).numpy()
    return p, sum(q.numel() for q in net.parameters())


def bits_per_event(p, y, p0):
    eps = 1e-7
    nll = -(y * np.log2(p + eps) + (1 - y) * np.log2(1 - p + eps)).sum()
    nll0 = -(y * np.log2(p0 + eps) +
             (1 - y) * np.log2(1 - p0 + eps)).sum()
    return (nll0 - nll) / max(y.sum(), 1)


def event_f1(p, y, tol_bins):
    pred = p > 0.5
    k = np.ones(2 * tol_bins + 1)
    prec = (pred & (np.convolve(y, k, 'same') > 0)).sum() / \
        max(pred.sum(), 1)
    rec = (y.astype(bool) &
           (np.convolve(pred, k, 'same') > 0)).sum() / max(y.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    LADDER = [
        ('L1-traces', 'trace', 0, None),
        ('L2-osc', 'osc', 0, None),
        ('L3-traces+osc', 'trace+osc', 0, None),
        ('L4-FIR500', 'fir', 0, 500_000),
        ('L5-nonlin', 'trace+osc', 16, None),
        ('O-beat', 'oracle-beat', 0, None),
        ('O-bar', 'oracle-bar', 0, None),
        ('O-bar+traces', 'oracle+trace', 0, None),
    ]
    STATE = {'L1-traces': 10, 'L2-osc': 32, 'L3-traces+osc': 42,
             'L4-FIR500': 500, 'L5-nonlin': 42, 'O-beat': 0,
             'O-bar': 0, 'O-bar+traces': 10}
    ytr_all = assemble('train', 'trace')[1]
    p0 = ytr_all.mean()
    results = {}
    print(f'base rate {p0:.4f}; bits/event over rate baseline:',
          flush=True)
    for name, kind, hidden, sub in LADDER:
        Xtr, ytr = assemble('train', kind)
        Xte, yte = assemble('test', kind)
        p, npar = readout(Xtr, ytr, Xte, hidden=hidden, sub=sub)
        bpe = bits_per_event(p, yte, p0)
        f1s = {t: event_f1(p, yte, t) for t in (1, 2, 4)}
        results[name] = dict(trainable=npar, state=STATE[name],
                             bits_per_event=float(bpe),
                             f1_20ms=float(f1s[1]),
                             f1_40ms=float(f1s[2]),
                             f1_80ms=float(f1s[4]))
        print(f'{name:14s} train {npar:4d} state {STATE[name]:4d} '
              f'| {bpe:+.3f} b/ev | F1 {f1s[1]:.3f}/{f1s[2]:.3f}/'
              f'{f1s[4]:.3f}', flush=True)
    (OUT / 'v2.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
