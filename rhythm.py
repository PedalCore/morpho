"""M10 — spikes ARE the rhythm: minimal-machinery ladder on the
Groove MIDI Dataset (M10-RHYTHM.md).

python3 -m whitebox.rhythm --prep     # parse GMD -> binned spike trains
python3 -m whitebox.rhythm --sweep    # run ladder L0-L3, print table
python3 -m whitebox.rhythm --continue-rung L3   # render continuation

Representation: 20 ms bins, all drum hits -> one channel, on/off.
Predict P(spike in bin t) from history < t. Dynamics (LIF decay,
oscillator rotations, counter decays) are FIXED physics; only
readouts are trained — parameter counts are honest.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.signal import lfilter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

GMD = pathlib.Path.home() / 'datasets' / 'groove'
OUT = pathlib.Path('whitebox/runs/rhythm')
BIN = 0.020                                   # 20 ms
PERIODS = np.geomspace(0.25, 4.0, 16)         # oscillator periods (s)
OSC_RHO = 0.9995
CNT_HL = np.array([0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24])
N_TAPS = 16


def prep():
    import csv
    import mido
    splits = {'train': [], 'validation': [], 'test': []}
    with open(GMD / 'info.csv') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        p = GMD / r['midi_filename']
        try:
            mid = mido.MidiFile(p)
        except Exception:
            continue
        t, ons = 0.0, []
        for msg in mid:
            t += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                ons.append(t)
        if len(ons) < 8:
            continue
        n = int(ons[-1] / BIN) + 2
        x = np.zeros(n, dtype=np.uint8)
        x[(np.array(ons) / BIN).astype(int)] = 1
        splits[r['split']].append(x)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'gmd.npz',
                        **{s: np.array(v, dtype=object)
                           for s, v in splits.items()},
                        allow_pickle=True)
    for s, v in splits.items():
        tot = sum(len(a) for a in v)
        rate = sum(a.sum() for a in v) / max(tot, 1)
        print(f'{s}: {len(v)} files, {tot/1e6:.2f}M bins, '
              f'spike rate {rate:.3f}', flush=True)


def feats_of(x):
    """Causal features for predicting bin t from history < t."""
    xf = x.astype(np.float64)
    cols, names = [], []
    for i in range(1, N_TAPS + 1):            # raw taps x_{t-i}
        cols.append(np.concatenate([np.zeros(i), xf[:-i]]))
        names.append(f'tap{i}')
    for hl in CNT_HL:                         # decayed counters
        lam = 0.5 ** (BIN / hl)
        c = lfilter([1.0], [1.0, -lam], xf)
        cols.append(np.concatenate([[0.0], c[:-1]]))
        names.append(f'cnt{hl}')
    for T in PERIODS:                         # oscillators (driven)
        pole = OSC_RHO * np.exp(1j * 2 * np.pi * BIN / T)
        z = lfilter([1.0], [1.0, -pole], xf.astype(complex))
        z = np.concatenate([[0.0], z[:-1]])
        cols.append(z.real)
        cols.append(z.imag)
        names.append(f'osc{T}re')
        names.append(f'osc{T}im')
    return np.stack(cols, 1).astype(np.float32), names


FEATSETS = {
    'L0b-markov': lambda n: [i for i, s in enumerate(n)
                             if s.startswith('tap')][:8],
    'L1-lif': lambda n: [n.index('tap1'), n.index(f'cnt{CNT_HL[2]}')],
    'L2-osc': lambda n: [i for i, s in enumerate(n)
                         if s.startswith('osc')],
    'L3-all': lambda n: list(range(len(n))),
}


def _readout(Xtr, ytr, Xte, hidden=0, epochs=10, seed=0):
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    d = Xtr.shape[1]
    b0 = float(np.log(ytr.mean() / (1 - ytr.mean())))
    if hidden:
        net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1))
        torch.nn.init.constant_(net[-1].bias, b0)
    else:
        net = torch.nn.Linear(d, 1)
        torch.nn.init.constant_(net.bias, b0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr.astype(np.float32))
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 65536):
            idx = perm[i:i+65536]
            p = net(Xt[idx]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                p, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pte = torch.sigmoid(net(torch.from_numpy(Xte)).squeeze(-1))
    nparams = sum(p.numel() for p in net.parameters())
    return pte.numpy(), nparams, net


def metrics(p, y):
    eps = 1e-7
    nll = -np.mean(y * np.log2(p + eps) + (1 - y) * np.log2(1 - p + eps))
    pred = p > 0.5
    hit = pred & (np.convolve(y, [1, 1, 1], 'same') > 0)
    prec = hit.sum() / max(pred.sum(), 1)
    truehit = y.astype(bool) & (np.convolve(pred, [1, 1, 1], 'same') > 0)
    rec = truehit.sum() / max(y.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return nll, prec, rec, f1


def load_feats(split, data):
    Xs, ys = [], []
    for x in data[split]:
        F, names = feats_of(x)
        Xs.append(F)
        ys.append(x)
    return np.concatenate(Xs), np.concatenate(ys), names


def sweep():
    data = np.load(OUT / 'gmd.npz', allow_pickle=True)
    Xtr, ytr, names = load_feats('train', data)
    Xte, yte, _ = load_feats('test', data)
    print(f'train {len(ytr)/1e6:.2f}M bins, test {len(yte)/1e6:.2f}M',
          flush=True)
    results = {}
    r = ytr.mean()                            # L0a constant rate
    nll, prec, rec, f1 = metrics(np.full(len(yte), r, np.float32), yte)
    results['L0a-rate'] = dict(params=1, nll=nll, f1=f1)
    print(f'L0a-rate         1 params  nll {nll:.4f}  f1 {f1:.3f}',
          flush=True)
    for name, sel in FEATSETS.items():
        idx = sel(names)
        hidden = 16 if name == 'L3-all' else 0
        p, nparams, net = _readout(Xtr[:, idx], ytr, Xte[:, idx],
                                   hidden=hidden)
        nll, prec, rec, f1 = metrics(p, yte)
        results[name] = dict(params=nparams, nll=nll, f1=f1,
                             prec=prec, rec=rec)
        print(f'{name:16s} {nparams:4d} params  nll {nll:.4f}  '
              f'f1 {f1:.3f}  (p {prec:.3f} r {rec:.3f})', flush=True)
        import torch
        torch.save(net.state_dict(), OUT / f'{name}.pt')
    (OUT / 'results.json').write_text(json.dumps(
        results, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    if args.prep:
        prep()
    if args.sweep:
        sweep()


if __name__ == '__main__':
    main()
