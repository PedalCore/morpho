"""M10 v3 — nine-voice clocked drum process (collaborator spec).

x_t in {0,1}^9 (Magenta 9-class kit reduction), 20 ms bins,
clock-conditioned (16th-phase + bar-phase bins from true BPM).
Rungs: clock-only / +own-voice traces / +all-voice traces /
+same-tick coupling (pseudolikelihood upper bound, noted).
Free-running eval: per-bar rate drift, metrical concentration R16,
voice-distribution drift.

python3 -m whitebox.rhythm3 --prep ; --sweep ; --freerun
"""

import argparse
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
TRACE_HL = [0.02 * 2 ** i for i in range(10)]
NV = 9
# Magenta 9-class mapping
VOICE = {36: 0, 38: 1, 40: 1, 37: 1, 42: 2, 44: 2, 22: 2, 46: 3,
         26: 3, 43: 4, 58: 4, 47: 5, 45: 5, 50: 6, 48: 6, 49: 7,
         55: 7, 57: 7, 52: 7, 51: 8, 59: 8, 53: 8}


def prep():
    import mido
    rows = list(csv.DictReader(open(GMD / 'info.csv')))
    store = {}
    for split in ('train', 'test'):
        items = []
        for r in [q for q in rows if q['split'] == split]:
            try:
                mid = mido.MidiFile(GMD / r['midi_filename'])
            except Exception:
                continue
            t, ev = 0.0, []
            for m in mid:
                t += m.time
                if m.type == 'note_on' and m.velocity > 0 \
                        and m.note in VOICE:
                    ev.append((t, VOICE[m.note]))
            if len(ev) < 8:
                continue
            n = int(ev[-1][0] / BIN) + 2
            x = np.zeros((n, NV), np.uint8)
            for tt, v in ev:
                x[int(tt / BIN), v] = 1
            items.append(dict(x=x, bpm=float(r['bpm']),
                              bpb=int(r['time_signature'].split('-')[0])))
        store[split] = items
        print(f'{split}: {len(items)} files', flush=True)
    np.save(OUT / 'gmd9.npy', np.array(store, dtype=object))


def clock_feats(x, bpm):
    t = np.arange(len(x)) * BIN
    beat = 2 * np.pi * t * bpm / 60.0
    any_on = np.where(x.any(1))[0]
    s16 = 4 * beat
    phi = np.angle(np.exp(1j * s16[any_on]).sum()) if len(any_on) else 0
    s16 = s16 - phi
    bar = (beat - phi / 4) / 4

    def bins(theta, k=16):
        idx = np.minimum(((theta / (2 * np.pi)) % 1.0 * k).astype(int),
                         k - 1)
        oh = np.zeros((len(theta), k), np.float32)
        oh[np.arange(len(theta)), idx] = 1.0
        return oh
    return np.concatenate([bins(s16), bins(bar)], 1)


def traces9(x):
    cols = []
    for v in range(NV):
        xf = x[:, v].astype(np.float64)
        for hl in TRACE_HL:
            lam = 0.5 ** (BIN / hl)
            c = lfilter([1.0], [1.0, -lam], xf)
            cols.append(np.concatenate([[0.0], c[:-1]]))
    return np.stack(cols, 1).astype(np.float32)   # (T, 90)


def assemble(split, store):
    Cs, Ts, Ys = [], [], []
    for it in store[split]:
        Cs.append(clock_feats(it['x'], it['bpm']))
        Ts.append(traces9(it['x']))
        Ys.append(it['x'])
    return (np.concatenate(Cs), np.concatenate(Ts),
            np.concatenate(Ys).astype(np.float32))


def fit_heads(Xtr, Ytr, Xte, epochs=8, seed=0):
    """9 independent logistic heads on shared features."""
    import torch
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    nout = Ytr.shape[1]
    net = torch.nn.Linear(Xtr.shape[1], nout)
    with torch.no_grad():
        r = Ytr.mean(0).clip(1e-4, 1 - 1e-4)
        net.bias.copy_(torch.from_numpy(np.log(r / (1 - r))))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 65536):
            ii = perm[i:i+65536]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(Xt[ii]), Yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(Xte))).numpy()
    return P, sum(p.numel() for p in net.parameters()), net, (mu, sd)


def bpe9(P, Y, r0):
    eps = 1e-7
    nll = -(Y * np.log2(P + eps) + (1 - Y) * np.log2(1 - P + eps)).sum()
    P0 = np.broadcast_to(r0, Y.shape)
    nll0 = -(Y * np.log2(P0 + eps) +
             (1 - Y) * np.log2(1 - P0 + eps)).sum()
    return (nll0 - nll) / max(Y.sum(), 1)


def sweep():
    store = np.load(OUT / 'gmd9.npy', allow_pickle=True).item()
    Ctr, Ttr, Ytr = assemble('train', store)
    Cte, Tte, Yte = assemble('test', store)
    r0 = Ytr.mean(0)
    print(f'voices rate: {np.round(r0, 3)}', flush=True)
    results = {}

    def run(name, Xtr, Xte):
        P, npar, _, _ = fit_heads(Xtr, Ytr, Xte)
        b = bpe9(P, Yte, r0)
        results[name] = dict(trainable=npar, bpe=float(b))
        print(f'{name:24s} {npar:5d} tr | {b:+.3f} bits/event',
              flush=True)
        return P

    run('clock-only', Ctr, Cte)
    # own-voice traces: block-diagonal — emulate by 9 separate fits
    import torch
    tot_nll_gain = 0.0
    npar_own = 0
    eps = 1e-7
    for v in range(NV):
        Xtr = np.concatenate([Ctr, Ttr[:, v*10:(v+1)*10]], 1)
        Xte = np.concatenate([Cte, Tte[:, v*10:(v+1)*10]], 1)
        p, npv, _, _ = fit_heads(Xtr, Ytr[:, v:v+1], Xte)
        npar_own += npv
        y = Yte[:, v]
        nll = -(y*np.log2(p[:, 0]+eps) + (1-y)*np.log2(1-p[:, 0]+eps)).sum()
        nll0 = -(y*np.log2(r0[v]+eps) + (1-y)*np.log2(1-r0[v]+eps)).sum()
        tot_nll_gain += nll0 - nll
    b = tot_nll_gain / Yte.sum()
    results['clock+own-traces'] = dict(trainable=npar_own, bpe=float(b))
    print(f'{"clock+own-traces":24s} {npar_own:5d} tr | {b:+.3f} '
          f'bits/event', flush=True)
    run('clock+all-traces',
        np.concatenate([Ctr, Ttr], 1), np.concatenate([Cte, Tte], 1))
    # same-tick coupling (pseudolikelihood upper bound: other voices
    # at t given — diagnostic of coincidence info, not a causal model)
    Ytr_o = Ytr.copy(); Yte_o = Yte.copy()
    run('  +coupling (PL bound)',
        np.concatenate([Ctr, Ttr, Ytr_o], 1),
        np.concatenate([Cte, Tte, Yte_o], 1))
    (OUT / 'v3.json').write_text(json.dumps(results, indent=1,
                                            default=float))


def freerun():
    """Best causal rung (clock+all-traces): seed 2 bars, generate 16,
    measure rate drift / metrical R16 / voice-distribution drift."""
    import torch
    store = np.load(OUT / 'gmd9.npy', allow_pickle=True).item()
    Ctr, Ttr, Ytr = assemble('train', store)
    _, _, Yte = assemble('test', store)
    Xtr = np.concatenate([Ctr, Ttr], 1)
    _, npar, net, (mu, sd) = fit_heads(Xtr, Ytr, Xtr[:1])
    lam = np.array([0.5 ** (BIN / hl) for hl in TRACE_HL])
    rates, R16s, voices = [], [], []
    rng = np.random.default_rng(0)
    for it in store['test'][:40]:
        bpm, x = it['bpm'], it['x']
        bar_bins = int(4 * 60 / bpm / BIN)
        seed_n = 2 * bar_bins
        gen_n = 16 * bar_bins
        if len(x) < seed_n + 8:
            continue
        tr = np.zeros((NV, 10))
        for t in range(seed_n):                   # warm up on real seed
            tr = tr * lam
            tr += x[t][:, None]
        C = clock_feats(np.vstack([x[:seed_n],
                                   np.zeros((gen_n, NV), np.uint8)]),
                        bpm)
        gen = np.zeros((gen_n, NV), np.uint8)
        for t in range(gen_n):
            f = np.concatenate([C[seed_n + t], tr.reshape(-1)])
            f = (f - mu) / sd
            with torch.no_grad():
                p = torch.sigmoid(net(torch.from_numpy(
                    f.astype(np.float32)))).numpy()
            s = (rng.random(NV) < p).astype(np.uint8)
            gen[t] = s
            tr = tr * lam
            tr += s[:, None]
        r_bars = gen.any(1).reshape(16, bar_bins).mean(1)
        seed_rate = x[:seed_n].any(1).mean()
        rates.append(r_bars / max(seed_rate, 1e-6))
        on = np.where(gen.any(1))[0]
        th = 2 * np.pi * 4 * (on + seed_n) * BIN * bpm / 60
        R16s.append(np.abs(np.exp(1j * th).mean()) if len(on) else 0)
        vd_gen = gen.mean(0)
        vd_true = x.mean(0)
        m = 0.5 * (vd_gen + vd_true) + 1e-9
        js = 0.5 * ((vd_gen + 1e-9) * np.log2((vd_gen + 1e-9) / m) +
                    (vd_true + 1e-9) * np.log2((vd_true + 1e-9) / m)).sum()
        voices.append(js)
    rates = np.array(rates)
    print(f'free-run ({len(rates)} files, clock+all-traces, '
          f'{npar} params):')
    print(f'  rate ratio bar1 {np.median(rates[:, 0]):.2f} -> '
          f'bar16 {np.median(rates[:, -1]):.2f}')
    print(f'  metrical R16 of generated: median {np.median(R16s):.3f} '
          f'(human ~0.59)')
    print(f'  voice-distribution JS: median {np.median(voices):.3f} bits')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prep', action='store_true')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--freerun', action='store_true')
    a = ap.parse_args()
    if a.prep:
        prep()
    if a.sweep:
        sweep()
    if a.freerun:
        freerun()


if __name__ == '__main__':
    main()
