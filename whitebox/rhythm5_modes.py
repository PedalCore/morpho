"""M10 v5a — navigable mode space: switching spike-GLM.

python3 -m whitebox.rhythm5_modes --cluster   # modes + label check
python3 -m whitebox.rhythm5_modes --fit       # mode-conditioned b/ev
python3 -m whitebox.rhythm5_modes --navigate  # 7-groove/1-fill demo
"""

import argparse
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.rhythm3_exact import (Config, GMD, NV, OUT,          # noqa
                                    clock_features, load_store,
                                    traces9, _trace_lam,
                                    _update_trace_state)

K = 12


def bar_grids(x, cfg):
    """(bars, 144) slot-occupancy grids: 16 sixteenth slots x 9."""
    spb, q = cfg.steps_per_bar, cfg.steps_per_sixteenth
    nb = len(x) // spb
    if nb == 0:
        return np.zeros((0, 144), np.float32)
    b = x[:nb * spb].reshape(nb, 16, q, NV).max(2)     # (nb, 16, 9)
    return b.reshape(nb, 144).astype(np.float32)


def beat_type_map():
    return {r['midi_filename']: r['beat_type']
            for r in csv.DictReader(open(GMD / 'info.csv'))}


def cluster():
    from scipy.cluster.vq import kmeans2
    store = load_store()
    bt = beat_type_map()
    G, lab = [], []
    for it in store['train']:
        g = bar_grids(it['x'], Config())
        G.append(g)
        lab += [bt.get(it['midi_filename'], '?')] * len(g)
    G = np.concatenate(G)
    lab = np.array(lab)
    np.random.seed(0)
    cent, asn = kmeans2(G, K, minit='++', seed=0)
    print(f'{len(G)} bars clustered into {K} modes')
    fill_frac = []
    for k in range(K):
        m = asn == k
        ff = (lab[m] == 'fill').mean() if m.sum() else 0
        fill_frac.append(ff)
        dens = G[m].sum(1).mean() if m.sum() else 0
        print(f'mode {k:2d}: {m.sum():5d} bars  fill-frac {ff:.2f}  '
              f'density {dens:.1f}')
    np.savez(OUT / 'modes.npz', cent=cent, fill_frac=fill_frac)
    base = (lab == 'fill').mean()
    print(f'base fill-frac {base:.2f}; max cluster fill-frac '
          f'{max(fill_frac):.2f}')


def assign(g, cent):
    d = ((g[:, None, :] - cent[None]) ** 2).sum(-1)
    return d.argmin(1)


def mode_feats(x, cfg, cent):
    spb = cfg.steps_per_bar
    g = bar_grids(x, cfg)
    asn = assign(g, cent) if len(g) else np.zeros(0, int)
    M = np.zeros((len(x), K), np.float32)
    for b, k in enumerate(asn):
        M[b * spb:(b + 1) * spb, k] = 1.0
    if len(g):
        M[len(g) * spb:, asn[-1]] = 1.0
    return M


def fit(with_modes=True):
    import torch
    cfg = Config()
    cent = np.load(OUT / 'modes.npz')['cent']
    store = load_store()

    def feats(x):
        F = [clock_features(len(x), cfg), traces9(x, cfg)]
        if with_modes:
            F.append(mode_feats(x, cfg, cent))
        return np.concatenate(F, 1)

    Xtr = np.concatenate([feats(it['x']) for it in store['train']])
    Ytr = np.concatenate([it['x'] for it in store['train']]
                         ).astype(np.float32)
    Xte = np.concatenate([feats(it['x']) for it in store['test']])
    Yte = np.concatenate([it['x'] for it in store['test']]
                         ).astype(np.float32)
    r0 = np.clip(Ytr.mean(0), 1e-5, 1 - 1e-5)
    torch.manual_seed(0)
    net = torch.nn.Linear(Xtr.shape[1], NV)
    with torch.no_grad():
        net.bias.copy_(torch.from_numpy(
            np.log(r0 / (1 - r0)).astype(np.float32)))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.from_numpy((Xtr - mu) / sd)
    Yt = torch.from_numpy(Ytr)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for ep in range(8):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 65536):
            ii = perm[i:i+65536]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(Xt[ii]), Yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.sigmoid(net(torch.from_numpy(
            (Xte - mu) / sd))).numpy()
    eps = 1e-7
    nll = -(Yte * np.log2(P + eps) +
            (1 - Yte) * np.log2(1 - P + eps)).sum()
    P0 = np.broadcast_to(r0, Yte.shape)
    nll0 = -(Yte * np.log2(P0 + eps) +
             (1 - Yte) * np.log2(1 - P0 + eps)).sum()
    b = (nll0 - nll) / Yte.sum()
    tag = 'clock+traces+MODE' if with_modes else 'clock+traces'
    print(f'{tag}: {sum(p.numel() for p in net.parameters())} params '
          f'{b:+.3f} b/ev')
    if with_modes:
        import pickle
        torch.save(net.state_dict(), OUT / 'modeglm.pt')
        np.savez(OUT / 'modeglm_norm.npz', mu=mu, sd=sd)
    return b


def navigate():
    """Render a user-chosen mode path: groove x7, fill, x2."""
    import torch
    cfg = Config()
    z = np.load(OUT / 'modes.npz')
    cent, ff = z['cent'], z['fill_frac']
    fill_mode = int(np.argmax(ff))
    store = load_store()
    bt = beat_type_map()
    it = next(i for i in store['test']
              if bt.get(i['midi_filename']) == 'beat'
              and len(i['x']) >= 4 * cfg.steps_per_bar)
    seed = it['x'][:2 * cfg.steps_per_bar]
    groove_mode = int(assign(bar_grids(seed, cfg), cent)[0])
    print(f'groove mode {groove_mode}, fill mode {fill_mode} '
          f'(fill-frac {ff[fill_mode]:.2f}); seed '
          f'{it["midi_filename"]}')
    path = [groove_mode]*7 + [fill_mode] + [groove_mode]*7 + [fill_mode]
    d = 28 + 90 + K
    net = torch.nn.Linear(d, NV)
    net.load_state_dict(torch.load(OUT / 'modeglm.pt'))
    nz = np.load(OUT / 'modeglm_norm.npz')
    mu, sd = nz['mu'], nz['sd']
    lam = _trace_lam(cfg)
    tr = np.zeros((NV, cfg.traces_per_voice))
    for y in seed:
        _update_trace_state(tr, y, lam)
    n = len(path) * cfg.steps_per_bar
    C = clock_features(n, cfg, start_step=len(seed))
    rng = np.random.default_rng(3)
    gen = np.zeros((n, NV), np.uint8)
    for t in range(n):
        mode1h = np.zeros(K, np.float32)
        mode1h[path[t // cfg.steps_per_bar]] = 1
        f = np.concatenate([C[t], tr.reshape(-1), mode1h])
        f = (f - mu) / sd
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(
                f.astype(np.float32)))).numpy()
        y = (rng.random(NV) < p).astype(np.uint8)
        gen[t] = y
        _update_trace_state(tr, y, lam)
    full = np.vstack([seed, gen])
    # write MIDI at the take's bpm
    import mido
    GM = [36, 38, 42, 46, 43, 47, 50, 49, 51]
    mid = mido.MidiFile()
    trk = mido.MidiTrack()
    mid.tracks.append(trk)
    tpb = mid.ticks_per_beat
    st = tpb / cfg.steps_per_quarter
    trk.append(mido.MetaMessage(
        'set_tempo', tempo=mido.bpm2tempo(it['bpm']), time=0))
    events = []
    for t in range(len(full)):
        for v in np.where(full[t])[0]:
            events.append((t * st, 1, GM[v], 96))
            events.append((t * st + st, 0, GM[v], 0))
    events.sort()
    prev = 0.0
    for tt, on, note, vv in events:
        dt = max(0, int(round(tt - prev)))
        prev = tt
        trk.append(mido.Message('note_on' if on else 'note_off',
                                note=note, velocity=vv, channel=9,
                                time=dt))
    mid.save(OUT / 'navigate.mid')
    print('rendered navigate.mid (7 groove / 1 fill / repeat)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', action='store_true')
    ap.add_argument('--fit', action='store_true')
    ap.add_argument('--navigate', action='store_true')
    a = ap.parse_args()
    if a.cluster:
        cluster()
    if a.fit:
        fit(with_modes=False)
        fit(with_modes=True)
    if a.navigate:
        navigate()


if __name__ == '__main__':
    main()
