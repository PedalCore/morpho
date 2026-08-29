"""M10-P — generate from the piano spike-GLM (key+pc rung).

Seed 10 s of a held-out ARIA performance, free-run 20 s. Listen for
TONAL coherence (pc memory holding key/harmony), not melody: no
clock, no durations, no velocity model — recorded scope cuts.

python3 -m whitebox.piano_gen
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.piano_ladder import (BIN, HL, NK, OUT, P0,           # noqa
                                   feats)

SEED_S, GEN_S = 10.0, 20.0


def main():
    import torch
    store = np.load(OUT / 'aria88.npy', allow_pickle=True).item()
    kind = ('key', 'pc')
    Xtr = np.concatenate([feats(x, kind)[::2] for x in store['train']])
    Ytr = np.concatenate([x[::2] for x in store['train']]
                         ).astype(np.float32)
    r0 = np.clip(Ytr.mean(0), 1e-6, 1 - 1e-6)
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
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                net(Xt[ii]), Yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()

    seed_n, gen_n = int(SEED_S / BIN), int(GEN_S / BIN)
    # densest 10s window across test files (silent seeds tell nothing)
    best, bx, bo = None, None, -1
    for a in store['test']:
        if len(a) < seed_n:
            continue
        c = np.convolve(a.sum(1), np.ones(seed_n), 'valid')
        i = int(np.argmax(c))
        if c[i] > bo:
            bo, bx, best = c[i], a, i
    x = bx
    seed = x[best:best + seed_n]
    lam = np.array([0.5 ** (BIN / hl) for hl in HL])
    skey = np.zeros((len(HL), NK))
    spc = np.zeros((len(HL), 12))
    pcmap = np.array([(k + P0) % 12 for k in range(NK)])

    def push(y):
        pc = np.zeros(12)
        for k in np.where(y)[0]:
            pc[pcmap[k]] = 1
        for h in range(len(HL)):
            skey[h] = lam[h] * skey[h] + (1 - lam[h]) * y
            spc[h] = lam[h] * spc[h] + (1 - lam[h]) * pc

    for y in seed:
        push(y)
    rng = np.random.default_rng(5)
    gen = np.zeros((gen_n, NK), np.uint8)
    target = seed.sum() / len(seed)          # homeostatic rate target
    for t in range(gen_n):
        f = np.concatenate([skey.reshape(-1), spc.reshape(-1)])
        f = (f - mu) / sd
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(
                f.astype(np.float32)))).numpy()
        # divisive inhibition: hold expected onsets/bin at seed rate
        if p.sum() > target:
            p = p * (target / p.sum())
        y = (rng.random(NK) < p).astype(np.uint8)
        if y.sum() > 8:                      # polyphony cap: keep the
            keep = np.argsort(p)[-8:]        # 8 most probable
            y2 = np.zeros(NK, np.uint8)
            y2[[k for k in keep if y[k]]] = 1
            y = y2
        gen[t] = y
        push(y)
    full = np.vstack([seed, gen])
    print(f'seed {seed.sum()} onsets, generated {gen.sum()} onsets '
          f'({gen.sum()/GEN_S:.1f}/s vs seed {seed.sum()/SEED_S:.1f}/s)')
    # pitch-class distribution match (tonal coherence)
    def pcd(a):
        d = np.zeros(12)
        for k in range(NK):
            d[pcmap[k]] += a[:, k].sum()
        return d / max(d.sum(), 1)
    ps, pg = pcd(seed), pcd(gen)
    m = 0.5 * (ps + pg) + 1e-12
    js = 0.5 * ((ps + 1e-12) * np.log2((ps + 1e-12) / m) +
                (pg + 1e-12) * np.log2((pg + 1e-12) / m)).sum()
    print(f'pitch-class JS(seed, gen) = {js:.3f} bits '
          f'(0 = same key/harmony profile)')
    import mido
    mid = mido.MidiFile()
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tpb = mid.ticks_per_beat
    spb = 0.5
    ev = []
    for t in range(len(full)):
        for k in np.where(full[t])[0]:
            ev.append((t * BIN, 1, k + P0))
            ev.append((t * BIN + 0.25, 0, k + P0))
    ev.sort()
    prev = 0.0
    for tt, on, note in ev:
        dt = max(0, int(round((tt - prev) / spb * tpb)))
        prev = tt
        tr.append(mido.Message('note_on' if on else 'note_off',
                               note=note, velocity=72 if on else 0,
                               time=dt))
    mid.save(OUT / 'piano_gen.mid')
    print('rendered piano_gen.mid (10s seed + 20s generated)')


if __name__ == '__main__':
    main()
