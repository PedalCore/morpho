"""M10-P v2 — calibrated free-running piano GLM.

Fix 1 (saturation): SCHEDULED SAMPLING — parallel file-streams; with
prob eps the traces are updated from the model's own sampled output;
loss always vs true events. Calibration is learned, not imposed.
Fix 2 (randomness of pitch): VOICE-LEADING kernels — per-key
proximity to the last onset (3 shared params).
Eval: free-run WITHOUT inhibition — rate trajectory, pitch-class
JS, |dpitch| interval median vs human.

python3 -m whitebox.piano_v2          # train + eval + render .mid
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.piano_ladder import BIN, HL, NK, OUT, P0             # noqa

MAXB = 6000                       # truncate streams to 120 s
EPS = [0.0, 0.25, 0.5, 0.5]       # scheduled-sampling ramp per epoch


def hand_feats(last_key):
    """(B, NK, 3) proximity kernels to last-sounded key."""
    B = len(last_key)
    k = np.arange(NK)[None, :]
    d = np.abs(k - last_key[:, None]).astype(np.float64)
    out = np.stack([np.exp(-d / 3), np.exp(-d / 12),
                    (d == 0).astype(np.float64)], -1)
    out[last_key < 0] = 0.0
    return out.astype(np.float32)


def main():
    import torch
    store = np.load(OUT / 'aria88.npy', allow_pickle=True).item()
    train = [x[:MAXB] for x in store['train']]
    lam = np.array([0.5 ** (BIN / hl) for hl in HL])
    pcmap = np.array([(k + P0) % 12 for k in range(NK)])
    r0 = np.clip(np.concatenate(train).mean(0), 1e-6, 1 - 1e-6)

    torch.manual_seed(0)
    D = 5 * NK + 5 * 12
    shared = torch.nn.Linear(D, NK)
    wprox = torch.nn.Parameter(torch.zeros(3))
    with torch.no_grad():
        shared.bias.copy_(torch.from_numpy(
            np.log(r0 / (1 - r0)).astype(np.float32)))
    # feature scales for standardization: estimate from teacher pass
    opt = torch.optim.Adam(list(shared.parameters()) + [wprox],
                           lr=2e-3)
    Bs = 64
    mu = np.zeros(D, np.float32)
    sd = np.ones(D, np.float32)

    def run_epoch(eps, files, fit_norm=False):
        rng = np.random.default_rng(1)
        tot, nst = 0.0, 0
        feat_acc = [] if fit_norm else None
        for lo in range(0, len(files), Bs):
            batch = files[lo:lo + Bs]
            B = len(batch)
            L = max(len(x) for x in batch)
            skey = np.zeros((B, 5, NK))
            spc = np.zeros((B, 5, 12))
            last = np.full(B, -1)
            for t in range(L - 1):
                y_true = np.stack(
                    [x[t] if t < len(x) else np.zeros(NK, np.uint8)
                     for x in batch])
                y_next = np.stack(
                    [x[t + 1] if t + 1 < len(x)
                     else np.zeros(NK, np.uint8) for x in batch])
                alive = np.array([t + 1 < len(x) for x in batch])
                f = np.concatenate(
                    [skey.reshape(B, -1), spc.reshape(B, -1)],
                    1).astype(np.float32)
                if fit_norm:
                    if t % 25 == 0:
                        feat_acc.append(f.copy())
                    y_use = y_true
                else:
                    fN = (f - mu) / sd
                    hf = hand_feats(last)
                    ft = torch.from_numpy(fN)
                    z = shared(ft) + (torch.from_numpy(hf) *
                                      wprox[None, None]).sum(-1)
                    yt = torch.from_numpy(
                        y_next.astype(np.float32))
                    m = torch.from_numpy(alive.astype(np.float32))
                    loss = (torch.nn.functional.
                            binary_cross_entropy_with_logits(
                                z, yt, reduction='none')
                            .mean(-1) * m).sum() / max(m.sum(), 1)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += float(loss); nst += 1
                    with torch.no_grad():
                        p = torch.sigmoid(z).numpy()
                    samp = (rng.random((B, NK)) < p).astype(np.uint8)
                    use_model = rng.random(B) < eps
                    y_use = np.where(use_model[:, None], samp, y_true)
                # update states
                pcs = np.zeros((B, 12))
                for b in range(B):
                    on = np.where(y_use[b])[0]
                    if len(on):
                        last[b] = on[-1]
                        pcs[b, pcmap[on]] = 1
                for h in range(5):
                    skey[:, h] = lam[h] * skey[:, h] + \
                        (1 - lam[h]) * y_use
                    spc[:, h] = lam[h] * spc[:, h] + \
                        (1 - lam[h]) * pcs
        if fit_norm:
            F = np.concatenate(feat_acc)
            return F.mean(0), F.std(0) + 1e-6
        return tot / max(nst, 1)

    mu, sd = run_epoch(0.0, train[:32], fit_norm=True)
    for ep, eps in enumerate(EPS):
        bce = run_epoch(eps, train)
        print(f'epoch {ep+1} eps {eps}: BCE {bce:.4f}', flush=True)

    # ---- free-run WITHOUT inhibition ----
    tests = store['test']
    seed_n, gen_n = 500, 1000
    best, bx, bo = None, None, -1
    for a in tests:
        if len(a) < seed_n:
            continue
        c = np.convolve(a.sum(1), np.ones(seed_n), 'valid')
        i = int(np.argmax(c))
        if c[i] > bo:
            bo, bx, best = c[i], a, i
    seed = bx[best:best + seed_n]
    skey = np.zeros((1, 5, NK))
    spc = np.zeros((1, 5, 12))
    last = np.full(1, -1)
    rng = np.random.default_rng(9)

    def push(y):
        pcs = np.zeros((1, 12))
        on = np.where(y)[0]
        if len(on):
            last[0] = on[-1]
            pcs[0, pcmap[on]] = 1
        for h in range(5):
            skey[0, h] = lam[h] * skey[0, h] + (1 - lam[h]) * y
            spc[0, h] = lam[h] * spc[0, h] + (1 - lam[h]) * pcs[0]

    for y in seed:
        push(y)
    gen = np.zeros((gen_n, NK), np.uint8)
    for t in range(gen_n):
        f = np.concatenate([skey.reshape(1, -1),
                            spc.reshape(1, -1)], 1).astype(np.float32)
        fN = (f - mu) / sd
        with torch.no_grad():
            z = shared(torch.from_numpy(fN)) + \
                (torch.from_numpy(hand_feats(last)) *
                 wprox[None, None]).sum(-1)
            p = torch.sigmoid(z).numpy()[0]
        y = (rng.random(NK) < p).astype(np.uint8)
        gen[t] = y
        push(y)
    rate_bars = gen.sum(1).reshape(10, 100).sum(1)
    print(f'free-run rate/2s: {rate_bars.tolist()} '
          f'(seed {seed.sum()/10:.0f}/2s equivalent)', flush=True)

    def pcd(a):
        d = np.zeros(12)
        for k in range(NK):
            d[pcmap[k]] += a[:, k].sum()
        return d / max(d.sum(), 1)
    ps, pg = pcd(seed), pcd(gen)
    m = 0.5 * (ps + pg) + 1e-12
    js = 0.5 * ((ps + 1e-12) * np.log2((ps + 1e-12) / m) +
                (pg + 1e-12) * np.log2((pg + 1e-12) / m)).sum()

    def med_int(a):
        seq = [on for t in range(len(a))
               for on in np.where(a[t])[0]]
        d = np.abs(np.diff(seq))
        return float(np.median(d)) if len(d) else -1
    print(f'pc JS {js:.3f} | median |dpitch| gen '
          f'{med_int(gen):.0f} vs seed {med_int(seed):.0f} '
          f'| wprox {wprox.detach().numpy().round(2)}', flush=True)

    import mido
    full = np.vstack([seed, gen])
    mid = mido.MidiFile()
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tpb = mid.ticks_per_beat
    ev = []
    for t in range(len(full)):
        for k in np.where(full[t])[0]:
            ev.append((t * BIN, 1, k + P0))
            ev.append((t * BIN + 0.25, 0, k + P0))
    ev.sort()
    prev = 0.0
    for tt, on, note in ev:
        dt = max(0, int(round((tt - prev) / 0.5 * tpb)))
        prev = tt
        tr.append(mido.Message('note_on' if on else 'note_off',
                               note=note, velocity=72 if on else 0,
                               time=dt))
    mid.save(OUT / 'piano_v2.mid')
    print('rendered piano_v2.mid', flush=True)


if __name__ == '__main__':
    main()
