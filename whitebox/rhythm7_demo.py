"""M10 v6 demo — audible motif return via consolidated slot memory.

Arc: 4 real groove bars (slots consolidate the motif) -> 3 generated
-> 1 REAL FILL bar forced in -> 8 generated. Two arms from the same
trained readout: A = memory feature live (retrieval weighted by
consolidation depth g), B = memory zeroed (copy-prev only).
PREDICT: A reinstates the seeded groove after the fill; B drifts.

python3 -m whitebox.rhythm7_demo
Outputs: arcA_mem.mid / arcB_copy.mid + return-similarity numbers.
"""

import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.rhythm3_exact import (Config, GMD, NV, OUT,          # noqa
                                    clock_features, load_store,
                                    traces9, _trace_lam,
                                    _update_trace_state)
from whitebox.rhythm5_modes import bar_grids, beat_type_map         # noqa
from whitebox.rhythm6_motif import MotifStore, full_feats           # noqa

GM = [36, 38, 42, 46, 43, 47, 50, 49, 51]


def train_net():
    import torch
    cfg = Config()
    stored = load_store()
    Xs, Es, Ys = [], [], []
    for it in stored['train']:
        S, E = full_feats(it['x'], cfg)
        Xs.append(S)
        Es.append(E)
        Ys.append(it['x'])
    Xtr = np.concatenate(Xs)
    Etr = np.concatenate(Es)
    Ytr = np.concatenate(Ys).astype(np.float32)
    r0 = np.clip(Ytr.mean(0), 1e-5, 1 - 1e-5)
    torch.manual_seed(0)
    shared = torch.nn.Linear(Xtr.shape[1], NV)
    wE = torch.nn.Parameter(torch.zeros(NV, 2))
    with torch.no_grad():
        shared.bias.copy_(torch.from_numpy(
            np.log(r0 / (1 - r0)).astype(np.float32)))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.from_numpy((Xtr - mu) / sd)
    Et = torch.from_numpy(Etr)
    Yt = torch.from_numpy(Ytr)
    opt = torch.optim.Adam(list(shared.parameters()) + [wE], lr=3e-3)
    for ep in range(8):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 65536):
            ii = perm[i:i+65536]
            z = shared(Xt[ii]) + (Et[ii] * wE[None]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, Yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    return shared, wE, mu, sd


def consolidated_retrieve(store, cue):
    """Attractor depth weighting: cos * g/(g+2)."""
    n = np.linalg.norm(store.W, axis=1) * (np.linalg.norm(cue) + 1e-9)
    cos = (store.W @ cue) / (n + 1e-9)
    score = cos * (store.g / (store.g + 2.0))
    i = int(np.argmax(score))
    return store.W[i], i


def demo():
    import torch
    cfg = Config()
    spb, q = cfg.steps_per_bar, cfg.steps_per_sixteenth
    shared, wE, mu, sd = train_net()
    stored = load_store()
    bt = beat_type_map()
    groove = next(i for i in stored['test']
                  if bt.get(i['midi_filename']) == 'beat'
                  and len(i['x']) >= 6 * spb)
    fills = [i for i in stored['test']
             if bt.get(i['midi_filename']) == 'fill'
             and len(i['x']) >= spb]
    if not fills:
        fills = [i for s in ('validation', 'train')
                 for i in stored[s]
                 if bt.get(i['midi_filename']) == 'fill'
                 and len(i['x']) >= spb]
    fill = min(fills, key=lambda i: abs(i['bpm'] - groove['bpm']))
    print(f"groove: {groove['midi_filename']} ({groove['bpm']:.0f}) | "
          f"fill: {fill['midi_filename']} ({fill['bpm']:.0f})",
          flush=True)
    seed = groove['x'][:4 * spb]
    fill_bar = fill['x'][:spb]
    seed_grid = bar_grids(seed, cfg).mean(0)

    def run_arm(use_mem):
        rng = np.random.default_rng(11)
        lam = _trace_lam(cfg)
        tr = np.zeros((NV, cfg.traces_per_voice))
        store = MotifStore()
        prev = np.zeros(144)
        out = []
        bar_plan = (['seed'] * 4 + ['gen'] * 3 + ['fill'] +
                    ['gen'] * 8)
        step0 = 0
        for b, kind in enumerate(bar_plan):
            if kind == 'seed':
                bar = seed[b * spb:(b + 1) * spb]
            elif kind == 'fill':
                bar = fill_bar
            else:
                bar = np.zeros((spb, NV), np.uint8)
            rec = (consolidated_retrieve(store, prev)[0]
                   if b else np.zeros(144))
            C = clock_features(spb, cfg, start_step=step0)
            for t in range(spb):
                slot = (t // q) % 16
                if kind == 'gen':
                    e = np.zeros(2, np.float32)
                    e_mem = rec[slot * NV:(slot + 1) * NV]
                    e_cp = prev[slot * NV:(slot + 1) * NV]
                    f = np.concatenate([C[t], tr.reshape(-1)])
                    f = (f - mu) / sd
                    with torch.no_grad():
                        z = shared(torch.from_numpy(
                            f.astype(np.float32)))
                        extra = (torch.from_numpy(
                            np.stack([e_mem, e_cp], 1)
                            .astype(np.float32)) * wE).sum(-1)
                        if not use_mem:
                            extra = (torch.from_numpy(
                                np.stack([np.zeros(NV), e_cp], 1)
                                .astype(np.float32)) * wE).sum(-1)
                        p = torch.sigmoid(z + extra).numpy()
                    y = (rng.random(NV) < p).astype(np.uint8)
                    bar[t] = y
                _update_trace_state(tr, bar[t], lam)
            g = bar_grids(bar, cfg)
            if len(g):
                store.store(g[0])
                prev = g[0]
            out.append(bar.copy())
            step0 += spb
        return np.vstack(out)

    for name, use_mem in (('arcA_mem', True), ('arcB_copy', False)):
        full = run_arm(use_mem)
        gpost = bar_grids(full[8 * spb:], cfg)     # bars 9-16
        sim = [float(np.dot(g, seed_grid) /
                     (np.linalg.norm(g) * np.linalg.norm(seed_grid)
                      + 1e-9)) for g in gpost]
        print(f'{name}: post-fill motif similarity to seed groove '
              f'per bar: {np.round(sim, 2)} (mean {np.mean(sim):.2f})',
              flush=True)
        import mido
        mid = mido.MidiFile()
        trk = mido.MidiTrack()
        mid.tracks.append(trk)
        tpb = mid.ticks_per_beat
        st = tpb / cfg.steps_per_quarter
        trk.append(mido.MetaMessage(
            'set_tempo', tempo=mido.bpm2tempo(groove['bpm']), time=0))
        ev = []
        for t in range(len(full)):
            for v in np.where(full[t])[0]:
                ev.append((t * st, 1, GM[v], 96))
                ev.append((t * st + st, 0, GM[v], 0))
        ev.sort()
        prevt = 0.0
        for tt, on, note, vv in ev:
            dt = max(0, int(round(tt - prevt)))
            prevt = tt
            trk.append(mido.Message('note_on' if on else 'note_off',
                                    note=note, velocity=vv, channel=9,
                                    time=dt))
        mid.save(OUT / f'{name}.mid')
    print('rendered arcA_mem.mid / arcB_copy.mid '
          '(4 seed | 3 gen | 1 fill | 8 gen)', flush=True)


if __name__ == '__main__':
    demo()
