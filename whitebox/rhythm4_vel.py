"""M10 v4 — velocity stage on the tick-lattice kit process.

Given a spike (which the coupled binary model emits), predict HOW
HARD: per-voice linear regression on the same clock+trace features,
sigmoid output scaled to 1..127. Ghost notes become learnable.

python3 -m whitebox.rhythm4_vel --train --render
Renders whitebox/runs/rhythm/spikeglm_vel.mid from the soul-groove
seed using the causal coupling model + learned velocities.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.rhythm3_exact import (Config, GMD, NV, OUT, VOICE,   # noqa
                                    clock_features, fit_model,
                                    generate, load_store, traces9)

GM_NOTE = [36, 38, 42, 46, 43, 47, 50, 49, 51]


def midi_vel_raster(path, cfg):
    """Tick-lattice binary + velocity rasters for one file."""
    import mido
    mid = mido.MidiFile(path)
    merged = mido.merge_tracks(mid.tracks)
    abs_tick = 0
    ev = []
    for msg in merged:
        abs_tick += int(msg.time)
        if msg.type == 'note_on' and msg.velocity > 0 \
                and msg.note in VOICE:
            step = int(round(abs_tick * cfg.steps_per_quarter
                             / mid.ticks_per_beat))
            ev.append((step, VOICE[msg.note], msg.velocity))
    n = max(s for s, _, _ in ev) + 1
    x = np.zeros((n, NV), np.uint8)
    vel = np.zeros((n, NV), np.float32)
    for s, v, w in ev:
        x[s, v] = 1
        vel[s, v] = max(vel[s, v], w / 127.0)
    return x, vel


def collect_vel(split, cfg):
    """Feature/target rows at spike positions across a split."""
    import csv
    rows = [r for r in csv.DictReader(open(GMD / 'info.csv'))
            if r['split'] == split
            and r['time_signature'] in ('4-4', '4/4')]
    F, T, V = [], [], []
    for r in rows:
        try:
            x, vel = midi_vel_raster(GMD / r['midi_filename'], cfg)
        except Exception:
            continue
        if x.sum() < 8:
            continue
        C = clock_features(len(x), cfg)
        Tr = traces9(x, cfg)
        feats = np.concatenate([C, Tr], 1)
        for v in range(NV):
            on = np.where(x[:, v])[0]
            if not len(on):
                continue
            F.append(feats[on])
            T.append(np.full(len(on), v))
            V.append(vel[on, v])
    return (np.concatenate(F), np.concatenate(T),
            np.concatenate(V))


def train_vel(cfg):
    import torch
    Ftr, Ttr, Vtr = collect_vel('train', cfg)
    Fte, Tte, Vte = collect_vel('test', cfg)
    print(f'{len(Vtr)} train / {len(Vte)} test spikes', flush=True)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr = (Ftr - mu) / sd
    Fte = (Fte - mu) / sd
    d = Ftr.shape[1]
    W = torch.nn.Parameter(torch.zeros(NV, d))
    b = torch.nn.Parameter(torch.zeros(NV))
    with torch.no_grad():
        for v in range(NV):
            m = Vtr[Ttr == v].mean() if (Ttr == v).any() else 0.7
            b[v] = float(np.log(m / (1 - m + 1e-6)))
    opt = torch.optim.Adam([W, b], lr=3e-3)
    Ft = torch.from_numpy(Ftr)
    Tt = torch.from_numpy(Ttr)
    Vt = torch.from_numpy(Vtr)
    for ep in range(10):
        perm = torch.randperm(len(Ft))
        for i in range(0, len(Ft), 65536):
            ii = perm[i:i+65536]
            z = (Ft[ii] * W[Tt[ii]]).sum(-1) + b[Tt[ii]]
            loss = ((torch.sigmoid(z) - Vt[ii]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (torch.from_numpy(Fte) * W[torch.from_numpy(Tte)]).sum(-1) \
            + b[torch.from_numpy(Tte)]
        pred = torch.sigmoid(z).numpy()
    rmse = np.sqrt(np.mean((pred - Vte) ** 2)) * 127
    base = np.zeros_like(Vte)
    for v in range(NV):
        m = Vtr[Ttr == v].mean() if (Ttr == v).any() else 0.7
        base[Tte == v] = m
    rmse0 = np.sqrt(np.mean((base - Vte) ** 2)) * 127
    npar = W.numel() + b.numel()
    print(f'velocity head: {npar} params  RMSE {rmse:.1f} '
          f'(per-voice-mean baseline {rmse0:.1f}) MIDI units',
          flush=True)
    np.savez(OUT / 'velhead.npz', W=W.detach().numpy(),
             b=b.detach().numpy(), mu=mu, sd=sd,
             rmse=rmse, rmse0=rmse0)
    return rmse, rmse0


def render(cfg):
    import csv
    import torch
    store = load_store()
    model, _ = fit_model('coupling', store, cfg, epochs=6)
    vh = np.load(OUT / 'velhead.npz')
    rows = [r for r in csv.DictReader(open(GMD / 'info.csv'))
            if r['split'] == 'test' and r['beat_type'] == 'beat'
            and float(r['duration']) > 40
            and r['time_signature'] == '4-4']
    row = rows[0]
    bpm = float(row['bpm'])
    x, vel = midi_vel_raster(GMD / row['midi_filename'], cfg)
    seed_n = cfg.seed_bars * cfg.steps_per_bar
    rng = np.random.default_rng(7)
    gen = generate(model, x[:seed_n], cfg, rng)
    full = np.vstack([x[:seed_n], gen])
    C = clock_features(len(full), cfg)
    Tr = traces9(full, cfg)
    feats = (np.concatenate([C, Tr], 1) - vh['mu']) / vh['sd']
    gvel = np.zeros_like(full, np.float32)
    gvel[:seed_n] = vel[:seed_n]
    for t in range(seed_n, len(full)):
        for v in np.where(full[t])[0]:
            z = float(feats[t] @ vh['W'][v] + vh['b'][v])
            gvel[t, v] = 1.0 / (1.0 + np.exp(-z))
    # write MIDI: step duration from bpm
    import mido
    mid = mido.MidiFile()
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tpb = mid.ticks_per_beat
    step_ticks = tpb / cfg.steps_per_quarter
    events = []
    for t in range(len(full)):
        for v in np.where(full[t])[0]:
            vv = max(1, min(127, int(round(gvel[t, v] * 127))))
            events.append((t * step_ticks, 1, GM_NOTE[v], vv))
            events.append((t * step_ticks + step_ticks, 0,
                           GM_NOTE[v], 0))
    events.sort()
    tempo = mido.bpm2tempo(bpm)
    tr.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    prev = 0.0
    for tt, on, note, vv in events:
        dt = max(0, int(round(tt - prev)))
        prev = tt
        tr.append(mido.Message('note_on' if on else 'note_off',
                               note=note, velocity=vv, channel=9,
                               time=dt))
    mid.save(OUT / 'spikeglm_vel.mid')
    print(f'rendered spikeglm_vel.mid ({row["midi_filename"]}, '
          f'{bpm:.0f} bpm)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--render', action='store_true')
    a = ap.parse_args()
    cfg = Config()
    if a.train:
        train_vel(cfg)
    if a.render:
        render(cfg)


if __name__ == '__main__':
    main()
