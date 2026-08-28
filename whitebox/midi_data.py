"""M9 — ARIA-MIDI event tokenization (Performance-RNN style, time
explicit). Vocab (219): NOTE_ON[88] NOTE_OFF[88] TIME_SHIFT[32 log
buckets 10ms..4s] VEL[8] BOS EOS PAD. Round-trips back to MIDI for
listenable continuations.

python3 -m whitebox.midi_data --src ~/aria --out ~/aria/tokens
        [--limit 20000]
Writes uint16 shards (tokens-XXX.npy, EOS-separated docs) + meta.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

P0, NP = 21, 88                    # piano range A0..C8
NOTE_ON, NOTE_OFF = 0, NP          # token bases
TS_BASE, N_TS = 2 * NP, 32
VEL_BASE, N_VEL = 2 * NP + N_TS, 8
BOS, EOS, PAD = VEL_BASE + N_VEL, VEL_BASE + N_VEL + 1, VEL_BASE + N_VEL + 2
VOCAB = PAD + 1                    # 219

_TS_EDGES = np.geomspace(0.01, 4.0, N_TS)


def ts_tokens(dt):
    """Greedy time-shift encoding of dt seconds."""
    out = []
    while dt > 0.005:
        b = int(np.searchsorted(_TS_EDGES, min(dt, 4.0), side='right') - 1)
        b = max(b, 0)
        out.append(TS_BASE + b)
        dt -= _TS_EDGES[b]
        if b == 0:
            break
    return out


def encode_file(path):
    import mido
    try:
        mid = mido.MidiFile(path)
    except Exception:
        return None
    events = []
    now = 0.0
    last_vel_b = -1
    toks = [BOS]
    for msg in mid:                      # iterates with .time in seconds
        now += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            events.append((now, 1, msg.note, msg.velocity))
        elif msg.type in ('note_off', 'note_on'):
            events.append((now, 0, msg.note, 0))
    prev = 0.0
    for t, on, note, vel in events:
        if not (P0 <= note < P0 + NP):
            continue
        toks.extend(ts_tokens(t - prev))
        prev = t
        if on:
            vb = min(vel * N_VEL // 128, N_VEL - 1)
            if vb != last_vel_b:
                toks.append(VEL_BASE + vb)
                last_vel_b = vb
            toks.append(NOTE_ON + note - P0)
        else:
            toks.append(NOTE_OFF + note - P0)
    toks.append(EOS)
    return toks if len(toks) > 64 else None


def decode_tokens(toks, out_path, tempo_bpm=120):
    """Tokens -> MIDI file (for listening to continuations)."""
    import mido
    mid = mido.MidiFile()
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tpb = mid.ticks_per_beat
    spb = 60.0 / tempo_bpm
    vel = 64
    pend = 0.0
    for t in toks:
        t = int(t)
        if TS_BASE <= t < TS_BASE + N_TS:
            pend += _TS_EDGES[t - TS_BASE]
        elif VEL_BASE <= t < VEL_BASE + N_VEL:
            vel = (t - VEL_BASE) * 16 + 8
        elif t < NP or NP <= t < 2 * NP:
            dt = int(round(pend / spb * tpb))
            pend = 0.0
            if t < NP:
                tr.append(mido.Message('note_on', note=t + P0,
                                       velocity=vel, time=dt))
            else:
                tr.append(mido.Message('note_off', note=t - NP + P0,
                                       velocity=0, time=dt))
    mid.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shard-tokens', type=int, default=50_000_000)
    args = ap.parse_args()
    src = pathlib.Path(args.src).expanduser()
    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob('*.mid')) + sorted(src.rglob('*.midi'))
    if args.limit:
        files = files[:args.limit]
    buf, shard, nfiles, ntok = [], 0, 0, 0
    for i, f in enumerate(files):
        toks = encode_file(f)
        if toks is None:
            continue
        buf.extend(toks)
        nfiles += 1
        ntok += len(toks)
        if len(buf) >= args.shard_tokens:
            np.save(out / f'tokens-{shard:03d}.npy',
                    np.array(buf, dtype=np.uint16))
            buf, shard = [], shard + 1
        if i % 2000 == 0:
            print(f'{i}/{len(files)} files, {ntok/1e6:.1f}M tokens',
                  flush=True)
    if buf:
        np.save(out / f'tokens-{shard:03d}.npy',
                np.array(buf, dtype=np.uint16))
    (out / 'meta.json').write_text(json.dumps(
        dict(vocab=VOCAB, files=nfiles, tokens=ntok, shards=shard + 1)))
    print(f'DONE {nfiles} files -> {ntok/1e6:.1f}M tokens, vocab {VOCAB}',
          flush=True)


if __name__ == '__main__':
    main()
