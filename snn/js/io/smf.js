// Minimal Standard MIDI File parser — just enough to train an organism from
// a .mid file: note-on events with real-time stamps (tempo map respected).
// Format 0 and 1, running status, variable-length deltas.

export function parseSMF(buffer) {
  const bytes = new Uint8Array(buffer);
  let pos = 0;
  const readStr = (n) => {
    const s = String.fromCharCode(...bytes.slice(pos, pos + n));
    pos += n;
    return s;
  };
  const readU32 = () => {
    const v = (bytes[pos] << 24) | (bytes[pos + 1] << 16) | (bytes[pos + 2] << 8) | bytes[pos + 3];
    pos += 4;
    return v >>> 0;
  };
  const readU16 = () => {
    const v = (bytes[pos] << 8) | bytes[pos + 1];
    pos += 2;
    return v;
  };

  if (readStr(4) !== 'MThd') throw new Error('not a MIDI file');
  const headerLen = readU32();
  const format = readU16();
  const nTracks = readU16();
  const division = readU16();
  pos += headerLen - 6;
  if (division & 0x8000) throw new Error('SMPTE time division not supported');

  // collect (tick, note, vel) plus a tempo map of (tick, usPerBeat)
  const noteEvents = [];
  const tempoMap = [{ tick: 0, usPerBeat: 500000 }];

  for (let t = 0; t < nTracks; t++) {
    if (readStr(4) !== 'MTrk') throw new Error('bad track chunk');
    const len = readU32();
    const end = pos + len;
    let tick = 0;
    let running = 0;
    while (pos < end) {
      // variable-length delta
      let delta = 0;
      let b;
      do {
        b = bytes[pos++];
        delta = (delta << 7) | (b & 0x7f);
      } while (b & 0x80);
      tick += delta;

      let status = bytes[pos];
      if (status & 0x80) {
        pos++;
        if (status < 0xf0) running = status;
      } else {
        status = running;
      }

      if (status === 0xff) {
        const type = bytes[pos++];
        let mlen = 0;
        do {
          b = bytes[pos++];
          mlen = (mlen << 7) | (b & 0x7f);
        } while (b & 0x80);
        if (type === 0x51 && mlen === 3) {
          tempoMap.push({
            tick,
            usPerBeat: (bytes[pos] << 16) | (bytes[pos + 1] << 8) | bytes[pos + 2],
          });
        }
        pos += mlen;
      } else if (status === 0xf0 || status === 0xf7) {
        let slen = 0;
        do {
          b = bytes[pos++];
          slen = (slen << 7) | (b & 0x7f);
        } while (b & 0x80);
        pos += slen;
      } else {
        const kind = status & 0xf0;
        const d1 = bytes[pos++];
        const twoByte = kind !== 0xc0 && kind !== 0xd0;
        const d2 = twoByte ? bytes[pos++] : 0;
        if (kind === 0x90 && d2 > 0) noteEvents.push({ tick, note: d1, vel: d2 / 127 });
      }
    }
    pos = end;
  }

  // ticks → ms through the tempo map
  tempoMap.sort((a, b) => a.tick - b.tick);
  const toMs = (tick) => {
    let ms = 0;
    let lastTick = 0;
    let us = 500000;
    for (const seg of tempoMap) {
      if (seg.tick >= tick) break;
      ms += ((seg.tick - lastTick) * us) / division / 1000;
      lastTick = seg.tick;
      us = seg.usPerBeat;
    }
    return ms + ((tick - lastTick) * us) / division / 1000;
  };

  const notes = noteEvents
    .map((e) => ({ tMs: toMs(e.tick), note: e.note, vel: e.vel }))
    .sort((a, b) => a.tMs - b.tMs);
  return { format, nTracks, notes, durationMs: notes.length ? notes[notes.length - 1].tMs : 0 };
}
