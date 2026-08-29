"""Fetch a slice of the MIMII dataset without downloading the whole archive.

The MIMII zips are 7-11 GB each and we need a few hundred files, so this
reads the zip's central directory over HTTP range requests and pulls only
the entries it wants. Files are converted to mono float32 and cached as a
single .npz, which is what the experiment actually consumes.

    python mimii_fetch.py [--machine pump] [--snr 6] [--id id_00]
                          [--normal 300] [--abnormal 120]

MIMII is CC-BY-SA 4.0 (Purohit et al., 2019, arXiv:1909.09347).
"""

import argparse
import io
import json
import os
import struct
import sys
import urllib.request
import wave
import zlib

import numpy as np

ZENODO = "https://zenodo.org/api/records/3384388"
CACHE = os.path.join(os.path.dirname(__file__), "mimii-cache")


def http(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{'' if end is None else end}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def file_url(name):
    with urllib.request.urlopen(ZENODO) as r:
        rec = json.load(r)
    for f in rec["files"]:
        if f["key"] == name:
            return f["links"]["self"], f["size"]
    raise SystemExit(f"{name} not in the Zenodo record")


def zip64_extra(extra, csize, usize, lho):
    """Pull 64-bit sizes/offsets out of a central-directory extra field.

    Fields appear in a fixed order (uncompressed, compressed, offset) but
    ONLY for those whose 32-bit slot was saturated, so which ones are
    present depends on the values already read.
    """
    p = 0
    while p + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[p:p + 4])
        body, p = extra[p + 4:p + 4 + hsz], p + 4 + hsz
        if hid != 0x0001:
            continue
        q = 0

        def take(current):
            nonlocal q
            if current != 0xFFFFFFFF or q + 8 > len(body):
                return current
            val, = struct.unpack("<Q", body[q:q + 8])
            q += 8
            return val

        usize, csize, lho = take(usize), take(csize), take(lho)
        break
    return csize, usize, lho


def central_directory(url, size):
    """Parse the zip's index from its tail — no need to fetch the body.

    The MIMII archives are over 4 GB, so they are Zip64: the legacy
    end-of-central-directory record holds 0xFFFFFFFF placeholders and the
    real 64-bit offsets live in a separate record found via a locator that
    sits just before it. Reading only the legacy record finds zero entries.
    """
    tail = http(url, size - 65536, size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise SystemExit("end-of-central-directory record not found")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])

    loc = tail.rfind(b"PK\x06\x07", 0, i)
    if loc >= 0 or cd_off == 0xFFFFFFFF:
        z64_off, = struct.unpack("<Q", tail[loc + 8:loc + 16])
        z64 = http(url, z64_off, z64_off + 55)
        if z64[:4] != b"PK\x06\x06":
            raise SystemExit("Zip64 end-of-central-directory record not found")
        cd_size, cd_off = struct.unpack("<QQ", z64[40:56])

    cd = http(url, cd_off, cd_off + cd_size - 1)
    entries, p = {}, 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        lho, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if 0xFFFFFFFF in (csize, usize, lho):
            csize, usize, lho = zip64_extra(extra, csize, usize, lho)
        entries[name] = (lho, csize, usize, method)
        p += 46 + nlen + elen + clen
    return entries


def fetch_member(url, entry):
    """Range-fetch one file out of the archive and inflate it."""
    lho, csize, usize, method = entry
    head = http(url, lho, lho + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = lho + 30 + nlen + elen
    blob = http(url, start, start + csize - 1)
    if method == 0:
        return blob
    return zlib.decompressobj(-zlib.MAX_WBITS).decompress(blob, usize)


def wav_mono(raw):
    """Average the 8 microphone channels into one signal.

    Parsed by hand rather than with the `wave` module: MIMII's files are
    WAVE_FORMAT_EXTENSIBLE (tag 65534), which `wave` refuses outright.
    """
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE stream")
    p, fmt, data = 12, None, None
    while p + 8 <= len(raw):
        cid, sz = raw[p:p + 4], struct.unpack("<I", raw[p + 4:p + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", raw[p + 8:p + 24])
        elif cid == b"data":
            data = raw[p + 8:p + 8 + sz]
        p += 8 + sz + (sz & 1)                   # chunks are word-aligned
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    ch, sr, bits = fmt[1], fmt[2], fmt[5]
    if bits != 16:
        raise ValueError(f"expected 16-bit samples, got {bits}")
    n = len(data) // (2 * ch)
    x = np.frombuffer(data[:n * 2 * ch], "<i2").reshape(n, ch)
    return x.astype(np.float32).mean(1) / 32768.0, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default="pump",
                    choices=["pump", "valve", "fan", "slider"])
    ap.add_argument("--snr", default="6", choices=["6", "0", "-6"])
    ap.add_argument("--id", default="id_00")
    ap.add_argument("--normal", type=int, default=300)
    ap.add_argument("--abnormal", type=int, default=120)
    a = ap.parse_args()

    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f"{a.machine}_{a.snr}dB_{a.id}.npz")
    if os.path.exists(out):
        print(f"already have {out}")
        return

    zipname = f"{a.snr}_dB_{a.machine}.zip"
    url, size = file_url(zipname)
    print(f"{zipname}: {size/1e9:.2f} GB in the archive — reading its index only")
    entries = central_directory(url, size)

    def pick(kind, want):
        names = sorted(n for n in entries
                       if f"/{a.id}/{kind}/" in n and n.endswith(".wav"))
        print(f"  {kind:8} available {len(names):5}  taking {min(want, len(names))}")
        return names[:want]

    wanted = [(n, 0) for n in pick("normal", a.normal)] + \
             [(n, 1) for n in pick("abnormal", a.abnormal)]
    if not wanted:
        raise SystemExit(f"no files matched {a.id} in {zipname}")

    xs, ys, sr = [], [], None
    total = 0
    for i, (name, label) in enumerate(wanted):
        raw = fetch_member(url, entries[name])
        total += entries[name][1]
        x, sr = wav_mono(raw)
        xs.append(x.astype(np.float32)); ys.append(label)
        if (i + 1) % 25 == 0 or i + 1 == len(wanted):
            print(f"  {i+1:4}/{len(wanted)}  {total/1e6:7.1f} MB fetched",
                  flush=True)

    n = min(len(x) for x in xs)
    X = np.stack([x[:n] for x in xs])
    np.savez_compressed(out, X=X, y=np.array(ys, np.int64), sr=sr)
    print(f"\nwrote {out}")
    print(f"  {X.shape[0]} clips x {X.shape[1]/sr:.1f}s @ {sr} Hz  "
          f"({int(np.sum(ys))} abnormal)  — {total/1e6:.0f} MB downloaded "
          f"instead of {size/1e9:.1f} GB")


if __name__ == "__main__":
    main()
