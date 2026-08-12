"""Export trained weights for the hardware/visualisation track.

Produces two things:

  export/wkv-atlas.json     small (~200KB): the exact per-block decay and
                            bonus vectors, in both base-e and base-2 form,
                            plus summary stats. Enough to make a decay atlas
                            exact rather than illustrative.

  export/model-int8.bin     the whole model, per-row symmetric int8 (~13MB)
  export/model-manifest.json  shapes, scales, dequant recipe

The int8 export is VALIDATED here — held-out perplexity is measured with the
quantized weights loaded back, so nothing ships untested.

python export_weights.py [--ckpt PATH]
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from spikelm.data import get_tokenizer, load_split
from spikelm.model import Config, RWKVMini

LOG2E = math.log2(math.e)
OUT = os.path.join(os.path.dirname(__file__), "export")


def atlas(sd, n_layer):
    doc = {"note": "w = -exp(time_decay) is the per-step decay in base e; "
                   "w2 = w*log2(e) is the base-2 form the circuit uses "
                   "(2^x instead of e^x). u/u2 are the 'bonus' applied to the "
                   "current token. Memory half-life in tokens = 1/|w2|.",
           "log2e": LOG2E, "blocks": []}
    for b in range(n_layer):
        td = sd[f"blocks.{b}.tm.time_decay"].double()
        tf = sd[f"blocks.{b}.tm.time_first"].double()
        w = -torch.exp(td)
        w2 = w * LOG2E
        half = (1.0 / w2.abs())
        doc["blocks"].append({
            "block": b,
            "w": [round(float(x), 6) for x in w],
            "w2": [round(float(x), 6) for x in w2],
            "u": [round(float(x), 6) for x in tf],
            "u2": [round(float(x), 6) for x in (tf * LOG2E)],
            "stats": {
                "w2_min": round(float(w2.min()), 4),
                "w2_max": round(float(w2.max()), 6),
                "w2_median": round(float(w2.median()), 4),
                "halflife_tokens_median": round(float(half.median()), 2),
                "halflife_tokens_max": round(float(half.max()), 1),
                "channels_slower_than_100_tokens": int((w2.abs() < 0.01).sum()),
            },
        })
    return doc


def quantize_int8(sd):
    """Per-row symmetric int8 for 2-D tensors; small vectors stay float32."""
    blobs, manifest = [], {"tensors": [], "dtype_small": "float32",
                           "recipe": "w = int8_row * scale_row"}
    offset = 0
    for name, t in sd.items():
        arr = t.detach().cpu().numpy()
        if arr.ndim == 2 and arr.size > 4096:
            scale = np.abs(arr).max(axis=1, keepdims=True) / 127.0
            scale[scale == 0] = 1e-8
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
            blobs.append(q.tobytes())
            s = scale.astype(np.float32).tobytes()
            manifest["tensors"].append(
                {"name": name, "shape": list(arr.shape), "kind": "int8",
                 "offset": offset, "nbytes": len(blobs[-1]),
                 "scale_offset": offset + len(blobs[-1]), "scale_nbytes": len(s)})
            blobs.append(s)
            offset += len(blobs[-2]) + len(s)
        else:
            b = arr.astype(np.float32).tobytes()
            blobs.append(b)
            manifest["tensors"].append(
                {"name": name, "shape": list(arr.shape), "kind": "float32",
                 "offset": offset, "nbytes": len(b)})
            offset += len(b)
    return b"".join(blobs), manifest


def dequantized_state(sd, blob, manifest):
    out = {}
    for t in manifest["tensors"]:
        if t["kind"] == "int8":
            q = np.frombuffer(blob, np.int8, count=t["nbytes"], offset=t["offset"])
            s = np.frombuffer(blob, np.float32, count=t["scale_nbytes"] // 4,
                              offset=t["scale_offset"])
            a = q.reshape(t["shape"]).astype(np.float32) * s.reshape(-1, 1)
        else:
            a = np.frombuffer(blob, np.float32,
                              count=t["nbytes"] // 4, offset=t["offset"]).reshape(t["shape"])
        out[t["name"]] = torch.from_numpy(a.copy())
    return out


def perplexity(model, data, batches=6, B=8, T=256, seed=7):
    rng = np.random.default_rng(seed)
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(batches):
            ix = rng.integers(0, len(data) - T - 1, size=B)
            x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64))
            y = torch.from_numpy(np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64))
            _, loss = model(x, y)
            tot += float(loss) * y.numel(); n += y.numel()
    return math.exp(tot / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base-rwkv-d384L6-s42/ckpt.pt")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size)
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    model = RWKVMini(cfg); model.load_state_dict(sd); model.eval()
    data = load_split("valid")

    # 1. the atlas
    doc = atlas(sd, cfg.n_layer)
    p = os.path.join(OUT, "wkv-atlas.json")
    json.dump(doc, open(p, "w"))
    print(f"{p}  ({os.path.getsize(p)/1024:.0f} KB)")
    for b in doc["blocks"]:
        s = b["stats"]
        print(f"   block {b['block']}: w2 {s['w2_min']} .. {s['w2_max']}  "
              f"median half-life {s['halflife_tokens_median']} tokens  "
              f"(slowest {s['halflife_tokens_max']})")

    # 2. int8 model + validation
    blob, manifest = quantize_int8(sd)
    manifest["config"] = {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd,
                          "ctx": cfg.ctx, "vocab_size": cfg.vocab_size}
    pb = os.path.join(OUT, "model-int8.bin")
    open(pb, "wb").write(blob)
    pm = os.path.join(OUT, "model-manifest.json")
    json.dump(manifest, open(pm, "w"), indent=1)
    # tokenizer travels with it
    tok.save(os.path.join(OUT, "tokenizer.json"))

    ppl_f = perplexity(model, data)
    model.load_state_dict(dequantized_state(sd, blob, manifest))
    ppl_q = perplexity(model, data)
    print(f"\n{pb}  ({os.path.getsize(pb)/1e6:.1f} MB)")
    print(f"{pm}  ({os.path.getsize(pm)/1024:.0f} KB)")
    print(f"\nVALIDATION  float32 ppl {ppl_f:.3f}  →  int8 ppl {ppl_q:.3f}  "
          f"({100*(ppl_q/ppl_f-1):+.2f}%)")
    json.dump({"ppl_float32": round(ppl_f, 4), "ppl_int8": round(ppl_q, 4),
               "delta_pct": round(100 * (ppl_q / ppl_f - 1), 3)},
              open(os.path.join(OUT, "int8-validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
