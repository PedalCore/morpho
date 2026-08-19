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
        if arr.ndim == 2 and arr.size > 4096 and "A_log" not in name:
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
    ap.add_argument("--out", default=None, help="subdirectory under export/")
    ap.add_argument("--levels", type=int, default=4, help="spike levels (1 = binary)")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = os.path.join(OUT, args.out)
    os.makedirs(OUT, exist_ok=True)
    tok = get_tokenizer()
    sd = torch.load(args.ckpt, map_location="cpu")["model"]
    mamba = any("A_log" in k for k in sd)
    spiking = any("spike_act.log_threshold" in k for k in sd)
    if mamba:
        from spikelm.mamba import MambaConfig, MambaMini

        cfg = MambaConfig(vocab_size=tok.vocab_size)
        model = MambaMini(cfg)
    else:
        cfg = Config(vocab_size=tok.vocab_size, spiking=spiking)
        model = RWKVMini(cfg)
    if spiking and args.levels != 4:
        from spikelm.spiking import SpikeAct

        for mod in model.modules():
            if isinstance(mod, SpikeAct):
                mod.set_levels(args.levels)
    model.load_state_dict(sd); model.eval()
    print(f"{'SPIKING' if spiking else 'float'} model, levels={args.levels if spiking else '-'}")
    data = load_split("valid")

    if mamba:   # no wkv decays to atlas; record the S6 shape instead
        doc = {"arch": "mamba-s6", "d_state": cfg.d_state, "d_conv": cfg.d_conv,
               "expand": cfg.expand, "d_inner": cfg.expand * cfg.n_embd,
               "dt_rank": max(1, cfg.n_embd // 16),
               "note": "A = -exp(A_log) < 0 and dt = softplus(..) > 0, so every "
                       "exponent argument dt*A is <= 0 — same property as wkv."}
        json.dump(doc, open(os.path.join(OUT, "s6-shape.json"), "w"), indent=1)
        blob, manifest = quantize_int8(sd)
        manifest["config"] = {"arch": "mamba", "n_layer": cfg.n_layer,
                              "n_embd": cfg.n_embd, "ctx": cfg.ctx,
                              "vocab_size": cfg.vocab_size, "d_state": cfg.d_state,
                              "d_conv": cfg.d_conv, "expand": cfg.expand,
                              "dt_rank": max(1, cfg.n_embd // 16)}
        if args.note:
            manifest["note"] = args.note
        pb = os.path.join(OUT, "model-int8.bin"); open(pb, "wb").write(blob)
        pm = os.path.join(OUT, "model-manifest.json")
        json.dump(manifest, open(pm, "w"), indent=1)
        tok.save(os.path.join(OUT, "tokenizer.json"))
        ppl_f = perplexity(model, data)
        model.load_state_dict(dequantized_state(sd, blob, manifest))
        ppl_q = perplexity(model, data)
        print(f"{pb} ({os.path.getsize(pb)/1e6:.1f} MB)")
        print(f"VALIDATION  float32 ppl {ppl_f:.3f}  ->  int8 ppl {ppl_q:.3f} "
              f"({100*(ppl_q/ppl_f-1):+.2f}%)")
        json.dump({"ppl_float32": round(ppl_f, 4), "ppl_int8": round(ppl_q, 4),
                   "delta_pct": round(100*(ppl_q/ppl_f-1), 3)},
                  open(os.path.join(OUT, "int8-validation.json"), "w"), indent=1)
        return

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
                          "ctx": cfg.ctx, "vocab_size": cfg.vocab_size,
                          "spiking": spiking, "levels": args.levels if spiking else 0}
    if spiking:
        thr = torch.exp(sd["blocks.0.cm.spike_act.log_threshold"])
        manifest["spike"] = {
            "threshold_uniform": bool(float(thr.min()) == float(thr.max())),
            "threshold": round(float(thr.mean()), 6),
            "rule": ("z = min(floor(x/thr), levels) * thr  (negatives -> 0); "
                     "thresholds did NOT train - grad is not routed to them - "
                     "so all 9216 are exactly their init value"),
            "consumer_matmul": "blocks.{L}.cm.value.weight (1536 -> 384)",
        }
    if args.note:
        manifest["note"] = args.note
    pb = os.path.join(OUT, "model-int8.bin")
    open(pb, "wb").write(blob)
    pm = os.path.join(OUT, "model-manifest.json")
    json.dump(manifest, open(pm, "w"), indent=1)
    # tokenizer travels with it
    tok.save(os.path.join(OUT, "tokenizer.json"))

    if spiking:
        from spikelm.spiking import SpikeAct

        rng = np.random.default_rng(11)
        ix = rng.integers(0, len(data) - cfg.ctx - 1, size=4)
        xb = torch.from_numpy(np.stack([data[i:i + cfg.ctx] for i in ix]).astype(np.int64))
        with torch.no_grad():
            model(xb)
        rates = [round(float(m.last_rate), 4) for m in model.modules()
                 if isinstance(m, SpikeAct)]
        manifest["spike"]["firing_rate_per_block"] = rates
        manifest["spike"]["firing_rate_mean"] = round(sum(rates) / len(rates), 4)
        json.dump(manifest, open(pm, "w"), indent=1)
        print(f"firing rates per block: {[f'{r*100:.0f}%' for r in rates]}  "
              f"mean {100*sum(rates)/len(rates):.1f}%")

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
