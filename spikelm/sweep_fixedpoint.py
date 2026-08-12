"""Fixed-point sweep: what the proposed wkv circuit costs a trained model.

Swaps the wkv recurrence for its fixed-point emulation (quantwkv) inside a
trained checkpoint and measures held-out perplexity, error accumulation over
the sequence, register saturation, and generated text — per numeric format.

python sweep_fixedpoint.py [--ckpt PATH] [--tokens N]
"""

import argparse
import json
import math

import numpy as np
import torch

from spikelm.data import get_tokenizer, load_split
from spikelm.model import Config, RWKVMini
from spikelm import quantwkv as qw
from spikelm.evaluate import generate, degeneration_metrics

ARMS = [
    dict(name="float64 reference", cfg=None),
    dict(name="Q8.8  LUT32 interp", cfg=qw.QConfig(8, 8, 32, True)),
    dict(name="Q8.8  LUT64 interp", cfg=qw.QConfig(8, 8, 64, True)),
    dict(name="Q8.8  LUT32 no-interp", cfg=qw.QConfig(8, 8, 32, False)),
    dict(name="Q8.8  exact exp", cfg=qw.QConfig(8, 8, 32, True, exact_exp=True)),
    dict(name="Q6.10 LUT32 interp", cfg=qw.QConfig(6, 10, 32, True)),
    dict(name="Q10.6 LUT32 interp", cfg=qw.QConfig(10, 6, 32, True)),
    dict(name="Q8.8  + restoring div", cfg=qw.QConfig(8, 8, 32, True, exact_div=False)),
]


def patch(model, cfg, stats_by_block, ref_by_block=None):
    """Replace every TimeMix wkv with the emulated circuit."""
    for bi, blk in enumerate(model.blocks):
        tm = blk.tm

        def fwd(x, tm=tm, bi=bi):
            xs = tm.shift(x)
            k = tm.key(x * tm.mix_k + xs * (1 - tm.mix_k)).double()
            v = tm.value(x * tm.mix_v + xs * (1 - tm.mix_v)).double()
            r = torch.sigmoid(tm.receptance(x * tm.mix_r + xs * (1 - tm.mix_r)))
            w = (-torch.exp(tm.time_decay)).double()
            u = tm.time_first.double()
            ref = ref_by_block[bi] if ref_by_block else None
            st = stats_by_block[bi] if stats_by_block else None
            out = qw.wkv(k, v, w, u, cfg, st, ref)
            if ref_by_block is not None and ref is None:
                ref_by_block[bi] = out
            return tm.output(r * out.to(x.dtype))

        tm.forward = fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base-rwkv-d384L6-s42/ckpt.pt")
    ap.add_argument("--tokens", type=int, default=4 * 8 * 256)
    args = ap.parse_args()

    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size)
    model = RWKVMini(cfg)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    data = load_split("valid")
    rng = np.random.default_rng(7)
    B, T = 8, cfg.ctx
    nb = max(1, args.tokens // (B * T))
    batches = []
    for _ in range(nb):
        ix = rng.integers(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64)
        batches.append((torch.from_numpy(x), torch.from_numpy(y)))
    print(f"model {args.ckpt}\n{nb} batches x {B}x{T} = {nb*B*T} tokens\n")

    results = []
    ref_cache = None
    for arm in ARMS:
        stats = [qw.Stats() for _ in range(cfg.n_layer)]
        patch(model, arm["cfg"], stats)
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for x, y in batches:
                logits, loss = model(x, y)
                tot += float(loss) * y.numel()
                ntok += y.numel()
        ppl = math.exp(tot / ntok)
        sat = sum(s.sat for s in stats)
        satn = sum(s.n for s in stats) or 1
        rec = dict(arm=arm["name"], ppl=round(ppl, 3),
                   sat_events=sat, sat_frac=round(sat / satn, 8))
        if arm["cfg"] is None:
            ref_cache = ppl
            rec["delta_pct"] = 0.0
        else:
            rec["delta_pct"] = round(100 * (ppl / ref_cache - 1), 2)
        results.append(rec)
        print(f"{arm['name']:24s} ppl {ppl:7.3f}  Δ{rec['delta_pct']:+6.2f}%  "
              f"saturations {sat}")

    # generation check on the leading candidate
    print("\n— generated text, Q8.8 LUT32 interp —")
    patch(model, qw.QConfig(8, 8, 32, True), None)
    txt = generate(model, tok, "Once upon a time, there was a little girl named",
                   max_new=120, device="cpu", seed=1000)
    print(txt[:300])
    m = degeneration_metrics(txt)
    print(f"rep4 {m['rep4']:.3f} distinct2 {m['distinct2']:.3f}")

    with open("runs/fixedpoint-sweep.json", "w") as f:
        json.dump(results, f, indent=1)
    print("\nsaved runs/fixedpoint-sweep.json")


if __name__ == "__main__":
    main()
