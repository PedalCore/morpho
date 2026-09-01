"""Replace the receptance gate with an event-driven one, at its measured budget.

Three measurements set this up. The bit-budget audit found the receptance
gate is the cheapest tensor in the model: 3 bits to stay within 1% of
float32 perplexity, and only +10% at ONE bit. The capacity experiment
found that sharp SELECTION - rank and take the top - reaches the same
associative capacity as amplitude-weighted retrieval. And the atlas found
the projections, not the recurrence, hold 99.8% of the arithmetic.

Put together they say: make the gate a selection event, and the sparsity
it creates should fall on a projection that actually costs something. In
RWKV the gate multiplies the wkv output immediately before self.output,

    return self.output(r * out)

so a zero in r is a column of that matmul that never has to happen. The
gate is therefore not just quantisable, it is skippable, and this measures
what the skipping costs.

Arms, from least to most aggressive:

  float           the trained model, reference
  quant b         gate rounded to b bits, dense (what the audit measured)
  binary t        gate is 0 or 1 either side of a threshold: an event
  top-p           keep only the largest p of channels per token, magnitudes
                  intact - amplitude-free SELECTION, the capacity result
  top-p binary    keep the largest p, and set them to 1: fully event-driven,
                  no multiply left in the gate path at all

Reported against measured sparsity, because the sparsity IS the saving:
a gate that is 80% zero removes 80% of the output projection's work.

NOTE ON WHAT THIS DOES NOT SHOW. This is post-hoc surgery on a model
trained with a dense float gate; the model never had a chance to adapt.
Every number here is therefore a FLOOR, and fine-tuning would recover
some of it. That is the honest next step, not this.

    python gate_event.py --ckpt runs/base-rwkv-d384L6-s42/ckpt.pt
"""

import argparse
import json
import types

import numpy as np
import torch

from spikelm.data import get_tokenizer, load_split
from spikelm.model import Config, RWKVMini, TimeMix, ChannelMix

STATE = {"mode": "float", "param": None, "nz": 0.0, "n": 0}


def apply_gate(r):
    """The gate transform under test. r is (B, T, C) in [0, 1]."""
    mode, p = STATE["mode"], STATE["param"]
    if mode == "float":
        out = r
    elif mode == "quant":
        n = 2 ** p - 1
        out = torch.round(r.clamp(0, 1) * n) / n
    elif mode == "binary":
        out = (r > p).to(r.dtype)
    elif mode in ("topp", "topp_binary", "topp_gain"):
        C = r.shape[-1]
        k = max(1, int(round(p * C)))
        thr = torch.topk(r, k, dim=-1).values[..., -1:]
        keep = (r >= thr).to(r.dtype)
        if mode == "topp":
            out = r * keep
        elif mode == "topp_binary":
            out = keep
        else:
            # Same selection, but the surviving channels carry the MEAN of
            # what they would have carried, so total gate mass is preserved.
            # Setting them to 1.0 instead inflates the gate, which is a gain
            # change masquerading as a sparsity result.
            m = (r * keep).sum(-1, keepdim=True) / keep.sum(-1, keepdim=True).clamp_min(1)
            out = keep * m
    elif mode == "binary_gain":
        keep = (r > p).to(r.dtype)
        m = (r * keep).sum(-1, keepdim=True) / keep.sum(-1, keepdim=True).clamp_min(1)
        out = keep * m
    else:
        raise ValueError(mode)
    STATE["nz"] += float((out != 0).sum())
    STATE["n"] += out.numel()
    return out


def timemix_forward(self, x):
    """model.py's TimeMix, with the gate routed through apply_gate."""
    xs = self.shift(x)
    k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
    v = self.value(x * self.mix_v + xs * (1 - self.mix_v))
    r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
    r = apply_gate(r)
    B, T, C = x.shape
    w = -torch.exp(self.time_decay)
    u = self.time_first
    aa = torch.zeros(B, C, device=x.device)
    bb = torch.zeros(B, C, device=x.device)
    pp = torch.full((B, C), -1e38, device=x.device)
    out = torch.empty(B, T, C, device=x.device)
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        ww = u + kt
        p_ = torch.maximum(pp, ww)
        e1, e2 = torch.exp(pp - p_), torch.exp(ww - p_)
        out[:, t] = (e1 * aa + e2 * vt) / (e1 * bb + e2 + 1e-9)
        ww2 = pp + w
        p2 = torch.maximum(ww2, kt)
        e1, e2 = torch.exp(ww2 - p2), torch.exp(kt - p2)
        aa, bb, pp = e1 * aa + e2 * vt, e1 * bb + e2, p2
    return self.output(r * out)


def channelmix_forward(self, x):
    """model.py's ChannelMix, with ITS gate routed through apply_gate too.

    This gate is the one that matters for arithmetic. It multiplies the
    output of value, an (4C -> C) projection, so a zero channel is a whole
    ROW of that matmul skipped — 4C^2 of work against TimeMix.output's C^2.
    Gating only the TimeMix side, as the first version of this file did,
    measures the small half.
    """
    if self.chanlif is not None:
        x = self.chanlif(x)
    xs = self.shift(x)
    kin = x * self.mix_k + xs * (1 - self.mix_k)
    if self.spike_in is not None:
        kin = self.spike_in(kin)
    k = self.key(kin)
    if self.spike_act is not None:
        k = self.spike_act(k)
    else:
        k = torch.square(torch.relu(k))
    r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
    return apply_gate(r) * self.value(k)


@torch.no_grad()
def run(model, batches, mode, param):
    import math
    STATE.update(mode=mode, param=param, nz=0.0, n=0)
    tot = ntok = 0
    for x, y in batches:
        _, loss = model(x, y)
        tot += float(loss) * y.numel()
        ntok += y.numel()
    density = STATE["nz"] / max(STATE["n"], 1)
    return math.exp(tot / ntok), density


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base-rwkv-d384L6-s42/ckpt.pt")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--out", default="gate-event.json")
    a = ap.parse_args()

    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size)
    model = RWKVMini(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval()
    for m in model.modules():
        if isinstance(m, TimeMix):
            m.forward = types.MethodType(timemix_forward, m)
        elif isinstance(m, ChannelMix):
            m.forward = types.MethodType(channelmix_forward, m)

    data = load_split("valid")
    rng = np.random.default_rng(7)
    B, T = 8, cfg.ctx
    nb = max(1, a.tokens // (B * T))
    batches = []
    for _ in range(nb):
        ix = rng.integers(0, len(data) - T - 1, size=B)
        batches.append((
            torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64)),
            torch.from_numpy(np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64))))

    ref, _ = run(model, batches, "float", None)
    print(f"{a.ckpt}\n{nb*B*T} tokens · float32 reference perplexity {ref:.4f}\n")

    arms = ([("quant", b, f"quantised, {b} bit") for b in (1, 2, 3, 4)] +
            [("binary", t, f"binary at r>{t}") for t in (0.3, 0.5, 0.7)] +
            [("topp", p, f"top-{int(p*100)}%, magnitudes") for p in (0.5, 0.25, 0.1, 0.05)] +
            [("topp_binary", p, f"top-{int(p*100)}%, binary (gain inflated)") for p in (0.5, 0.25, 0.1, 0.05)] +
            [("topp_gain", p, f"top-{int(p*100)}%, event + gain kept") for p in (0.5, 0.25, 0.1, 0.05)] +
            [("binary_gain", t, f"event at r>{t}, gain kept") for t in (0.3, 0.5, 0.7)])

    # both gates multiply a projection's OUTPUT, so a zero channel is a row
    # of that matmul skipped: TimeMix.output is C^2, ChannelMix.value is 4C^2,
    # against 13C^2 of matmul per block.
    GATEABLE = 5.0 / 13.0
    print(f"  {'gate':<28}{'ppl':>9}{'Δ%':>9}{'density':>10}{'model MACs saved':>18}")
    print("  " + "-" * 75)
    res = {"reference_ppl": ref, "arms": []}
    for mode, param, label in arms:
        ppl, dens = run(model, batches, mode, param)
        d = 100 * (ppl / ref - 1)
        saved = (1 - dens) * GATEABLE
        print(f"  {label:<28}{ppl:>9.3f}{d:>+9.1f}{dens:>9.1%}{saved:>17.1%}")
        res["arms"].append(dict(mode=mode, param=param, label=label,
                                ppl=ppl, delta_pct=d, density=dens,
                                model_macs_saved=saved))

    print("\n  the trade, read off the gain-preserving event arm:")
    for r_ in res["arms"]:
        if r_["mode"] == "topp_gain":
            print(f"    {r_['label']:<26} {r_['delta_pct']:+6.1f}% perplexity "
                  f"for {r_['model_macs_saved']:.1%} of ALL block matmuls skipped")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
