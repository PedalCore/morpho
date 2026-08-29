"""What precision does the detector that actually works need?

The recurrent experiment in mimii_experiment.py failed its own control: on
MIMII pump id_00 at 6 dB, the distance between a clip's average log-mel
spectrum and the average NORMAL spectrum scores 0.983 AUC with no model and
no training, and nothing we trained beat it. So the honest hardware question
is not "does ternary survive our model" — it is "does ternary survive the
detector that works".

That detector is a stored 64-value template, and scoring a clip is a dot
product against it. That is precisely the crossbar primitive the Morpho
circuit files build, which makes the question concrete:

    template mu     stored in the array          <- the weights
    clip mean x     streamed in                  <- the activations
    ||x - mu||^2    the score                    <- one MAC per mel bin

Measured gate costs per multiply-accumulate, from morpho_quantcompare.py:
exact 305, ternary 74, binary 9.5. So this asks what each of those buys.

    python mimii_template_quant.py --data pump_6dB_id_00.npz
"""

import argparse
import json

import numpy as np
import torch

from mimii_experiment import log_mel, auc

GATES = {"float32": 305, "int8": 305, "ternary": 74, "binary": 9.5}
BITS = {"float32": 32, "int8": 8, "ternary": 1.58, "binary": 1}


def quantize(v, mode):
    """Per-vector scale, along the last axis. The scale is one float kept
    outside the array; everything else is the low-precision part."""
    if mode == "float32":
        return v
    a = np.abs(v)
    if mode == "int8":
        s = a.max(-1, keepdims=True) / 127.0
        return np.round(v / s).clip(-127, 127) * s
    if mode == "ternary":                        # BitNet b1.58's rounding
        s = a.mean(-1, keepdims=True) / 0.7
        return np.round((v / s).clip(-1, 1)) * s
    if mode == "binary":
        return np.sign(v) * a.mean(-1, keepdims=True)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="template-quant.json")
    a = ap.parse_args()

    z = np.load(a.data)
    X, y, sr = z["X"], z["y"], int(z["sr"])
    if X.dtype == np.int16:
        X = X.astype(np.float32) / 32768.0
    F = torch.stack([log_mel(x, sr) for x in X]).numpy()
    print(f"{len(X)} clips, {int(y.sum())} abnormal · template is "
          f"{F.shape[-1]} numbers")

    modes = ["float32", "int8", "ternary", "binary"]
    res = {m: {w: [] for w in modes} for m in modes}

    for seed in range(a.seeds):
        idx = np.flatnonzero(y == 0)
        rng = np.random.default_rng(seed); rng.shuffle(idx)
        tr, ho = idx[:int(len(idx) * 0.67)], idx[int(len(idx) * 0.67):]
        test = np.concatenate([ho, np.flatnonzero(y == 1)])
        ty = np.concatenate([np.zeros(len(ho), int), np.ones(int(y.sum()), int)])

        # Normalisation statistics come from the TRAINING normals only. A global
        # shift is not harmless here: quantisation scales are data-derived, so a
        # scale cancels but an offset changes the sign pattern binary depends on.
        m, s = F[tr].mean(), F[tr].std()
        xbar = ((F - m) / s).mean(1)             # (clips, mels) — what streams in
        mu = xbar[tr].mean(0)                    # template from TRAIN normals only

        for wq in modes:
            muq = quantize(mu, wq)
            for aq in modes:
                Xq = quantize(xbar[test], aq)
                v = auc(((Xq - muq) ** 2).mean(1), ty)
                res[wq][aq].append(max(v, 1 - v))

    print(f"\nAUC, mean over {a.seeds} seeds "
          f"(rows: stored template · columns: streamed input)\n")
    print(f"  {'template':<10}" + "".join(f"{m:>10}" for m in modes)
          + f"{'gates/MAC':>12}{'template':>10}")
    for wq in modes:
        row = "".join(f"{np.mean(res[wq][m]):>10.3f}" for m in modes)
        print(f"  {wq:<10}{row}{GATES[wq]:>12}"
              f"{int(64 * BITS[wq]):>9}b")

    f32 = np.mean(res["float32"]["float32"])
    print(f"\n  float32 reference: {f32:.3f}")
    for wq in ("int8", "ternary", "binary"):
        d = np.mean(res[wq][wq]) - f32
        print(f"  {wq:<8} both sides: {np.mean(res[wq][wq]):.3f} ({d:+.3f})  "
              f"{GATES['float32']/GATES[wq]:.0f}x fewer gates, "
              f"{32/BITS[wq]:.0f}x smaller template")
    print("\n  (the trained recurrence, for comparison: 0.796)")

    json.dump({k: {m: v for m, v in r.items()} for k, r in res.items()},
              open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
