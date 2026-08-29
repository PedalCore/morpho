"""MIMII anomaly detection with a streaming recurrent model — and what
ternary weights cost it.

The question this answers is the one TinyStories cannot: our hardware work
says a weight restricted to {-1, 0, +1} turns a 305-gate multiplier into a
74-gate add/subtract. Does a model still work when you do that, on a task
someone would actually deploy?

Setup, chosen to exercise what the architecture is good at:

  * MIMII industrial machine sound (Purohit et al. 2019), pump at +6 dB SNR
  * log-mel frames, streamed one at a time — never a fixed window
  * a diagonal recurrence (the RWKV/S6 primitive) with CONSTANT state,
    trained only on NORMAL sound to predict its next frame
  * anomaly score = how badly it predicts; a machine that starts leaking
    stops sounding like what the model learned
  * scored by ROC AUC, normal vs abnormal, which is the metric the MIMII
    baseline and the DCASE challenges report

Then the same model with its weights forced to {-1, 0, +1}, scored again.

    python mimii_experiment.py --data mimii-cache/pump_6dB_id_00.npz
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- features

def log_mel(x, sr, n_fft=1024, hop=512, n_mels=64):
    """Streaming-friendly frames: no lookahead beyond one window."""
    win = torch.hann_window(n_fft)
    spec = torch.stft(torch.as_tensor(x, dtype=torch.float32), n_fft, hop,
                      window=win, return_complex=True).abs() ** 2
    # mel filterbank, built once
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    edges = mel2hz(np.linspace(hz2mel(50), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(n_mels):
        l, c, r = bins[m], bins[m + 1], bins[m + 2]
        if c > l: fb[m, l:c] = np.linspace(0, 1, c - l)
        if r > c: fb[m, c:r] = np.linspace(1, 0, r - c)
    mel = torch.as_tensor(fb) @ spec
    return torch.log(mel + 1e-6).T                      # (frames, n_mels)


# ------------------------------------------------------------------- model

class Recur(nn.Module):
    """The same primitive the circuit work describes: h <- a*h + b*u, with a
    a per-channel constant. One frame in, next frame predicted, constant
    state — no window, no growing cache."""

    def __init__(self, n_in, d=64):
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.decay_raw = nn.Parameter(torch.linspace(-3, 3, d))
        self.out = nn.Linear(d, n_in)
        self.d = d

    def forward(self, x):                                # (B, T, n_in)
        u = self.inp(x)
        a = torch.sigmoid(self.decay_raw)
        h = torch.zeros(x.shape[0], self.d, device=x.device)
        ys = []
        for t in range(x.shape[1]):
            h = a * h + u[:, t]
            ys.append(self.out(h))
        return torch.stack(ys, 1)                        # prediction of frame t+1


def ternarize(model, keep=0.7):
    """Weights -> {-1, 0, +1} x a per-row scale. BitNet b1.58's arithmetic:
    the multiplier becomes an add, a subtract, or nothing at all."""
    tm = type(model)(model.inp.in_features, model.d)
    tm.load_state_dict(model.state_dict())
    with torch.no_grad():
        for lin in (tm.inp, tm.out):
            W = lin.weight.data
            scale = W.abs().mean(1, keepdim=True) / keep
            lin.weight.data = torch.round((W / scale).clamp(-1, 1)) * scale
    return tm


def sparsity(model):
    z = t = 0
    for lin in (model.inp, model.out):
        z += int((lin.weight.data == 0).sum()); t += lin.weight.numel()
    return z / t


# -------------------------------------------------------------------- eval

def clip_scores(model, feats, device, bs=32):
    """One anomaly score per clip: mean squared next-frame prediction error."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            b = feats[i:i + bs].to(device)
            pred = model(b[:, :-1])
            err = ((pred - b[:, 1:]) ** 2).mean((1, 2))
            out.append(err.cpu())
    return torch.cat(out).numpy()


def auc(scores, labels):
    """ROC AUC without sklearn: rank-sum."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n1, n0 = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def trivial_scores(feats, tr, test):
    """Anomaly scores that involve no model and no training.

    These exist because the first version of this experiment produced a
    result that looked good and meant nothing: on pump id_00 at 6 dB, the
    distance between a clip's average spectrum and the average NORMAL
    spectrum scores 0.98 AUC. Any model not beating that is being credited
    for work a 64-number template already did, so every run reports these
    alongside. The template comes from the TRAINING normals only.

    `delta` and `texture` are the temporal ones — a model whose whole claim
    is that it captures dynamics has to beat those specifically.

    Always computed on the pre-CMVN features. Computing the template on
    CMVN'd features gives a score that is identically zero in exact
    arithmetic, whose float32 rounding residue (~5e-7) still reaches 0.89
    AUC — measuring numerical dust, not the signal.
    """
    mu_spec = feats[tr].mean((0, 1))
    F = feats[test]
    return {
        "template": ((F.mean(1) - mu_spec) ** 2).mean(1).numpy(),
        "loudness": F.mean((1, 2)).numpy(),
        "delta": (F[:, 1:] - F[:, :-1]).abs().mean((1, 2)).numpy(),
        "texture": F.std(1).mean(1).numpy(),
    }


def one_run(raw, y, seed, a, dev, verbose=False):
    """One seed: split, train on normal only, score float and ternary.

    The seed moves both the held-out split and the initialisation, so the
    spread across seeds covers data luck as well as optimiser luck.
    """
    idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(len(idx) * 0.67)
    tr, ho = idx[:cut], idx[cut:]
    test = np.concatenate([ho, np.flatnonzero(y == 1)])
    test_y = np.concatenate([np.zeros(len(ho), int), np.ones(int(y.sum()), int)])

    # normalisation from TRAINING normals only — held-out clips must not touch
    # the statistics, even though it is only two scalars
    feats_raw = (raw - raw[tr].mean()) / raw[tr].std()
    feats = feats_raw - feats_raw.mean(1, keepdim=True) if a.cmvn else feats_raw

    out = {"seed": seed}
    for name, s in trivial_scores(feats_raw, tr, test).items():
        v = auc(s, test_y)
        out[f"auc_{name}"] = max(v, 1 - v)      # a detector may fire either way

    torch.manual_seed(seed)
    model = Recur(feats.shape[-1], a.d).to(dev)

    # untrained control: if a random recurrence already separates the classes,
    # the result is about the log-mel features, not about anything we learned
    a_init = auc(clip_scores(model, feats[test], dev), test_y)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    Xtr = feats[tr].to(dev)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot = 0.0
        for i in range(0, len(Xtr), 16):
            b = Xtr[perm[i:i + 16]]
            loss = ((model(b[:, :-1]) - b[:, 1:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3}  loss {tot/max(1,len(Xtr)//16):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    out["auc_init"] = a_init
    out["auc_float"] = auc(clip_scores(model, feats[test], dev), test_y)
    for keep in a.keep:
        tm = ternarize(model, keep).to(dev)
        out[f"auc_tern_{keep}"] = auc(clip_scores(tm, feats[test], dev), test_y)
        out[f"sparsity_{keep}"] = sparsity(tm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--keep", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cmvn", action="store_true",
                    help="subtract each clip's own mean spectrum, which removes "
                         "the static-template shortcut and leaves only temporal "
                         "structure for the model to work with")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.data)
    X, y, sr = z["X"], z["y"], int(z["sr"])
    if X.dtype == np.int16:                             # cached as int16 samples
        X = X.astype(np.float32) / 32768.0
    print(f"{len(X)} clips @ {sr} Hz, {int(y.sum())} abnormal — features…")

    feats = torch.stack([log_mel(x, sr) for x in X])
    print(f"  {tuple(feats.shape)}  (clips, frames, mels)  on {dev}"
          f"{'  [cmvn: per-clip mean spectrum removed]' if a.cmvn else ''}")

    runs = []
    for s in range(a.seeds):
        runs.append(one_run(feats, y, s, a, dev, verbose=(s == 0)))
        r = runs[-1]
        print(f"  seed {s}: tmpl {r['auc_template']:.3f}  delta {r['auc_delta']:.3f}"
              f"  init {r['auc_init']:.3f}  float {r['auc_float']:.3f}"
              + "".join(f"  t{k} {r[f'auc_tern_{k}']:.3f}" for k in a.keep),
              flush=True)

    def stat(key):
        v = np.array([r[key] for r in runs])
        return v.mean(), v.std()

    params = sum(p.numel() for p in Recur(feats.shape[-1], a.d).parameters())
    print(f"\nmodel: {params:,} parameters, d={a.d}, constant state, "
          f"{a.seeds} seeds\n")
    print(f"  {'weights':<24}{'AUC (mean±sd)':>18}  {'zeros':>7}  {'gates/MAC':>10}")
    for name, label in (("template", "no model: spectrum tmpl"),
                        ("delta", "no model: frame delta"),
                        ("texture", "no model: texture"),
                        ("loudness", "no model: loudness")):
        m, s = stat(f"auc_{name}")
        print(f"  {label:<24}{m:>10.3f} ±{s:.3f}  {'—':>7}  {'—':>10}")
    m, s = stat("auc_init")
    print(f"  {'untrained control':<24}{m:>10.3f} ±{s:.3f}  {'—':>7}  {'—':>10}")
    m0, s0 = stat("auc_float")
    print(f"  {'float32':<24}{m0:>10.3f} ±{s0:.3f}  {'—':>7}  {305:>10}")
    for k in a.keep:
        m, s = stat(f"auc_tern_{k}")
        z_, _ = stat(f"sparsity_{k}")
        print(f"  {f'ternary keep={k}':<24}{m:>10.3f} ±{s:.3f}  "
              f"{z_*100:>6.0f}%  {74:>10}")

    # paired across seeds: the comparison that matters is per-seed, not marginal
    print("\n  paired float -> ternary, per seed:")
    for k in a.keep:
        d = np.array([r[f"auc_tern_{k}"] - r["auc_float"] for r in runs])
        print(f"    keep={k}: {d.mean():+.3f} ±{d.std():.3f}  "
              f"(wins {int((d>0).sum())}/{len(d)})")
    mt, _ = stat("auc_template")
    print(f"\n  vs the no-model template ({mt:.3f}): "
          f"float {m0 - mt:+.3f}, best ternary "
          f"{max(stat(f'auc_tern_{k}')[0] for k in a.keep) - mt:+.3f}")
    print("  MIMII paper's autoencoder baseline, pump @ 6 dB: ~0.85")

    out = a.out or os.path.splitext(a.data)[0] + "-result.json"
    json.dump({"runs": runs, "params": params, "d": a.d,
               "epochs": a.epochs, "seeds": a.seeds, "keep": a.keep},
              open(out, "w"), indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
