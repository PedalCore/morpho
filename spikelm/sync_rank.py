"""How wide is a synchronisation representation, really?

The scaling case for sync rests on one claim: that P sampled neuron PAIRS
carry more usable dimensions than the D neurons they come from, so a
narrow state can drive a wide representation and the O(D^2) synapse model
shrinks. If that is false, sync is just an expensive readout and the whole
direction is dead.

There is a reason to doubt it. For a single input, S = Z Z^T is built from
Z of shape (D, t), so it has rank at most min(D, t). Sampling P > D pairs
from a rank-limited matrix could easily give P numbers spanning far fewer
than P directions - width on paper, not in the representation.

The reason to hope is the per-pair learnable decay. Each pair applies its
OWN exponential filter over the tick history, so the pairs are not entries
of one Gram matrix; they are products read through different temporal
filters. That breaks the plain rank bound. By how much is what this
measures.

Method: train a CTM, record the post-activations Z it actually produces on
real inputs, then compute sync features OFFLINE under many pair counts and
decay settings. Same activations throughout, so nothing varies but the
readout being measured. Effective rank by three standard measures, because
any single one can be gamed by a long tail of tiny eigenvalues:

    participation ratio  (sum L)^2 / sum L^2     - the usual one
    n95                  components for 95% var  - the practical one
    stable rank          ||X||_F^2 / ||X||_2^2   - the pessimistic one

The comparison that decides it: effective rank of the SYNC features
against effective rank of the ACTIVATIONS they were computed from. Above
D means genuine width. At or below D means the pairs are decoration.

    python sync_rank.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ctm import tail_mean_loss
from ctm_parity import ParityCTM, batch


def effective_rank(X):
    """Three views of how many directions a feature matrix really uses."""
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    lam = s ** 2
    tot = lam.sum()
    if tot <= 0:
        return dict(participation=0.0, n95=0, stable=0.0)
    pr = float(tot ** 2 / (lam ** 2).sum())
    n95 = int(np.searchsorted(np.cumsum(lam) / tot, 0.95) + 1)
    stable = float(tot / lam[0])
    return dict(participation=pr, n95=n95, stable=stable)


def sync_features(Z, ia, ib, r):
    """Decay-weighted pair products, exactly as the model computes them.

    Z: (n, T, D) post-activations. r: (P,) decay rates, one per pair.
    Returns (n, P) — the representation at the final tick.
    """
    n, T, D = Z.shape
    decay = np.exp(-r)[None, :]                       # (1, P)
    num = np.zeros((n, len(ia)), dtype=np.float64)
    den = np.zeros((1, len(ia)), dtype=np.float64)
    for t in range(T):
        num = decay * num + Z[:, t, ia] * Z[:, t, ib]
        den = decay * den + 1.0
    return num / np.sqrt(np.maximum(den, 1e-9))


def train_and_record(D, L, T, steps, device, n_record=2048, seed=0):
    """Train on parity, then record the activations it actually produces."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = ParityCTM(L, ticks=T, D=D).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y = batch(128, L, rng, device)
        loss, _, _ = tail_mean_loss(m(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()

    # accuracy, so we know whether we are measuring a model that works
    ev = np.random.default_rng(999)
    x, y = batch(1024, L, ev, device)
    with torch.no_grad():
        acc = float((m(x)[:, -1].argmax(-1) == y).float().mean())

    # record post-activations by re-running the tick loop with a hook
    Z = []
    ctm = m.ctm
    with torch.no_grad():
        xr, _ = batch(n_record, L, np.random.default_rng(7), device)
        kv = m.emb(xr) + m.pos
        B = kv.shape[0]
        z = ctm.z0.expand(B, ctm.D).contiguous()
        A = ctm.a0.t().expand(B, ctm.D, ctm.M).contiguous()
        no, do = ctm.sync_out.reset(B, kv.device, kv.dtype)
        na, da = ctm.sync_act.reset(B, kv.device, kv.dtype)
        o = torch.zeros(B, ctm.attn.q.in_features, device=kv.device)
        for _ in range(T):
            a = ctm.synapse(torch.cat([z, o], dim=-1))
            A = torch.cat([A[:, :, 1:], a.unsqueeze(-1)], dim=-1)
            z = ctm.ln(ctm.nlm(A))
            Z.append(z.cpu().numpy().copy())
            s_out, no, do = ctm.sync_out.step(z, no, do)
            s_act, na, da = ctm.sync_act.step(z, na, da)
            o = ctm.attn(ctm.w_in(s_act), kv)
    learned_r = F.softplus(ctm.sync_out.r_raw).detach().cpu().numpy()
    return np.stack(Z, axis=1), acc, learned_r    # (n, T, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dims", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--pairs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--out", default="sync-rank.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}

    for D in a.dims:
        Z, acc, learned_r = train_and_record(D, a.length, a.ticks, a.steps, dev)
        n, T, _ = Z.shape
        act = effective_rank(Z[:, -1, :])          # activations, final tick
        print(f"\nD={D} neurons, T={T} ticks · parity accuracy {acc:.1%}")
        print(f"  activations at the final tick:  "
              f"participation {act['participation']:.1f}  n95 {act['n95']}  "
              f"stable {act['stable']:.1f}   (ceiling D={D})")
        print(f"  learned decays: min {learned_r.min():.3f} "
              f"max {learned_r.max():.3f} — {'diverse' if learned_r.std()>1e-3 else 'UNIFORM'}")
        print(f"\n  {'pairs':>6}{'decays':>12}{'participation':>15}{'n95':>7}"
              f"{'stable':>9}")
        res[f"D{D}"] = {"acc": acc, "activations": act, "arms": {}}

        rng = np.random.default_rng(0)
        for P in a.pairs:
            ia = rng.integers(0, D, P); ib = rng.integers(0, D, P)
            r_learned = rng.choice(learned_r, P) if len(learned_r) else np.zeros(P)
            for name, r in (("learned", r_learned),
                            ("uniform", np.full(P, float(np.mean(r_learned)))),
                            ("none (r=0)", np.zeros(P))):
                X = sync_features(Z, ia, ib, r)
                er = effective_rank(X)
                print(f"  {P:>6}{name:>12}{er['participation']:>15.1f}"
                      f"{er['n95']:>7}{er['stable']:>9.1f}")
                res[f"D{D}"]["arms"][f"P{P}-{name}"] = er

    print("\n  the question: does sync exceed the neuron count D?")
    for D in a.dims:
        best = max(v["participation"] for v in res[f"D{D}"]["arms"].values())
        actpr = res[f"D{D}"]["activations"]["participation"]
        print(f"    D={D:<4} activations {actpr:>7.1f} -> best sync "
              f"{best:>7.1f}   {'WIDER' if best > actpr * 1.2 else 'no gain'}")

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
