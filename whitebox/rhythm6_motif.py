"""M10 v6 — SHN-inspired motif memory (paper 3798: WTA slots +
SIM-STDP refresh + TTG consolidation), online and gradient-free.

Slots store BAR GRIDS (144-d). Per completed bar: cosine winner; if
similar enough (TTG gate) refresh winner toward the bar (STDP-like
w += A(x-w)) and consolidate (threshold decays with win count);
else claim the weakest slot. The retrieved slot supplies a per-
(position,voice) MEMORY PRIOR feature to the GLM for the next bar.

NULL CONTROL (preregistered): copy-previous-bar feature. Slot memory
must beat it, especially on POST-FILL RECOVERY (bars after sparse
bars), where copying repeats the fill but consolidated memory
reinstates the groove.

python3 -m whitebox.rhythm6_motif --fit
python3 -m whitebox.rhythm6_motif --navigate
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.rhythm3_exact import (Config, NV, OUT,               # noqa
                                    clock_features, load_store,
                                    traces9, _trace_lam,
                                    _update_trace_state)
from whitebox.rhythm5_modes import bar_grids                        # noqa

M_SLOTS = 16
A_REFRESH = 0.35
TTG_GAMMA = 4.0


class MotifStore:
    """Online WTA slot memory over bar grids."""

    def __init__(self):
        self.W = np.zeros((M_SLOTS, 144), np.float64)
        self.g = np.zeros(M_SLOTS, int)           # win counts
        self.theta = np.full(M_SLOTS, 0.35)       # TTG gates (max MSE)

    def _sim(self, x):
        n = np.linalg.norm(self.W, axis=1) * (np.linalg.norm(x) + 1e-9)
        return (self.W @ x) / (n + 1e-9)

    def retrieve(self, x):
        """Best consolidated match to cue x (no update)."""
        s = self._sim(x)
        i = int(np.argmax(s))
        return (self.W[i], i, float(s[i]))

    def store(self, x):
        if x.sum() == 0:
            return
        s = self._sim(x)
        i = int(np.argmax(s))
        delta = float(((self.W[i] - x) ** 2).mean())
        if self.g[i] == 0 or delta <= self.theta[i]:
            self.W[i] += A_REFRESH * (x - self.W[i])   # SIM-STDP refresh
            self.g[i] += 1
            self.theta[i] *= (1 - 1 / (TTG_GAMMA + self.g[i]))  # TTG
        else:
            j = int(np.argmin(self.g))                 # claim weakest
            self.W[j] = x.copy()
            self.g[j] = 1


def memory_feats(x, cfg):
    """Per-step features: [slot-memory prior, copy-prev-bar (null)].
    Causal: bar b's features come from bars < b."""
    spb = cfg.steps_per_bar
    q = cfg.steps_per_sixteenth
    g = bar_grids(x, cfg)
    store = MotifStore()
    F = np.zeros((len(x), 2), np.float32)
    prev = np.zeros(144)
    for b in range(len(g) + 1):
        lo, hi = b * spb, min((b + 1) * spb, len(x))
        if lo >= len(x):
            break
        rec, _, sim = store.retrieve(prev) if b else (np.zeros(144), -1, 0)
        for t in range(lo, hi):
            slot = ((t - lo) // q) % 16
            F[t, 0] = rec[slot * NV:(slot + 1) * NV].sum() and 0 or 0
        # vector fill (per voice) below
        for t in range(lo, hi):
            slot = ((t - lo) // q) % 16
            F[t, 0] = 0.0                      # placeholder, set in caller
        if b < len(g):
            store.store(g[b])
            prev = g[b]
    return F


def full_feats(x, cfg):
    """clock + traces + [memory prior, copy-prev] per (step, voice).
    Voice-resolved: returns (T, 28+90) shared + (T, NV, 2) extras."""
    spb, q = cfg.steps_per_bar, cfg.steps_per_sixteenth
    C = clock_features(len(x), cfg)
    Tr = traces9(x, cfg)
    g = bar_grids(x, cfg)
    store = MotifStore()
    E = np.zeros((len(x), NV, 2), np.float32)
    prev = np.zeros(144)
    for b in range(int(np.ceil(len(x) / spb))):
        lo, hi = b * spb, min((b + 1) * spb, len(x))
        rec = store.retrieve(prev)[0] if b else np.zeros(144)
        for t in range(lo, hi):
            slot = ((t - lo) // q) % 16
            E[t, :, 0] = rec[slot * NV:(slot + 1) * NV]
            E[t, :, 1] = prev[slot * NV:(slot + 1) * NV]
        if b < len(g):
            store.store(g[b])
            prev = g[b]
    return np.concatenate([C, Tr], 1), E


def fit(use=('mem', 'copy')):
    import torch
    cfg = Config()
    stored = load_store()

    def build(split):
        Xs, Es, Ys = [], [], []
        for it in stored[split]:
            S, E = full_feats(it['x'], cfg)
            Xs.append(S)
            Es.append(E)
            Ys.append(it['x'])
        return (np.concatenate(Xs), np.concatenate(Es),
                np.concatenate(Ys).astype(np.float32))

    Xtr, Etr, Ytr = build('train')
    Xte, Ete, Yte = build('test')
    cols = []
    if 'mem' in use:
        cols.append(0)
    if 'copy' in use:
        cols.append(1)
    r0 = np.clip(Ytr.mean(0), 1e-5, 1 - 1e-5)
    torch.manual_seed(0)
    shared = torch.nn.Linear(Xtr.shape[1], NV)
    wE = torch.nn.Parameter(torch.zeros(NV, len(cols)))
    with torch.no_grad():
        shared.bias.copy_(torch.from_numpy(
            np.log(r0 / (1 - r0)).astype(np.float32)))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.from_numpy((Xtr - mu) / sd)
    Et = torch.from_numpy(Etr[:, :, cols])
    Yt = torch.from_numpy(Ytr)
    opt = torch.optim.Adam(list(shared.parameters()) + [wE], lr=3e-3)
    for ep in range(8):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 65536):
            ii = perm[i:i+65536]
            z = shared(Xt[ii]) + (Et[ii] * wE[None]).sum(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, Yt[ii])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = shared(torch.from_numpy((Xte - mu) / sd)) + \
            (torch.from_numpy(Ete[:, :, cols]) * wE[None]).sum(-1)
        P = torch.sigmoid(z).numpy()
    eps = 1e-7

    def bpe(Pm, Ym):
        nll = -(Ym * np.log2(Pm + eps) +
                (1 - Ym) * np.log2(1 - Pm + eps)).sum()
        P0 = np.broadcast_to(r0, Ym.shape)
        nll0 = -(Ym * np.log2(P0 + eps) +
                 (1 - Ym) * np.log2(1 - P0 + eps)).sum()
        return (nll0 - nll) / max(Ym.sum(), 1)

    tag = '+'.join(use) if use else 'base'
    npar = sum(p.numel() for p in shared.parameters()) + wE.numel()
    print(f'{tag:12s}: {npar} params  {bpe(P, Yte):+.3f} b/ev',
        flush=True)
    # POST-SPARSE-BAR RECOVERY metric: bars whose PREVIOUS bar had
    # low density (fill-ish) — where copy-prev misleads
    cfg2 = Config()
    spb = cfg2.steps_per_bar
    mask = np.zeros(len(Yte), bool)
    off = 0
    for it in stored['test']:
        n = len(it['x'])
        gg = bar_grids(it['x'], cfg2)
        dens = gg.sum(1)
        med = np.median(dens[dens > 0]) if (dens > 0).any() else 0
        for b in range(1, len(gg)):
            if dens[b - 1] < 0.5 * med:
                mask[off + b*spb: off + min((b+1)*spb, n)] = True
        off += n
    if mask.any():
        print(f'  post-sparse-bar b/ev: {bpe(P[mask], Yte[mask]):+.3f} '
              f'({mask.sum()} steps)', flush=True)
    return npar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', action='store_true')
    a = ap.parse_args()
    if a.fit:
        fit(use=())
        fit(use=('copy',))
        fit(use=('mem',))
        fit(use=('mem', 'copy'))


if __name__ == '__main__':
    main()
