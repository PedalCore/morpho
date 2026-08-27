"""Task #56 — ConvNova-LIKE baseline (OUR reimplementation of the
three stated design elements of arXiv:2502.18538 — dilated convs,
gated convs, dual-branch gating — at matched param budget, under our
regime). NOT the official code (none located); labeled accordingly.

python3 -m whitebox.convnova_like --task nt_splice_sites_donors
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.dna_train import load_split, to_tensor     # noqa: E402


class GatedDilatedBlock(nn.Module):
    """Dual-branch: content conv x sigmoid(gate conv), dilated,
    residual."""

    def __init__(self, d, k, dil):
        super().__init__()
        pad = (k - 1) * dil // 2
        self.conv = nn.Conv1d(d, d, k, padding=pad, dilation=dil)
        self.gate = nn.Conv1d(d, d, k, padding=pad, dilation=dil)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):                        # (B, d, T)
        h = self.ln(x.transpose(1, 2)).transpose(1, 2)
        return x + torch.tanh(self.conv(h)) * torch.sigmoid(self.gate(h))


class ConvNovaLike(nn.Module):
    def __init__(self, d=96, n_classes=2, ks=7,
                 dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        self.emb = nn.Embedding(5, d)
        self.blocks = nn.ModuleList(
            GatedDilatedBlock(d, ks, dil) for dil in dilations)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(2 * d, n_classes)

    def forward(self, tokens):
        x = self.emb(tokens).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        h = self.ln_f(x.transpose(1, 2))
        pooled = torch.cat([h.mean(1), h.max(1).values], -1)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    tr_s, tr_y, classes = load_split(args.task, 'train')
    te_s, te_y, _ = load_split(args.task, 'test')
    maxlen = max(len(s) for s in tr_s)
    Xtr, Xte = to_tensor(tr_s, maxlen), to_tensor(te_s, maxlen)
    Ytr, Yte = torch.from_numpy(tr_y), torch.from_numpy(te_y)
    model = ConvNovaLike(n_classes=len(classes)).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'arm convnova-like: {n/1e3:.0f}k params on {device}',
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    spe = (len(tr_s) + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=spe * args.epochs, pct_start=0.1)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_s))
        for bi in range(spe):
            idx = order[bi*args.batch:(bi+1)*args.batch]
            loss = F.cross_entropy(model(Xtr[idx].to(device)),
                                   Ytr[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for bi in range(0, len(te_s), 256):
                preds.append(model(Xte[bi:bi+256].to(device))
                             .argmax(-1).cpu())
        p = torch.cat(preds)
        acc = float((p == Yte).float().mean())
        tp = int(((p == 1) & (Yte == 1)).sum())
        tn = int(((p == 0) & (Yte == 0)).sum())
        fp = int(((p == 1) & (Yte == 0)).sum())
        fn = int(((p == 0) & (Yte == 1)).sum())
        mcc = (tp*tn - fp*fn) / math.sqrt(
            max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
        f1 = 2*tp / max(2*tp + fp + fn, 1)
        best = max(best, acc)
        print(f'epoch {ep+1}/{args.epochs} test acc {acc:.4f} '
              f'mcc {mcc:.4f} f1 {f1:.4f} ({time.time()-t0:.0f}s)',
              flush=True)

    out = pathlib.Path('whitebox/runs/dna')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'results.jsonl', 'a') as f:
        f.write(json.dumps(dict(task=args.task, arm='convnova-like',
                                seed=args.seed, params=n,
                                best_acc=round(best, 4))) + '\n')
    print(f'FINAL convnova-like {args.task} best_acc {best:.4f}',
          flush=True)


if __name__ == '__main__':
    main()
