"""Task #56 — published architectures WITHOUT pretraining, under our
exact regime (8 ep, batch 64, AdamW 3e-4 OneCycle, clip 1.0), for the
missing from-scratch comparison column.

python3 -m whitebox.hyena_scratch --task nt_splice_sites_donors
Uses HyenaDNA-tiny architecture (LongSafari HF, random init). NOTE:
unidirectional, no RC averaging — their standard usage; recorded as a
protocol difference.
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.dna_train import load_split                    # noqa: E402

NAME = 'LongSafari/hyenadna-tiny-1k-seqlen-hf'


def build(n_classes):
    from transformers import (AutoConfig, AutoModel,
                              AutoModelForSequenceClassification)
    cfg = AutoConfig.from_pretrained(NAME, trust_remote_code=True)
    try:
        cfg.num_labels = n_classes
        return AutoModelForSequenceClassification.from_config(
            cfg, trust_remote_code=True), None
    except Exception:
        core = AutoModel.from_config(cfg, trust_remote_code=True)
        head = torch.nn.Linear(cfg.d_model, n_classes)
        return core, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    # random-init training: token IDENTITY is irrelevant (no pretrained
    # embedding to match) — use our 5-symbol encoding directly.
    from whitebox.dna_train import to_tensor
    tr_s, tr_y, classes = load_split(args.task, 'train')
    te_s, te_y, _ = load_split(args.task, 'test')
    maxlen = max(len(s) for s in tr_s)
    Xtr, Xte = to_tensor(tr_s, maxlen), to_tensor(te_s, maxlen)
    Ytr = torch.from_numpy(tr_y)
    Yte = torch.from_numpy(te_y)
    print(f'{args.task}: {len(tr_s)} train / {len(te_s)} test', flush=True)

    model, head = build(len(classes))
    model.to(device)
    params = list(model.parameters())
    if head is not None:
        head.to(device)
        params += list(head.parameters())
    n = sum(p.numel() for p in params)
    print(f'arm hyenadna-scratch: {n/1e3:.0f}k params on {device}',
          flush=True)

    def logits_of(xb):
        if head is None:
            return model(xb).logits
        h = model(xb).last_hidden_state
        return head(h.mean(1))

    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
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
            loss = F.cross_entropy(logits_of(Xtr[idx].to(device)),
                                   Ytr[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for bi in range(0, len(te_s), 256):
                preds.append(logits_of(Xte[bi:bi+256].to(device))
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
        f.write(json.dumps(dict(task=args.task, arm='hyenadna-scratch',
                                seed=args.seed, params=n,
                                best_acc=round(best, 4))) + '\n')
    print(f'FINAL hyenadna-scratch {args.task} best_acc {best:.4f}',
          flush=True)


if __name__ == '__main__':
    main()
