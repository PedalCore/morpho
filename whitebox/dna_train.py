"""M7 trainer — GenomicBenchmarks classification, three matched arms.

python3 -m whitebox.dna_train --arm {cnn,counter,longhorn}
        [--task human_enhancers_cohn] [--epochs 8]

Same tokens, stem, MLP, width, depth, pooling, schedule everywhere;
only the mixer differs (M7-DNA.md). Results append to
whitebox/runs/dna/results.jsonl.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.dna_model import DNAClassifier, BASES  # noqa: E402

DATA = pathlib.Path.home() / '.genomic_benchmarks'


def load_split(task, split):
    root = DATA / task / split
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    seqs, labels = [], []
    for ci, cname in enumerate(classes):
        for f in (root / cname).glob('*.txt'):
            seqs.append(f.read_text().strip().upper())
            labels.append(ci)
    return seqs, np.array(labels), classes


def to_tensor(seqs, maxlen):
    x = np.full((len(seqs), maxlen), BASES['N'], dtype=np.int64)
    for i, s in enumerate(seqs):
        t = [BASES.get(c, 4) for c in s[:maxlen]]
        x[i, :len(t)] = t
    return torch.from_numpy(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True,
                    choices=['cnn', 'counter', 'longhorn'])
    ap.add_argument('--task', default='human_enhancers_cohn')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(
            f'whitebox/runs/tb/{args.arm}-{args.task}-s{args.seed}')
    except ImportError:
        tb = None
    torch.manual_seed(args.seed)

    tr_s, tr_y, classes = load_split(args.task, 'train')
    te_s, te_y, _ = load_split(args.task, 'test')
    maxlen = max(len(s) for s in tr_s)
    Xtr, Xte = to_tensor(tr_s, maxlen), to_tensor(te_s, maxlen)
    Ytr = torch.from_numpy(tr_y)
    Yte = torch.from_numpy(te_y)
    print(f'{args.task}: {len(tr_s)} train / {len(te_s)} test, '
          f'len {maxlen}, classes {classes}', flush=True)

    model = DNAClassifier(arm=args.arm, n_classes=len(classes)).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'arm {args.arm}: {n/1e3:.0f}k params on {device}', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    steps_per_epoch = (len(tr_s) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=total, pct_start=0.1)

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_s))
        for bi in range(steps_per_epoch):
            idx = order[bi*args.batch:(bi+1)*args.batch]
            xb = Xtr[idx].to(device)
            yb = Ytr[idx].to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for bi in range(0, len(te_s), 256):
                xb = Xte[bi:bi+256].to(device)
                pred = model(xb).argmax(-1).cpu()
                correct += int((pred == Yte[bi:bi+256]).sum())
        acc = correct / len(te_s)
        best = max(best, acc)
        if tb is not None:
            tb.add_scalar('test_acc', acc, ep + 1)
            tb.add_scalar('train_loss', float(loss), ep + 1)
        print(f'epoch {ep+1}/{args.epochs} test acc {acc:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)

    out = pathlib.Path('whitebox/runs/dna')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), arm=args.arm,
                    task=args.task, seed=args.seed),
               out / f'ckpt-{args.arm}-{args.task}-s{args.seed}.pt')
    with open(out / 'results.jsonl', 'a') as f:
        f.write(json.dumps(dict(task=args.task, arm=args.arm,
                                seed=args.seed, params=n,
                                best_acc=round(best, 4))) + '\n')
    print(f'FINAL {args.arm} {args.task} best_acc {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
