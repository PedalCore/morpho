"""Post-hoc eval of saved DNA checkpoints: accuracy, MCC, F1.

python3 -m whitebox.dna_eval --ckpt whitebox/runs/dna/ckpt-counter-nt_H3K4me3-s0.pt
(or --all to sweep every ckpt in whitebox/runs/dna/)
"""

import argparse
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.dna_model import DNAClassifier            # noqa: E402
from whitebox.dna_train import load_split, to_tensor    # noqa: E402


def evaluate(path, device):
    ck = torch.load(path, map_location='cpu')
    te_s, te_y, classes = load_split(ck['task'], 'test')
    maxlen = max(len(s) for s in te_s)
    X = to_tensor(te_s, maxlen)
    model = DNAClassifier(arm=ck['arm'], n_classes=len(classes))
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te_s), 256):
            preds.append(model(X[i:i+256].to(device)).argmax(-1).cpu())
    p = torch.cat(preds).numpy()
    y = te_y
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)
    den = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
    mcc = (tp*tn - fp*fn) / den
    f1 = 2*tp / max(2*tp + fp + fn, 1)
    print(f'{ck["arm"]:8s} {ck["task"]:28s} s{ck.get("seed",0)} '
          f'acc {acc*100:.2f}  MCC {mcc*100:.2f}  F1 {f1*100:.2f}',
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    device = ('cuda' if torch.cuda.is_available() else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.all:
        for p in sorted(pathlib.Path('whitebox/runs/dna').glob('ckpt-*.pt')):
            evaluate(p, device)
    else:
        evaluate(args.ckpt, device)


if __name__ == '__main__':
    main()
