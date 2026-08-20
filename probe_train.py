"""The locked probe suite (PROBES.md): train each architecture on the
task family, evaluate on the fixed delay grid.

python3 -m whitebox.probe_train --arch {kv,win32,dval,crsa} --seed {0,1,2}

Declared configs (matched parameter budgets): d=128, L=4, K=4, ctx=128,
vocab 64, M2-control wiring with identity codes, 20k steps, spikelm
optimizer recipe. Loss and accuracy only on answer spans. Results append
to whitebox/runs/probes/results.jsonl.
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

from whitebox.model import Config, CausalCRATEM2  # noqa: E402
from whitebox import probes  # noqa: E402

ARCHS = {
    'kv':    dict(attn='mssa', window=0),
    'win32': dict(attn='mssa', window=32),
    'dval':  dict(attn='dval', window=0),
    'crsa':  dict(attn='crsa', window=0),
}


def pick_device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def make_model(arch):
    cfg = Config(vocab_size=probes.VOCAB, n_layer=4, n_embd=128, n_head=4,
                 ctx=128, m2='b', m2_identity=True, **ARCHS[arch])
    return CausalCRATEM2(cfg), cfg


def batch_loss(model, x, mask, device):
    x = torch.from_numpy(x).to(device)
    mask = torch.from_numpy(mask).to(device)
    logits, _ = model(x[:, :-1])
    tgt = x[:, 1:]
    m = mask[:, 1:]
    lo = logits.reshape(-1, logits.shape[-1])[m.reshape(-1)]
    return F.cross_entropy(lo, tgt.reshape(-1)[m.reshape(-1)])


@torch.no_grad()
def eval_grid(model, device):
    model.eval()
    out = {}
    for task in probes.TASKS:
        for delay in probes.DELAY_GRID:
            ex = probes.eval_set(task, delay, n=200)
            L = max(len(e[0]) for e in ex)
            x = np.full((len(ex), L), probes.PAD, dtype=np.int64)
            for i, (seq, *_rest) in enumerate(ex):
                x[i, :len(seq)] = seq
            xt = torch.from_numpy(x).to(device)
            logits, _ = model(xt[:, :-1])
            pred = logits.argmax(-1).cpu().numpy()
            correct = total = 0
            for i, (seq, start, ans, _, _) in enumerate(ex):
                for j, a in enumerate(ans):
                    correct += int(pred[i, start + j - 1] == a)
                    total += 1
            out[f'{task}@{delay}'] = round(correct / total, 4)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=list(ARCHS), required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20000)
    args = ap.parse_args()

    device = pick_device()
    torch.manual_seed(args.seed)
    model, cfg = make_model(args.arch)
    model = model.to(device)
    n_par = model.num_params()
    print(f'probe {args.arch} seed {args.seed}: {n_par/1e3:.0f}k params '
          f'on {device}', flush=True)

    stream = probes.train_stream(seed=probes.DATA_SEED + args.seed, batch=16)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.99),
                            weight_decay=0.01)
    warmup = 200

    def lr_at(step):
        if step < warmup:
            return 6e-4 * step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return 6e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        x, mask = next(stream)
        loss = batch_loss(model, x, mask, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 4000 == 0 or step == args.steps - 1:
            acc = eval_grid(model, device)
            mean_acc = float(np.mean(list(acc.values())))
            print(f'{args.arch} s{args.seed} step {step:5d} loss '
                  f'{float(loss):.3f} mean-acc {mean_acc:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    acc = eval_grid(model, device)
    rec = dict(arch=args.arch, seed=args.seed, steps=args.steps,
               params=n_par, acc=acc)
    outdir = pathlib.Path(__file__).parent / 'runs' / 'probes'
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'results.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    torch.save(dict(model=model.state_dict(), cfg=vars(cfg)),
               outdir / f'{args.arch}-s{args.seed}.pt')
    print('FINAL', args.arch, args.seed,
          {k: v for k, v in list(acc.items())[:6]}, '...', flush=True)


if __name__ == '__main__':
    main()
