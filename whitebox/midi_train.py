"""M9 — MIDI-event LM on ARIA tokens (whitebox stack, ctx 512).

python3 -m whitebox.midi_train --arm longhorn [--steps 20000]
arms: counter (crsa) | longhorn | mixed | qkv (control)
Logs to wandb project morpho-whitebox-midi. Saves ckpt + a sampled
continuation (MIDI) from a held-out prompt every eval.
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

from whitebox.model import Config, CausalCRATEM2          # noqa: E402
from whitebox.midi_data import VOCAB, decode_tokens        # noqa: E402

ARMS = {'counter': 'crsa', 'longhorn': 'longhorn',
        'mixed': 'mixed', 'qkv': 'qkv'}


def load_tokens(d):
    d = pathlib.Path(d).expanduser()
    shards = sorted(d.glob('tokens-*.npy'))
    data = np.concatenate([np.load(s) for s in shards])
    n_val = max(len(data) // 50, 10_000)
    return data[:-n_val], data[-n_val:]


def get_batch(data, bs, ctx, rng, device):
    ix = rng.integers(0, len(data) - ctx - 1, size=bs)
    x = np.stack([data[i:i+ctx] for i in ix]).astype(np.int64)
    y = np.stack([data[i+1:i+ctx+1] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--tokens', default='~/aria/tokens')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--n-embd', type=int, default=448)
    ap.add_argument('--n-layer', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    torch.manual_seed(args.seed)

    train, valid = load_tokens(args.tokens)
    print(f'tokens: {len(train)/1e6:.1f}M train / {len(valid)/1e6:.2f}M '
          f'val, vocab {VOCAB}', flush=True)

    cfg = Config(n_embd=args.n_embd, n_head=16, n_layer=args.n_layer,
                 ctx=args.ctx, vocab_size=VOCAB, attn=ARMS[args.arm],
                 mlp=True, tied=True, m2='b', m2_identity=True)
    model = CausalCRATEM2(cfg).to(device)
    n = model.num_params()
    print(f'arm {args.arm}: {n/1e6:.1f}M params on {device}', flush=True)

    try:
        import wandb
        wb = wandb.init(project='morpho-whitebox-midi',
                        name=f'{args.arm}-aria-s{args.seed}',
                        config=dict(arm=args.arm, params=n,
                                    **{k: getattr(args, k) for k in
                                       ('steps', 'batch', 'ctx',
                                        'n_embd', 'n_layer', 'seed')}))
    except Exception:
        wb = None

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.99), weight_decay=0.1)
    warm = 200
    rng = np.random.default_rng(args.seed)
    out = pathlib.Path(f'whitebox/runs/midi/{args.arm}-s{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best = float('inf')
    for step in range(args.steps):
        lr = 3e-4 * min(1.0, (step + 1) / warm) * \
            (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        x, y = get_batch(train, args.batch, args.ctx, rng, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                tot = m = 0
                for _ in range(20):
                    xv, yv = get_batch(valid, args.batch, args.ctx,
                                       rng, device)
                    _, lv = model(xv, yv)
                    tot += float(lv) * yv.numel()
                    m += yv.numel()
            vl = tot / m
            print(f'step {step:5d} loss {float(loss):.3f} val {vl:.3f} '
                  f'ppl {math.exp(vl):.2f} ({time.time()-t0:.0f}s)',
                  flush=True)
            if wb is not None:
                wb.log(dict(train_loss=float(loss), val_loss=vl,
                            val_ppl=math.exp(vl), lr=lr), step=step)
            if vl < best:
                best = vl
                torch.save(dict(model=model.state_dict(),
                                cfg=vars(cfg)), out / 'ckpt.pt')
            model.train()
    # sample a continuation from a held-out prompt
    model.eval()
    prompt = torch.from_numpy(
        valid[:256].astype(np.int64))[None].to(device)
    toks = prompt[0].tolist()
    with torch.no_grad():
        for _ in range(1024):
            logits, _ = model(torch.tensor([toks[-args.ctx:]],
                                           device=device))
            p = F.softmax(logits[0, -1] / 0.95, -1)
            toks.append(int(torch.multinomial(p, 1)))
    decode_tokens(toks, out / 'continuation.mid')
    if wb is not None:
        wb.summary['best_val'] = best
        wb.finish()
    print(f'FINAL {args.arm} aria best_val {best:.4f} '
          f'ppl {math.exp(best):.2f}', flush=True)


if __name__ == '__main__':
    main()
