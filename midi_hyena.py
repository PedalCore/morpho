"""M9 — Hyena causal LM on ARIA tokens (param-matched to our arms).

python3 -m whitebox.midi_hyena [--steps 20000]
HyenaDNA architecture (LongSafari -hf remote code), random init,
config scaled to d_model 448 / 12 layers (~our param class),
vocab 219, our exact regime. wandb: morpho-whitebox-midi.
"""

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.midi_data import VOCAB, decode_tokens        # noqa: E402
from whitebox.midi_train import load_tokens, get_batch     # noqa: E402

NAME = 'LongSafari/hyenadna-tiny-1k-seqlen-hf'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', default='~/aria/tokens')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--ctx', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    device = 'cuda'
    torch.manual_seed(args.seed)

    train, valid = load_tokens(args.tokens)
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(NAME, trust_remote_code=True)
    cfg.vocab_size = VOCAB
    cfg.d_model = 448
    cfg.n_layer = 12
    cfg.max_seq_len = args.ctx + 2
    model = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f'arm hyena-music: {n/1e6:.1f}M params on {device}', flush=True)

    try:
        import wandb
        wb = wandb.init(project='morpho-whitebox-midi',
                        name=f'hyena-aria-s{args.seed}',
                        config=dict(arm='hyena', params=n,
                                    steps=args.steps, ctx=args.ctx))
    except Exception:
        wb = None

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            betas=(0.9, 0.99), weight_decay=0.1)
    rng = np.random.default_rng(args.seed)
    out = pathlib.Path('whitebox/runs/midi/hyena-s0')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    best = float('inf')

    def loss_of(x, y):
        logits = model(x).logits
        return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))

    for step in range(args.steps):
        lr = 3e-4 * min(1.0, (step + 1) / 200) * \
            (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
        for g in opt.param_groups:
            g['lr'] = lr
        x, y = get_batch(train, args.batch, args.ctx, rng, device)
        loss = loss_of(x, y)
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
                    tot += float(loss_of(xv, yv)) * yv.numel()
                    m += yv.numel()
            vl = tot / m
            print(f'step {step:5d} loss {float(loss):.3f} val {vl:.3f} '
                  f'ppl {math.exp(vl):.2f} ({time.time()-t0:.0f}s)',
                  flush=True)
            if wb is not None:
                wb.log(dict(train_loss=float(loss), val_loss=vl,
                            val_ppl=math.exp(vl)), step=step)
            if vl < best:
                best = vl
                torch.save(dict(model=model.state_dict()),
                           out / 'ckpt.pt')
            model.train()
    model.eval()
    toks = valid[:256].astype(np.int64).tolist()
    with torch.no_grad():
        for _ in range(1024):
            logits = model(torch.tensor([toks[-args.ctx:]],
                                        device=device)).logits
            p = F.softmax(logits[0, -1] / 0.95, -1)
            toks.append(int(torch.multinomial(p, 1)))
    decode_tokens(toks, out / 'continuation.mid')
    if wb is not None:
        wb.summary['best_val'] = best
        wb.finish()
    print(f'FINAL hyena aria best_val {best:.4f} '
          f'ppl {math.exp(best):.2f}', flush=True)


if __name__ == '__main__':
    main()
