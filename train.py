"""Train causal CRATE on TinyStories, matched to the spikelm conventions.

python3 -m whitebox.train [--spike-prox] [--steps N] [--layers L]

Reuses spikelm's tokenizer and tokenized splits (machine-local path), and
its exact optimizer recipe — AdamW 6e-4, betas (0.9, 0.99), wd 0.01,
warmup 200, cosine to 10% — so perplexities sit in the same comparison
table as the RWKV/spiking/Mamba baselines. Logs val ppl AND the white-box
layer metrics (coding rate before/after each MSSA, ISTA sparsity) to
runs/<name>/log.jsonl.
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
SPIKELM = '/Users/marycarrigan/coding/morpho-snn/spikelm'
sys.path.insert(0, SPIKELM)

from spikelm.data import get_tokenizer, load_split, get_batch  # noqa: E402
from spikelm.evaluate import generate  # noqa: E402
from whitebox.model import Config, CausalCRATE  # noqa: E402


def pick_device():
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def evaluate(model, data, batch, ctx, device, iters=25, rng=None):
    model.eval()
    tot, n = 0.0, 0
    for _ in range(iters):
        x, y = get_batch(data, batch, ctx, rng, device)
        _, loss = model(x, y)
        tot += float(loss) * y.numel()
        n += y.numel()
    model.train()
    return math.exp(tot / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5500)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=6e-4)
    ap.add_argument('--layers', type=int, default=12)
    ap.add_argument('--width', type=int, default=384)
    ap.add_argument('--spike-prox', action='store_true')
    ap.add_argument('--untied', action='store_true')
    ap.add_argument('--scale-init', type=float, default=0.1)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    device = pick_device()
    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size, n_layer=args.layers,
                 n_embd=args.width,
                 tied=not args.untied, spike_prox=args.spike_prox,
                 mssa_scale=args.scale_init)
    model = CausalCRATE(cfg).to(device)
    name = args.name or ('crate-spike' if args.spike_prox else 'crate') + \
        f'-d{cfg.n_embd}L{cfg.n_layer}'
    run_dir = pathlib.Path(__file__).parent / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'{name}: {model.num_params() / 1e6:.1f}M params on {device}')

    train_data, valid_data = load_split('train'), load_split('valid')
    rng = np.random.default_rng(42)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.99), weight_decay=0.01)
    warmup = 200

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    log = open(run_dir / 'log.jsonl', 'a')
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        x, y = get_batch(train_data, args.batch, cfg.ctx, rng, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 500 == 0 or step == args.steps - 1:
            ppl = evaluate(model, valid_data, args.batch, cfg.ctx, device,
                           rng=np.random.default_rng(7))
            xm, _ = get_batch(valid_data, 8, cfg.ctx,
                              np.random.default_rng(3), device)
            metrics = model.layer_metrics(xm)
            rec = dict(step=step, train_loss=round(float(loss), 4),
                       val_ppl=round(ppl, 3),
                       elapsed=round(time.time() - t0),
                       layers=[dict(l=m['layer'],
                                    drc=round(m['rc_after'] - m['rc_before'], 3),
                                    sp=round(m['sparsity'], 3))
                               for m in metrics])
            log.write(json.dumps(rec) + '\n')
            log.flush()
            mean_drc = np.mean([m['rc_after'] - m['rc_before'] for m in metrics])
            mean_sp = np.mean([m['sparsity'] for m in metrics])
            print(f'step {step:5d}  loss {loss:.3f}  val_ppl {ppl:7.3f}  '
                  f'mean dR^c {mean_drc:+.3f}  sparsity {mean_sp:.2f}  '
                  f'({time.time() - t0:.0f}s)', flush=True)
            torch.save(dict(model=model.state_dict(), step=step,
                            cfg=vars(cfg)), run_dir / 'ckpt.pt')

    txt = generate(model, tok, 'Once upon a time, there was a little girl '
                   'named', max_new=100, device=device, seed=1000)
    print('\nsample:', txt[:280])
    log.close()


if __name__ == '__main__':
    main()
