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
from whitebox.model import Config, CausalCRATE, CausalCRATEM2  # noqa: E402


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
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--dict', type=int, default=1, dest='dict_expand',
                    help='overcomplete dictionary expansion (DICTIONARY.md)')
    ap.add_argument('--dict-local', action='store_true',
                    help='block-local a0=0 form (the factorial design)')
    ap.add_argument('--dict-identity', action='store_true',
                    help='factorial F1/F3: prox disabled (linear)')
    ap.add_argument('--mlp', action='store_true',
                    help='conventional 4d GELU MLP control block')
    ap.add_argument('--spike-prox', action='store_true')
    ap.add_argument('--untied', action='store_true')
    ap.add_argument('--scale-init', type=float, default=0.1)
    ap.add_argument('--init-from', default=None,
                    help='warm-start weights from another run ckpt (the '
                         'annealing lesson: quantized variants fine-tune '
                         'from float rather than train from scratch)')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--m2', choices=['a', 'b'], default=None)
    ap.add_argument('--crsa', '--tssa', dest='crsa', action='store_true',
                    help='CRSA statistics attention instead of MSSA')
    ap.add_argument('--tost', action='store_true',
                    help='uniform-measure (ToST) ablation of CRSA')
    ap.add_argument('--m2-identity', action='store_true',
                    help='M2 control: reordered wiring, identity quantizer')
    ap.add_argument('--anneal', default=None,
                    help='level schedule, e.g. "2500:2,4000:1"')
    ap.add_argument('--calibrate-from', default=None,
                    help='M2-control ckpt: init weights AND set thresholds '
                         'from measured pre-consumer activation quantiles')
    ap.add_argument('--blend-steps', type=int, default=0,
                    help='ramp float->quantized alpha 0->1 over N steps')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    device = 'cpu' if args.cpu else pick_device()
    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size, n_layer=args.layers,
                 n_embd=args.width, n_head=args.heads,
                 tied=not args.untied, spike_prox=args.spike_prox,
                 mssa_scale=args.scale_init,
                 m2=args.m2 or '', m2_identity=args.m2_identity,
                 dict_expand=args.dict_expand,
                 dict_local=args.dict_local, dict_identity=args.dict_identity,
                 mlp=args.mlp,
                 attn='tost' if args.tost else 'crsa' if args.crsa else 'mssa')
    model = (CausalCRATEM2(cfg) if args.m2 else CausalCRATE(cfg)).to(device)
    anneal = ([(int(s.split(':')[0]), int(s.split(':')[1]))
               for s in args.anneal.split(',')] if args.anneal else [])
    name = args.name or ('crate-spike' if args.spike_prox else 'crate') + \
        f'-d{cfg.n_embd}L{cfg.n_layer}'
    run_dir = pathlib.Path(__file__).parent / 'runs' / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.init_from:
        src = torch.load(args.init_from, map_location=device)
        sd = src['model']
        if args.m2:      # remap M0 module names onto the M2 wiring
            sd = {k.replace('.ista.D', '.D').replace('.ln2.', '.ln.'): v
                  for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'warm-start from {args.init_from}: '
              f'{len(missing)} new params, {len(unexpected)} dropped')
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

    if args.calibrate_from:
        from whitebox.calibrate import calibrate
        src = torch.load(args.calibrate_from, map_location=device)
        sd = {k.replace('.ista.D', '.D').replace('.ln2.', '.ln.'): v
              for k, v in src['model'].items()}
        model.load_state_dict(sd, strict=False)
        cal_batches = [get_batch(valid_data, 8, cfg.ctx,
                                 np.random.default_rng(11 + i), device)[0]
                       for i in range(8)]
        rep = calibrate(model, cal_batches)
        print('calibrated thresholds:',
              [(li, kind, f'{want}->{got}') for li, kind, want, got in rep[:6]],
              '...', flush=True)

        # --- three shadow evaluations BEFORE optimizer step 1 (M2.md) ---
        # a blend schedule starting at alpha=0 would trivially reproduce the
        # float parent and zero the threshold gradients by construction; the
        # meaningful test of calibration is the HARD alpha=1 shadow.
        def _set_blend_all(a):
            from whitebox.model import SpikeProx, SignedProx
            for mm in model.modules():
                if isinstance(mm, (SpikeProx, SignedProx)):
                    mm.blend = a

        rng7 = lambda: np.random.default_rng(7)
        _set_blend_all(0.0)
        ppl_float = evaluate(model, valid_data, args.batch, cfg.ctx, device,
                             iters=10, rng=rng7())
        _set_blend_all(1.0)
        ppl_hard = evaluate(model, valid_data, args.batch, cfg.ctx, device,
                            iters=10, rng=rng7())
        xm0, _ = get_batch(valid_data, 8, cfg.ctx,
                           np.random.default_rng(3), device)
        m0 = model.layer_metrics(xm0)
        rates = [1 - mm['sparsity'] for mm in m0]
        ents = [mm['entropy'] for mm in m0 if mm['entropy'] is not None]
        ers = [mm['err_rate'] for mm in m0 if mm['err_rate'] is not None]
        # probe backward at alpha=1 for threshold-gradient coverage; no
        # optimizer update, gradients cleared afterwards
        xb, yb = get_batch(valid_data, args.batch, cfg.ctx,
                           np.random.default_rng(13), device)
        _, probe_loss = model(xb, yb)
        probe_loss.backward()
        thr = [(n2, float(p.grad.abs().sum()))
               for n2, p in model.named_parameters()
               if 'log_threshold' in n2 and p.grad is not None]
        cov = sum(g > 0 for _, g in thr)
        opt.zero_grad(set_to_none=True)
        a0 = min(1.0, 0 / args.blend_steps) if args.blend_steps else 1.0
        _set_blend_all(a0)
        ppl_sched = evaluate(model, valid_data, args.batch, cfg.ctx, device,
                             iters=10, rng=rng7())
        print(f'SHADOW EVALS (pre-training): float a=0 ppl {ppl_float:.1f} '
              f'(parent 14.22) | HARD a=1 ppl {ppl_hard:.0f} (uncalibrated '
              f'was 4086) | scheduled a={a0:.2f} ppl {ppl_sched:.1f}',
              flush=True)
        print(f'HARD-MODEL SIGNATURE: firing {min(rates):.2f}-{max(rates):.2f} '
              f'(envelope 0.35-0.47) | entropy {min(ents):.2f}-{max(ents):.2f} '
              f'| err-rate {np.mean(ers):.2f} (~0.20) | thr-grad coverage '
              f'{cov}/{len(thr)}', flush=True)

    def set_blend(a):
        from whitebox.model import SpikeProx, SignedProx
        for m in model.modules():
            if isinstance(m, (SpikeProx, SignedProx)):
                m.blend = a

    log = open(run_dir / 'log.jsonl', 'a')
    t0 = time.time()
    for step in range(args.steps):
        if args.blend_steps:
            set_blend(min(1.0, step / args.blend_steps))
        for at_step, lv in anneal:
            if step == at_step:
                model.set_levels(lv)
                print(f'annealed to levels={lv} at step {step}', flush=True)
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
                       thr_grads=[round(float(p.grad.norm()), 5)
                                  for n2, p in model.named_parameters()
                                  if 'log_threshold' in n2 and p.grad is not None],
                       layers=[dict(l=m['layer'],
                                    drc=round(m['rc_after'] - m['rc_before'], 3),
                                    r=round(m['r_total'], 2),
                                    sp=round(m['sparsity'], 3),
                                    **({'ent': m['entropy'], 'mag': m['mag'],
                                        'er': m['err_rate']}
                                       if 'entropy' in m else {}))
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
