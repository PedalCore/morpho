"""M4 probe trainer: five arms, tiny models, the binding sweep grid.

python3 -m whitebox.m4_probe_train --arch {crsa,crsa-sm,cache16,kv} \
        --seed {0,1,2} [--cpu]

Matched protocol across arms: d=128, L=4, K=4, ctx=128, M2-control
wiring, 20k steps, spikelm optimizer recipe (as the locked M3 runs).
Training stream: locked tasks + binding (m4_probes.train_stream).
Eval: the locked M3 grid (gates 1-2) + the M4 binding grid (gates 3-4).
Results append to whitebox/runs/m4probes/results.jsonl.
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.model import Config, CausalCRATEM2            # noqa: E402
from whitebox import probes, m4_probes                      # noqa: E402
from whitebox.probe_train import batch_loss, eval_grid      # noqa: E402

ARCHS = {
    'kv':      dict(attn='mssa'),
    'qkv':     dict(attn='qkv'),   # TRUE ceiling: untied standard attention
    'crsa':    dict(attn='crsa'),
    'crsa-sm': dict(attn='crsa', signed_moment=True),
    'cache16': dict(attn='crsa', local_window=16),
    'cache16-qkv': dict(attn='crsa', local_window=16, local_qkv=True),
    'cache16-qkv-g01': dict(attn='crsa', local_window=16, local_qkv=True,
                            local_gamma_init=0.1),
    'qkv-w16': dict(attn='qkv', window=16),      # local-only: can a 16-token
                                                 # role-separated branch bind
                                                 # inside its window, alone?
    'qkv-w16-reach': dict(attn='qkv', window=16),  # + reachability-aware
                                                   # binding curriculum
    'qk-v':    dict(attn='qkv', qkv_tie='qk'),   # Q=K shared, V separate
    'q-kv':    dict(attn='qkv', qkv_tie='kv'),   # Q separate, K=V shared
    'slots':   dict(attn='slots'),   # rung 3: gate fired (reach arm failed)
    'slots-frozen': dict(attn='slots', slot_frozen_basis=True),  # middle rung
    'slots-own': dict(attn='slots', slot_own_basis=True),  # own k=v basis
    'slots-prev': dict(attn='slots', slot_own_basis=True,
                       slot_prev_route=True),  # v2: route write by owner
    'slots-m4':  dict(attn='slots', slot_own_basis=True,
                      slot_prev_route=True, slot_m=4),
    'slots-m16': dict(attn='slots', slot_own_basis=True,
                      slot_prev_route=True, slot_m=16),
    'slots-prev-shared': dict(attn='slots', slot_prev_route=True),
    # basis recheck ON the working mechanism (v1 verdict confounded)
    'slots-sel':      dict(attn='slots', slot_own_basis=True,
                           slot_owner_sel=True),   # gate 3: learned selector
    'slots-prev-pat': dict(attn='slots', slot_own_basis=True,
                           slot_prev_route=True),  # oracle on pattern grammar
                                                   # (must FAIL post/far)
    'slots-conv-self': dict(attn='slots', slot_own_basis=True,
                            local_conv=4),  # conv transports owner ->
                                            # SELF-routing may bind w/o oracle
    'slots-prev-gated': dict(attn='slots', slot_own_basis=True,
                             slot_prev_route=True, slot_gated=True),
    'slots-prev-g4': dict(attn='slots', slot_own_basis=True,
                          slot_prev_route=True, slot_groups=4),
    'slots-conv-g4': dict(attn='slots', slot_own_basis=True,
                          local_conv=4, slot_groups=4),  # the composition:
                          # learned write path + cheap basis
    'slots-prev-1L': dict(attn='slots', slot_own_basis=True,
                          slot_prev_route=True, slot_layer_set='2'),
    'slots-prev-2L': dict(attn='slots', slot_own_basis=True,
                          slot_prev_route=True, slot_layer_set='1,3'),
}


@torch.no_grad()
def eval_binding(model, device):
    model.eval()
    out = {}
    for tag, ex in m4_probes.eval_sets(n=200).items():
        L = max(len(e[0]) for e in ex)
        x = np.full((len(ex), L), probes.PAD, dtype=np.int64)
        for i, (seq, _s, _a) in enumerate(ex):
            x[i, :len(seq)] = seq
        logits, _ = model(torch.from_numpy(x).to(device)[:, :-1])
        pred = logits.argmax(-1).cpu().numpy()
        correct = total = 0
        for i, (_seq, start, ans) in enumerate(ex):
            for j, a in enumerate(ans):
                correct += int(pred[i, start + j - 1] == a)
                total += 1
        out[tag] = round(correct / total, 4)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=list(ARCHS), required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = ('cpu' if args.cpu else
              'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    cfg = Config(vocab_size=probes.VOCAB, n_layer=4, n_embd=128, n_head=4,
                 ctx=128, m2='b', m2_identity=True, **ARCHS[args.arch])
    model = CausalCRATEM2(cfg).to(device)
    print(f'm4probe {args.arch} s{args.seed}: '
          f'{model.num_params()/1e3:.0f}k params on {device}', flush=True)

    patterns = args.arch in ('slots-sel', 'slots-prev-pat')
    stream = m4_probes.train_stream(
        seed=m4_probes.TRAIN_SEED + args.seed, batch=16,
        reachable=16 if args.arch.endswith('-reach') else 0,
        patterns=patterns)
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
            b = eval_binding(model, device)
            mean_b = float(np.mean(list(b.values())))
            print(f'{args.arch} s{args.seed} step {step:5d} '
                  f'loss {float(loss):.3f} bind-mean {mean_b:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    m3 = eval_grid(model, device)      # locked grid: gates 1-2
    b = eval_binding(model, device)    # binding grid: gates 3-4
    if patterns:                       # gate-3 offset-varied grid
        from whitebox.m4_probes import eval_sets_pat
        import whitebox.m4_probe_train as _self
        ev = eval_sets_pat(n=200)
        for tag, ex in ev.items():
            L = max(len(e[0]) for e in ex)
            xg = np.full((len(ex), L), probes.PAD, dtype=np.int64)
            for i, (seq, _s, _a) in enumerate(ex):
                xg[i, :len(seq)] = seq
            with torch.no_grad():
                lg, _ = model(torch.from_numpy(xg).to(device)[:, :-1])
            pred = lg.argmax(-1).cpu().numpy()
            c = t_ = 0
            for i, (_seq, start, ans) in enumerate(ex):
                for j, aa in enumerate(ans):
                    c += int(pred[i, start + j - 1] == aa)
                    t_ += 1
            b[tag] = round(c / t_, 4)
    rec = dict(arch=args.arch, seed=args.seed, steps=args.steps,
               params=model.num_params(), m3=m3, binding=b)
    outdir = pathlib.Path(__file__).parent / 'runs' / 'm4probes'
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'results.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    torch.save(dict(model=model.state_dict(), cfg=vars(cfg)),
               outdir / f'{args.arch}-s{args.seed}.pt')
    print('FINAL', args.arch, args.seed,
          'induction', [m3[k] for k in m3 if k.startswith('induction')],
          'selective', [m3[k] for k in m3 if k.startswith('selective')],
          flush=True)


if __name__ == '__main__':
    main()
