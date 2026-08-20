"""Paired float/hard shadow autopsy on identical tokens (M2.md).

The calibrated model's blend toggle IS the pair: alpha=0 (float parent
function) vs alpha=1 (hard codes), same weights, same tokens. Measures:

  1. per-layer representation cosine + linear CKA (float vs hard z_l)
  2. per-consumer relative error |W q - W x| / |W x|  (U, D, head)
  3. logit KL(float || hard) + correct-token rank shift
  4. one-layer-at-a-time hard substitution -> ppl per layer
     (does the shock accumulate gradually or localize at one consumer?)

python3 -m whitebox.hard_autopsy whitebox/runs/m2-calibrated/ckpt.pt
"""

import math
import sys
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/Users/marycarrigan/coding/morpho-snn/spikelm')

from spikelm.data import load_split, get_batch  # noqa: E402
from whitebox.model import (Config, CausalCRATEM2,  # noqa: E402
                            SpikeProx, SignedProx)
from whitebox.train import evaluate, pick_device  # noqa: E402


def set_blend(model, a, layer=None):
    """a for all prox sites, or only for block `layer` (others 0)."""
    if model.prox_in is not None:
        model.prox_in.blend = a if layer is None or layer == -1 else 0.0
    for li, b in enumerate(model.blocks):
        aa = a if layer is None or li == layer else 0.0
        if b.prox is not None:
            b.prox.blend = aa
        if b.eprox is not None:
            b.eprox.blend = aa


@torch.no_grad()
def states(model, idx):
    z = model._embed(idx)
    out = [z]
    for b in model.blocks:
        z = b(z)
        out.append(z)
    return out


def cka(x, y):
    x = x.reshape(-1, x.shape[-1]).double()
    y = y.reshape(-1, y.shape[-1]).double()
    x = x - x.mean(0)
    y = y - y.mean(0)
    xy = (x.t() @ y).norm() ** 2
    return float(xy / (((x.t() @ x).norm()) * ((y.t() @ y).norm()) + 1e-12))


def main():
    ckpt = sys.argv[1]
    device = pick_device()
    ck = torch.load(ckpt, map_location=device)
    model = CausalCRATEM2(Config(**ck['cfg'])).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    valid = load_split('valid')
    x, y = get_batch(valid, 8, 256, np.random.default_rng(3), device)

    set_blend(model, 0.0)
    zf = states(model, x)
    lf, _ = model(x)
    set_blend(model, 1.0)
    zh = states(model, x)
    lh, _ = model(x)

    print('1. representation alignment per layer (float vs hard):')
    for li, (a, b) in enumerate(zip(zf, zh)):
        cos = float(F.cosine_similarity(a.reshape(-1, a.shape[-1]),
                                        b.reshape(-1, b.shape[-1]),
                                        dim=-1).mean())
        print(f'   z{li:2d}  cos {cos:.3f}  CKA {cka(a, b):.3f}')

    print('2. per-consumer relative error |Wq-Wx|/|Wx|:')
    for li in (0, 3, 6, 9, 11):
        b = model.blocks[li]
        xf, xh = zf[li], zh[li]
        for name, W in (('U', b.attn.U.weight), ('D', b.D)):
            ef = xf.reshape(-1, xf.shape[-1]) @ W.t()
            eh = xh.reshape(-1, xh.shape[-1]) @ W.t()
            print(f'   L{li:2d} {name}: {float((eh - ef).norm() / (ef.norm() + 1e-9)):.3f}',
                  end='')
        print()
    hf = zf[-1].reshape(-1, zf[-1].shape[-1]) @ model.head.weight.t()
    hh = zh[-1].reshape(-1, zh[-1].shape[-1]) @ model.head.weight.t()
    print(f'   head: {float((hh - hf).norm() / (hf.norm() + 1e-9)):.3f}')

    print('3. logits:')
    pf = F.log_softmax(lf.reshape(-1, lf.shape[-1]), -1)
    ph = F.log_softmax(lh.reshape(-1, lh.shape[-1]), -1)
    kl = float(F.kl_div(ph, pf, log_target=True, reduction='batchmean'))
    yt = y.reshape(-1)
    rank_f = (pf > pf.gather(1, yt[:, None])).sum(1).float().mean()
    rank_h = (ph > ph.gather(1, yt[:, None])).sum(1).float().mean()
    print(f'   KL(float||hard) {kl:.3f}  correct-token mean rank '
          f'{float(rank_f):.1f} -> {float(rank_h):.1f}')

    print('4. one-layer-at-a-time hard substitution (ppl):')
    base = evaluate(model, valid, 16, 256, device, iters=8,
                    rng=np.random.default_rng(7))  # still alpha=1 everywhere
    set_blend(model, 0.0)
    ppl_f = evaluate(model, valid, 16, 256, device, iters=8,
                     rng=np.random.default_rng(7))
    print(f'   all-float {ppl_f:.1f} | all-hard {base:.1f}')
    for li in [-1] + list(range(len(model.blocks))):
        set_blend(model, 0.0)
        set_blend(model, 1.0, layer=li)
        p = evaluate(model, valid, 16, 256, device, iters=8,
                     rng=np.random.default_rng(7))
        nm = 'emb' if li == -1 else f'L{li}'
        print(f'   hard@{nm}: {p:.1f} (+{p - ppl_f:.1f})')
    set_blend(model, 1.0)


if __name__ == '__main__':
    main()
