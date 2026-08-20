"""Branch micro-experiment: is transition injury calibration or schedule?

From a levels-2-adapted checkpoint (recreated pre-2->1 state), two
matched 300-step arms after switching to binary:

  A. fixed-threshold 2->1   (the ladder's treatment)
  B. recalibrated  2->1     (per-channel thresholds set against the
                             POST-transition distributions, layer by
                             layer, targeting pre-transition firing)

Reports ppl at 0/100/200/300 steps + first-post-switch per-layer
rate/entropy/err for both arms.

python3 -m whitebox.branch_experiment <branch_ckpt>
"""

import copy
import math
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/Users/marycarrigan/coding/morpho-snn/spikelm')

from spikelm.data import load_split, get_batch  # noqa: E402
from whitebox.model import Config, CausalCRATEM2  # noqa: E402
from whitebox.calibrate import calibrate  # noqa: E402
from whitebox.train import evaluate, pick_device  # noqa: E402


def measured_rates(model, batches):
    """Per-layer firing rates under the CURRENT levels."""
    dev = next(model.parameters()).device
    rates = None
    n = 0
    for idx in batches:
        mets = model.layer_metrics(idx.to(dev))
        r = np.array([1 - m['sparsity'] for m in mets])
        rates = r if rates is None else rates + r
        n += 1
    return (rates / n).tolist()


def train_arm(model, steps, device, valid, train, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5,   # tail-of-cosine lr
                            betas=(0.9, 0.99), weight_decay=0.01)
    rng = np.random.default_rng(99)
    marks = {}
    for step in range(steps + 1):
        if step in (0, 100, 200, 300):
            ppl = evaluate(model, valid, 16, 256, device, iters=10,
                           rng=np.random.default_rng(7))
            marks[step] = round(ppl, 1)
            if step == 0:
                xm, _ = get_batch(valid, 8, 256, np.random.default_rng(3),
                                  device)
                mets = model.layer_metrics(xm)
                print(f'  {tag} first-batch layers: ' + ' '.join(
                    f"L{m['layer']}:sp{m['sparsity']:.2f}/e{m['entropy']}"
                    for m in mets[::3]), flush=True)
        if step == steps:
            break
        x, y = get_batch(train, 16, 256, rng, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    print(f'  {tag}: ' + ' -> '.join(f'{s}:{p}' for s, p in marks.items()),
          flush=True)
    return marks


def main():
    ckpt_path = sys.argv[1]
    device = pick_device()
    valid, train = load_split('valid'), load_split('train')
    ck = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ck['cfg'])

    def fresh():
        m = CausalCRATEM2(cfg).to(device)
        m.load_state_dict(ck['model'])
        m.set_levels(2)
        return m

    base = fresh()
    cal_batches = [get_batch(valid, 8, cfg.ctx,
                             np.random.default_rng(11 + i), device)[0]
                   for i in range(6)]
    targets = measured_rates(base, cal_batches[:3])
    ppl2 = evaluate(base, valid, 16, cfg.ctx, device, iters=10,
                    rng=np.random.default_rng(7))
    print(f'branch point: levels=2 ppl {ppl2:.1f}, firing targets '
          f'{[round(t, 2) for t in targets[::3]]}', flush=True)

    print('ARM A — fixed-threshold 2->1:', flush=True)
    a = fresh()
    a.set_levels(1)
    train_arm(a, 300, device, valid, train, 'A')

    print('ARM B — recalibrated 2->1:', flush=True)
    b = fresh()
    b.set_levels(1)
    calibrate(b, cal_batches, rate_envelope=targets)
    train_arm(b, 300, device, valid, train, 'B')


if __name__ == '__main__':
    main()
