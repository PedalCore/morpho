"""Calibrated conversion: initialize M2 spike thresholds from the
M2-control checkpoint's actual pre-consumer activation distributions.

Per layer and channel, set thr so that P(v > thr) matches a target
firing-rate envelope (default: the fluent M1 per-layer rates), keeping
saturation rare. Used by train.py --calibrate-from; combined with
--blend-steps (float->quantized alpha ramp 0->1) and only-then precision
coarsening, per the protocol amendment in M2.md.
"""

import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# fluent M1 per-layer firing rates (1 - sparsity), from crate-spike-warm logs
M1_RATE_ENVELOPE = [0.47, 0.38, 0.39, 0.35, 0.36, 0.27,
                    0.25, 0.22, 0.17, 0.16, 0.13, 0.15]
ERR_RATE_TARGET = 0.20        # sparse error codes, saturation rare


@torch.no_grad()
def calibrate(model, batches, rate_envelope=None, err_rate=ERR_RATE_TARGET):
    """Walk the M2 forward pass capturing each prox's input distribution;
    set per-channel log-thresholds at the (1 - target_rate) quantile."""
    rates = rate_envelope or M1_RATE_ENVELOPE
    dev = next(model.parameters()).device
    caps = {'in': []}
    for li in range(len(model.blocks)):
        caps[f'v{li}'] = []
        caps[f'r{li}'] = []
    for idx in batches:
        idx = idx.to(dev)
        T = idx.shape[1]
        x = model.ln_in(model.emb(idx) +
                        model.pos(torch.arange(T, device=dev)))
        caps['in'].append(x.reshape(-1, x.shape[-1]).cpu())
        z = model.prox_in(x) if model.prox_in is not None else x
        for li, b in enumerate(model.blocks):
            xm = z + b.attn(z)
            u = b.ln(xm)
            r = u - z @ b.D.t()
            caps[f'r{li}'].append(r.reshape(-1, r.shape[-1]).cpu())
            if b.eprox is not None:
                r = b.eprox(r)
            v = z + b.eta * (r @ b.D)
            caps[f'v{li}'].append(v.reshape(-1, v.shape[-1]).cpu())
            z = b.prox(v) if b.prox is not None else v

    def set_thr(prox, samples, rate, signed=False):
        vals = torch.cat(samples)                       # (N, d)
        src = vals.abs() if signed else vals
        q = torch.quantile(src.float(), 1.0 - rate, dim=0)
        q = torch.clamp(q, min=1e-3)
        prox.log_threshold.copy_(q.log().to(dev))
        got = ((src > q.cpu()).float().mean(0)).mean()
        return float(got)

    if model.prox_in is not None:
        set_thr(model.prox_in, caps['in'], rates[0])
    report = []
    for li, b in enumerate(model.blocks):
        if b.prox is not None:
            got = set_thr(b.prox, caps[f'v{li}'], rates[li])
            report.append((li, 'prox', rates[li], round(got, 3)))
        if b.eprox is not None:
            got = set_thr(b.eprox, caps[f'r{li}'], err_rate, signed=True)
            report.append((li, 'eprox', err_rate, round(got, 3)))
    return report
