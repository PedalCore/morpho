"""The derivation-execution gap autopsy (post-hoc, no training change).

For each layer, delta = attn(z). Measures, per the agreed taxonomy:
  1. g_dir = <grad_z R^c(z;U), delta>  and the alpha-sweep R^c(z + a*delta)
     -> wrong direction vs oversized step vs repurposing
  2. dR = R(z+delta) - R(z)  and  d(R - R^c)
     -> abandoned compression role vs overriding the whole objective
  3. per-head ||W_k W_k^T - I_p||_F
     -> are the subspace-basis assumptions still satisfied?

Runs on M2-control (the finding) and M0 (the healthy-compressor
reference) from their checkpoints.

python3 -m whitebox.autopsy
"""

import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/Users/marycarrigan/coding/morpho-snn/spikelm')

from spikelm.data import load_split, get_batch  # noqa: E402
from whitebox.model import (Config, CausalCRATE, CausalCRATEM2,  # noqa: E402
                            _coding_rate_impl)

EPS_SQ = 0.5
ALPHAS = (0.125, 0.25, 0.5, 1.0)


def expansion(z):
    B, T, d = z.shape
    g = z.transpose(-2, -1) @ z
    eye = torch.eye(d)
    return (torch.logdet(eye + (d / (T * EPS_SQ)) * g) / 2).mean()


def head_orthonormality(attn):
    W = attn.U.weight            # (d, d), head k = rows [kp:(k+1)p]
    devs = []
    for k in range(attn.K):
        Wk = W[k * attn.p:(k + 1) * attn.p]          # (p, d)
        G = Wk @ Wk.t()
        devs.append(float((G - torch.eye(attn.p)).norm()))
    return float(np.mean(devs))


def autopsy_layer(z, attn):
    zg = z.detach().requires_grad_(True)
    rc0 = _coding_rate_impl(zg, attn, EPS_SQ)
    (grad,) = torch.autograd.grad(rc0, zg)
    with torch.no_grad():
        delta = attn(z)
        g_dir = float((grad * delta).sum())
        sweep = [float(_coding_rate_impl(z + a * delta, attn, EPS_SQ) - rc0)
                 for a in ALPHAS]
        dR = float(expansion(z + delta) - expansion(z))
        drc = sweep[-1]
        return dict(g_dir=g_dir, sweep=sweep, drc=drc, dR=dR,
                    d_gap=dR - drc, ortho=head_orthonormality(attn))


@torch.no_grad()
def _states_m2(model, idx):
    z = model._embed(idx)
    for b in model.blocks:
        yield z, b.attn
        z = b(z)


@torch.no_grad()
def _states_m0(model, idx):
    T = idx.shape[1]
    x = model.emb(idx) + model.pos(torch.arange(T))
    for b in model.blocks:
        z = b.ln1(x)
        yield z, b.attn
        x = b.ista(b.ln2(x + b.attn(z)))


def run(name, path, is_m2):
    ck = torch.load(path, map_location='cpu')
    cfg = Config(**ck['cfg'])
    model = (CausalCRATEM2 if is_m2 else CausalCRATE)(cfg)
    model.load_state_dict(ck['model'])
    model.eval()
    x, _ = get_batch(load_split('valid'), 6, cfg.ctx,
                     np.random.default_rng(3), 'cpu')
    print(f'\n== {name} ==')
    print(' L   g_dir      dRc(a=1/8 .. 1)                dR       d(R-Rc)  ortho')
    rows = []
    states = _states_m2(model, x) if is_m2 else _states_m0(model, x)
    for li, (z, attn) in enumerate(states):
        r = autopsy_layer(z, attn)
        rows.append(r)
        sw = ' '.join(f'{s:+7.1f}' for s in r['sweep'])
        print(f'{li:2d} {r["g_dir"]:+9.1f}  [{sw}]  {r["dR"]:+8.1f} '
              f'{r["d_gap"]:+8.1f}  {r["ortho"]:.2f}')
    gd = [r['g_dir'] for r in rows]
    mono = [all(s > 0 for s in r['sweep']) for r in rows]
    print(f'summary: g_dir>0 in {sum(g > 0 for g in gd)}/{len(gd)} layers; '
          f'expands at ALL alpha in {sum(mono)}/{len(rows)}; '
          f'mean d(R-Rc) {np.mean([r["d_gap"] for r in rows]):+.1f}; '
          f'mean ortho dev {np.mean([r["ortho"] for r in rows]):.2f}')


if __name__ == '__main__':
    run('M2-control (the finding)', 'whitebox/runs/m2-control/ckpt.pt', True)
    run('M0 (healthy-compressor reference)',
        'whitebox/runs/crate-d384L12/ckpt.pt', False)
