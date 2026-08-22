"""Slot-health + gradient-conflict diagnostics on a SlotCRSA checkpoint.

Distinguishes "complex mechanism still organizing" from "mixed loss
falling while slots do nothing" (collaborator protocol):
  - write-routing entropy + per-slot occupancy (dead/dominant slots)
  - slot-value divergence (do slots hold distinct content?)
  - cos(g_binding, g_locked) on each layer's shared U: negative =>
    objective conflict; ~0/positive with flat binding => routing/read
    optimization is the blocker instead.

python3 -m whitebox.m4_slotdiag whitebox/runs/m4probes/slots-s0.pt
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.model import Config, CausalCRATEM2       # noqa: E402
from whitebox import probes, m4_probes                 # noqa: E402
from whitebox.probe_train import batch_loss            # noqa: E402


def grad_on_U(model, x, mask):
    model.zero_grad(set_to_none=True)
    loss = batch_loss(model, x, mask, 'cpu')
    loss.backward()
    return [b.attn.U.weight.grad.clone() for b in model.blocks]


def main():
    ck = torch.load(sys.argv[1], map_location='cpu')
    cfg = Config(**ck['cfg'])
    model = CausalCRATEM2(cfg)
    model.load_state_dict(ck['model'])
    model.train()

    # --- routing health on a binding batch ---
    rng = np.random.default_rng(123)
    ex = [m4_probes.make_binding(rng, facts=4, gap=8) for _ in range(64)]
    L = max(len(e[0]) for e in ex)
    x = np.full((64, L), probes.PAD, dtype=np.int64)
    for i, (seq, _s, _a) in enumerate(ex):
        x[i, :len(seq)] = seq
    xt = torch.from_numpy(x)
    with torch.no_grad():
        for li, b in enumerate(model.blocks):
            a = b.attn
            h = model._embed(xt) if li == 0 else None  # layer-0 view only
            if h is None:
                continue
            kv = (a.Ukv(h) if a.Ukv is not None else
                  F.linear(h, a.U.weight.detach()) if a.frozen else a.U(h))
            tau = F.softplus(a.log_tau) + 0.1
            w = torch.softmax(kv @ a.slot_keys.t() /
                              (tau * cfg.n_embd ** 0.5), -1)
            ent = float((-w * (w + 1e-9).log()).sum(-1).mean())
            occ = w.mean(dim=(0, 1))
            keys = F.normalize(a.slot_keys, dim=-1)
            key_cos = float((keys @ keys.t()).triu(1).abs().max())
            print(f'L0 routing: entropy {ent:.3f}/{np.log(a.M):.3f} max, '
                  f'occupancy {[round(float(o), 3) for o in occ]}, '
                  f'max |key cos| {key_cos:.3f}, tau {float(tau):.3f}')

    # --- gradient conflict on shared U ---
    def batch_from(stream):
        return next(stream)

    bind_stream = m4_probes.train_stream(seed=999)
    # binding-only batch: rebuild manually
    exb = [m4_probes.make_binding(rng, int(rng.choice([2, 4, 8])),
                                  int(rng.integers(4, 60)), 0)
           for _ in range(16)]
    Lb = max(len(e[0]) for e in exb)
    xb = np.full((16, Lb), probes.PAD, dtype=np.int64)
    mb = np.zeros((16, Lb), dtype=bool)
    for i, (seq, start, ans) in enumerate(exb):
        xb[i, :len(seq)] = seq
        mb[i, start:start + len(ans)] = True
    exl = [probes.sample(rng) for _ in range(16)]
    Ll = max(len(e[0]) for e in exl)
    xl = np.full((16, Ll), probes.PAD, dtype=np.int64)
    ml = np.zeros((16, Ll), dtype=bool)
    for i, (seq, start, ans, _n, _d) in enumerate(exl):
        xl[i, :len(seq)] = seq
        ml[i, start:start + len(ans)] = True

    gb = grad_on_U(model, xb, mb)
    gl = grad_on_U(model, xl, ml)
    for li, (a, b) in enumerate(zip(gb, gl)):
        cos = float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        print(f'layer {li}: cos(g_binding, g_locked) on U = {cos:+.3f}  '
              f'|g_b| {float(a.norm()):.2e}  |g_l| {float(b.norm()):.2e}')


if __name__ == '__main__':
    main()
