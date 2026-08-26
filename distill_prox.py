"""M6 bridge experiment 1 — paired-atom prox distillation (M6.md).

Teacher: the working delta + linear-dictionary LM (screen-m5diag-ista,
9.97 @3k). Student: the SAME network with a PairedProx appended to
each block's output — exactly the identity at tau ~ 0, so the student
IS the teacher at step 0 (verified before training).

Phase A (bridge): freeze everything except prox atoms + thresholds;
ramp tau on a schedule; per-layer alignment loss against the frozen
teacher (relative MSE + cosine — magnitude matters to the delta
memory's writes). The ramp HALTS whenever any layer's relative error
exceeds tol, and resumes when realignment brings it back under.

Phase B (joint): unfreeze all; CE + decaying teacher-alignment.

Interpretations (preregistered in M6.md): success => the scratch-prox
failures were reachability; aligns frozen but collapses in phase B =>
genuine co-adaptation conflict; cannot align at tau=0 => bug.

python3 -m whitebox.distill_prox [--steps-a 1500] [--steps-b 3000]
"""

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
SPIKELM = '/Users/marycarrigan/coding/morpho-snn/spikelm'
sys.path.insert(0, SPIKELM)

from spikelm.data import load_split, get_batch                  # noqa: E402
from whitebox.model import Config, CausalCRATEM2, PairedProx    # noqa: E402

TEACHER = 'whitebox/runs/screen-m5diag-ista/ckpt.pt'
TOL = 0.05          # per-layer relative error gate for the tau ramp
TAU_MAX = 0.10      # target threshold (teacher-scale units)


class Student(nn.Module):
    """Teacher backbone + per-block PairedProx on the block output."""

    def __init__(self, cfg):
        super().__init__()
        self.core = CausalCRATEM2(cfg)
        self.prox = nn.ModuleList(PairedProx(cfg.n_embd)
                                  for _ in range(cfg.n_layer))

    def blocks_forward(self, idx):
        """Returns per-block outputs (student) for alignment."""
        z = self.core._embed(idx)
        outs = []
        for b, p in zip(self.core.blocks, self.prox):
            z = p(b(z))
            outs.append(z)
        return z, outs

    def forward(self, idx, targets=None):
        z, _ = self.blocks_forward(idx)
        logits = self.core.head(z)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def teacher_blocks(teacher, idx):
    z = teacher._embed(idx)
    outs = []
    for b in teacher.blocks:
        z = b(z)
        outs.append(z)
    return outs


def align_loss(s_outs, t_outs):
    per_layer = []
    for s, t in zip(s_outs, t_outs):
        rel = ((s - t) ** 2).sum() / ((t ** 2).sum() + 1e-8)
        cos = 1 - F.cosine_similarity(
            s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]),
            dim=-1).mean()
        per_layer.append((rel, cos))
    loss = sum(r + c for r, c in per_layer)
    max_rel = max(float(r) for r, _ in per_layer)
    return loss, max_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps-a', type=int, default=1500)
    ap.add_argument('--steps-b', type=int, default=3000)
    args = ap.parse_args()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    ck = torch.load(TEACHER, map_location='cpu')
    cfg = Config(**ck['cfg'])
    teacher = CausalCRATEM2(cfg)
    teacher.load_state_dict(ck['model'])
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = Student(cfg)
    student.core.load_state_dict(ck['model'])
    student.to(device)

    train, valid = load_split('train'), load_split('valid')
    rng = np.random.default_rng(42)

    # sanity: identity at init
    x, y = get_batch(valid, 8, 256, rng, device)
    with torch.no_grad():
        _, s_outs = student.blocks_forward(x)
        t_outs = teacher_blocks(teacher, x)
        _, max_rel = align_loss(s_outs, t_outs)
    print(f'init identity check: max per-layer rel err {max_rel:.2e}',
          flush=True)
    assert max_rel < 1e-6, 'not function-preserving at init'

    # ---- Phase A: bridge (prox params only, tau ramp gated on error)
    prox_params = list(student.prox.parameters())
    opt = torch.optim.AdamW(prox_params, lr=3e-4, weight_decay=0.0)
    tau_now = 1e-9
    t0 = time.time()
    for step in range(args.steps_a):
        frac = min(1.0, step / max(1, args.steps_a * 0.8))
        tau_target = TAU_MAX * frac
        x, y = get_batch(train, 16, 256, rng, device)
        _, s_outs = student.blocks_forward(x)
        with torch.no_grad():
            t_outs = teacher_blocks(teacher, x)
        loss, max_rel = align_loss(s_outs, t_outs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        # gated tau ramp: raise floors only while aligned
        if max_rel < TOL and tau_target > tau_now:
            tau_now = tau_target
            with torch.no_grad():
                for p in student.prox:
                    p.log_tau.clamp_(min=math.log(max(tau_now, 1e-9)))
        if step % 150 == 0 or step == args.steps_a - 1:
            rates = [float(p.last_rate) for p in student.prox]
            print(f'A {step:4d} align {float(loss):.4f} maxrel {max_rel:.4f} '
                  f'tau {tau_now:.4f} act {np.mean(rates):.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)

    # ---- Phase B: joint fine-tune (CE + decaying alignment)
    opt = torch.optim.AdamW(student.parameters(), lr=1.5e-4,
                            betas=(0.9, 0.99), weight_decay=0.01)
    for step in range(args.steps_b):
        lr = 1.5e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(
            math.pi * step / args.steps_b)))
        for g in opt.param_groups:
            g['lr'] = lr
        x, y = get_batch(train, 16, 256, rng, device)
        logits, ce = student(x, y)
        _, s_outs = student.blocks_forward(x)
        with torch.no_grad():
            t_outs = teacher_blocks(teacher, x)
        al, max_rel = align_loss(s_outs, t_outs)
        w = 0.5 * max(0.0, 1 - step / (args.steps_b * 0.5))
        loss = ce + w * al
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == args.steps_b - 1:
            with torch.no_grad():
                tot = n = 0
                for _ in range(25):
                    xv, yv = get_batch(valid, 16, 256, rng, device)
                    _, l = student(xv, yv)
                    tot += float(l) * yv.numel(); n += yv.numel()
            rates = [float(p.last_rate) for p in student.prox]
            print(f'B {step:4d} ce {float(ce):.3f} val_ppl '
                  f'{math.exp(tot/n):7.3f} act {np.mean(rates):.3f} '
                  f'maxrel {max_rel:.4f} ({time.time()-t0:.0f}s)',
                  flush=True)

    out = pathlib.Path('whitebox/runs/distill-prox')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=student.state_dict(), cfg=vars(cfg)),
               out / 'ckpt.pt')
    print('DISTILL-PROX-DONE', flush=True)


if __name__ == '__main__':
    main()
