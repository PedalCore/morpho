"""Surrogate-gradient spiking units (milestone 1) — hard spikes forward,
smooth derivative backward. BICLab-v3 lesson built in: `levels` > 1 gives
integer spike counts (easier optimization), annealed toward binary later.
"""

import torch
import torch.nn as nn


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, levels, signed):
        ctx.save_for_backward(x, threshold)
        ctx.levels = levels
        ctx.signed = signed
        if signed:
            # ternary/signed spikes: a unit may fire negative events too, so a
            # zero-mean signal (e.g. a LayerNorm output) survives spiking.
            # Still multiply-free downstream: add or subtract, never scale.
            z = torch.sign(x) * torch.clamp(torch.floor(x.abs() / threshold), 0, levels)
        else:
            z = torch.clamp(torch.floor(x / threshold), 0, levels)
        return z * threshold

    @staticmethod
    def backward(ctx, grad_out):
        x, threshold = ctx.saved_tensors
        width = threshold * (ctx.levels + 0.5)
        if ctx.signed:
            mask = (x.abs() < width).to(grad_out.dtype)
        else:
            mask = ((x > -0.5 * threshold) & (x < width)).to(grad_out.dtype)
        return grad_out * mask, None, None, None


class SpikeAct(nn.Module):
    """Drop-in for ChannelMix's nonlinearity: integer spikes with PER-CHANNEL
    learnable thresholds (SpikeDecoder's parameterized-LIF lesson: per-unit
    learnable spike parameters were their largest quality lever, +11pp)."""

    def __init__(self, dim, levels=4, init_threshold=0.5, signed=False):
        super().__init__()
        self.levels = levels
        self.signed = signed
        self.log_threshold = nn.Parameter(
            torch.full((dim,), float(init_threshold)).log())
        self.last_rate = None  # populated each forward, for the rate regularizer

    def forward(self, x):
        thr = self.log_threshold.exp()
        z = _SpikeFn.apply(x, thr, self.levels, self.signed)
        self.last_rate = (z.detach() != 0).float().mean()
        return z

    def set_levels(self, levels):  # annealing hook: 4 → 2 → 1
        self.levels = levels


def rate_loss(model, target_rate=0.1):
    """Mean squared excess firing rate across all SpikeAct modules."""
    losses = []
    for m in model.modules():
        if isinstance(m, SpikeAct) and m.last_rate is not None:
            losses.append(torch.square(torch.clamp(m.last_rate - target_rate, min=0)))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()
