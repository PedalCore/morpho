"""RWKV-mini: the milestone-0 conventional baseline.

RWKV-v4-style blocks — token-shift time-mix with the numerically stable
wkv recurrence, squared-ReLU channel-mix — small enough to train on MPS.
The wkv scan is a Python-level loop over T; bench.py decides whether that
is acceptable before any training-time claims are made.

Milestone 1 swaps activations for spiking.SpikeAct units; the block
structure is written so that swap is a config flag, not a rewrite.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    n_layer: int = 6
    n_embd: int = 384
    ctx: int = 256
    spiking: bool = False  # milestone 1


def token_shift(x):
    return F.pad(x, (0, 0, 1, -1))


class TimeMix(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        C = cfg.n_embd
        ratio = layer_id / max(1, cfg.n_layer - 1)
        self.mix_k = nn.Parameter(torch.pow(torch.linspace(0, 1, C), 1 - ratio).unsqueeze(0).unsqueeze(0))
        self.mix_v = nn.Parameter(self.mix_k.data.clone())
        self.mix_r = nn.Parameter(self.mix_k.data.clone())
        decay = -5 + 8 * torch.pow(torch.linspace(0, 1, C), 0.7 + 1.3 * ratio)
        self.time_decay = nn.Parameter(decay)
        self.time_first = nn.Parameter(torch.full((C,), 0.5))
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

    def forward(self, x):
        xs = token_shift(x)
        k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
        v = self.value(x * self.mix_v + xs * (1 - self.mix_v))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
        B, T, C = x.shape
        w = -torch.exp(self.time_decay)  # negative decay per channel
        u = self.time_first
        aa = torch.zeros(B, C, device=x.device)
        bb = torch.zeros(B, C, device=x.device)
        pp = torch.full((B, C), -1e38, device=x.device)
        out = torch.empty(B, T, C, device=x.device)
        for t in range(T):
            kt, vt = k[:, t], v[:, t]
            ww = u + kt
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            out[:, t] = (e1 * aa + e2 * vt) / (e1 * bb + e2 + 1e-9)
            ww2 = pp + w
            p2 = torch.maximum(ww2, kt)
            e1 = torch.exp(ww2 - p2)
            e2 = torch.exp(kt - p2)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = p2
        return self.output(r * out)


class ChannelMix(nn.Module):
    def __init__(self, cfg, layer_id, spiking=False):
        super().__init__()
        C = cfg.n_embd
        spike_act = None
        if spiking:
            from .spiking import SpikeAct

            spike_act = SpikeAct(dim=4 * C)  # per-block, per-channel thresholds
        ratio = layer_id / max(1, cfg.n_layer - 1)
        self.mix_k = nn.Parameter(torch.pow(torch.linspace(0, 1, C), 1 - ratio).unsqueeze(0).unsqueeze(0))
        self.mix_r = nn.Parameter(self.mix_k.data.clone())
        self.key = nn.Linear(C, 4 * C, bias=False)
        self.value = nn.Linear(4 * C, C, bias=False)
        self.receptance = nn.Linear(C, C, bias=False)
        self.spike_act = spike_act  # None → squared ReLU (conventional)

    def forward(self, x):
        xs = token_shift(x)
        k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
        if self.spike_act is not None:
            k = self.spike_act(k)
        else:
            k = torch.square(torch.relu(k))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
        return r * self.value(k)


class Block(nn.Module):
    def __init__(self, cfg, layer_id, spiking=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.tm = TimeMix(cfg, layer_id)
        self.cm = ChannelMix(cfg, layer_id, spiking)

    def forward(self, x):
        x = x + self.tm(self.ln1(x))
        x = x + self.cm(self.ln2(x))
        return x


class RWKVMini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.ln_in = nn.LayerNorm(cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg, i, cfg.spiking) for i in range(cfg.n_layer))
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tied
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.ln_in(self.emb(idx))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_out(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
