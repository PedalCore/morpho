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
    nlm: bool = False      # milestone 4a: CTM-style neuron-level temporal models
    nlm_k: int = 4         # history window each unit sees
    sync: bool = False     # milestone 4b: CTM-style synchronization representation
    sync_pairs: int = 512  # how many unit pairs are watched
    sync_k: int = 32       # temporal window of the synchrony trace


def token_shift(x):
    return F.pad(x, (0, 0, 1, -1))


class NeuronTime(nn.Module):
    """CTM idea #1: neuron-level temporal processing — every unit gets its
    OWN weights over its OWN recent history, instead of the architecture's
    single fixed token-shift. Implemented as a causal depthwise convolution
    (one k-tap filter per channel), initialized to reproduce token_shift
    exactly, so an NLM model starts life identical to the baseline and any
    difference is learned rather than granted.
    """

    def __init__(self, C, k=4):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(C, C, kernel_size=k, groups=C, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, -2] = 1.0  # tap on t-1 == token_shift

    def forward(self, x):  # (B, T, C)
        y = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return self.conv(y).transpose(1, 2)


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
        self.shift = NeuronTime(C, cfg.nlm_k) if cfg.nlm else token_shift

    def forward(self, x):
        xs = self.shift(x)
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
        self.shift = NeuronTime(C, cfg.nlm_k) if cfg.nlm else token_shift

    def forward(self, x):
        xs = self.shift(x)
        k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
        if self.spike_act is not None:
            k = self.spike_act(k)
        else:
            k = torch.square(torch.relu(k))
        r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))
        return r * self.value(k)


class SyncReadout(nn.Module):
    """CTM idea #2: SYNCHRONIZATION as representation. Instead of reading the
    model's state off individual unit activations, watch whether pairs of
    units are active TOGETHER, and how persistently. For a random fixed set
    of unit pairs we track an exponentially-decaying trace of their product
    — a streaming co-activation measure, causal by construction — and add a
    projection of those traces to the representation the output head sees.

    Our own reservoir ablation pointed here independently: destroying the
    temporal ALIGNMENT of spike traces cost more than removing them, and
    pairwise trace products beat per-unit rates at matched parameter count.

    The output projection is zero-initialized, so at step 0 the model is
    exactly the model without it: any gain must be learned.
    """

    def __init__(self, C, n_pairs=512, k=32, seed=1234):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("ia", torch.randint(0, C, (n_pairs,), generator=g))
        self.register_buffer("ib", torch.randint(0, C, (n_pairs,), generator=g))
        # spread initial persistence: some pairs watch the last token, some ~30
        self.decay_raw = nn.Parameter(torch.linspace(-2.0, 3.0, n_pairs))
        self.k = k
        self.proj = nn.Linear(n_pairs, C, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):                      # (B, T, C), already normalized
        pair = x[..., self.ia] * x[..., self.ib]           # (B, T, P) co-activation
        lam = torch.sigmoid(self.decay_raw).clamp(0.05, 0.995)
        j = torch.arange(self.k - 1, -1, -1, device=x.device, dtype=x.dtype)
        kern = lam.unsqueeze(1) ** j                        # (P, k), newest tap last
        y = F.pad(pair.transpose(1, 2), (self.k - 1, 0))
        s = F.conv1d(y, kern.unsqueeze(1), groups=kern.size(0)).transpose(1, 2)
        return self.proj(s)


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
        self.sync = SyncReadout(cfg.n_embd, cfg.sync_pairs, cfg.sync_k) if cfg.sync else None
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tied
        self.apply(self._init)
        if self.sync is not None:
            # AFTER apply(): the generic Linear init would otherwise overwrite
            # the zeros, and the sync path must start as an exact no-op
            nn.init.zeros_(self.sync.proj.weight)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.ln_in(self.emb(idx))
        for b in self.blocks:
            x = b(x)
        x = self.ln_out(x)
        if self.sync is not None:
            x = x + self.sync(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
