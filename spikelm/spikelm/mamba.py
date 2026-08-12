"""Mamba-mini: an S6 selective state-space model, parameter-matched to RWKV-mini.

Built for the hardware comparison, not for novelty: both architectures reduce
to a diagonal linear recurrence with a decay of the form exp(negative), so a
single Morpho cell could generate either. The differences that matter for a
circuit:

  RWKV wkv   3 scalar registers per channel, fixed decay, DIVIDES (normalizer)
  Mamba S6   N-vector state per channel, INPUT-DEPENDENT decay, no division

A is parameterized as -exp(A_log) and dt as softplus(...) > 0, so dt*A <= 0
always — the same "every exponent argument is non-positive" property that made
the wkv exponential a barrel shift plus a table.

expand=4 at d=384, 6 layers matches RWKV-mini's 13.1M parameters exactly.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    vocab_size: int = 4096
    n_layer: int = 6
    n_embd: int = 384
    ctx: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 4


def selective_scan(u, dt, A, B, C, D):
    """h <- exp(dt*A)·h + (dt·B)·u ; y = <h, C> + D·u   (sequential, T loop)."""
    Bsz, T, di = u.shape
    N = A.shape[1]
    h = torch.zeros(Bsz, di, N, device=u.device, dtype=u.dtype)
    ys = torch.empty(Bsz, T, di, device=u.device, dtype=u.dtype)
    for t in range(T):
        dtt = dt[:, t].unsqueeze(-1)                     # B,di,1
        dA = torch.exp(dtt * A)                          # B,di,N   (A<0, dt>0)
        dBu = dtt * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1)
        h = dA * h + dBu
        ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)
    return ys + u * D


class MambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, N = cfg.n_embd, cfg.d_state
        di = cfg.expand * d
        self.di, self.N, self.d_conv = di, N, cfg.d_conv
        self.dt_rank = max(1, d // 16)
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, cfg.d_conv, groups=di, padding=cfg.d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * N, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di, bias=True)
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))          # A = -exp(A_log) < 0
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)
        with torch.no_grad():                            # dt starts in [0.001, 0.1]
            dt = torch.exp(torch.rand(di) * (torch.log(torch.tensor(0.1))
                                             - torch.log(torch.tensor(1e-3)))
                           + torch.log(torch.tensor(1e-3)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm(x)
        xin, z = self.in_proj(h).chunk(2, dim=-1)
        xc = self.conv(xin.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xc = F.silu(xc)
        dbc = self.x_proj(xc)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                # > 0
        A = -torch.exp(self.A_log)                       # < 0
        y = selective_scan(xc, dt, A, Bm, Cm, self.D)
        return x + self.out_proj(y * F.silu(z))


class MambaMini(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(MambaBlock(cfg) for _ in range(cfg.n_layer))
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_out(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
