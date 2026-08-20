"""Causal CRATE — a white-box transformer LM, with a spiking-prox variant.

The two block operators follow Yu et al., "White-Box Transformers via
Sparse Rate Reduction" (arXiv:2306.01129; JMLR version 2311.13110 has the
causal GPT-style variant), transcribed from the paper's equations:

  MSSA (compression, eq. 10-12): per-head SSA with query = key = value =
    U_k* Z — ONE shared projection per head — heads aggregated by [U_1..U_K]
    (weight-tied to the input projection when `tied=True`), applied as a
    residual step.

  ISTA (sparsification, eq. 17): one proximal-gradient step of sparse
    coding against a dictionary D, REPLACING the representation (not a
    residual):  Z' = ReLU(Z + eta * D^T (Z - D Z) - eta*lambda).

The spiking variant swaps ISTA's soft-threshold + ReLU for an INTEGER
SPIKE QUANTIZER:  Z' = clamp(floor(v / thr), 0, levels) * thr  with v =
Z + eta * D^T (Z - D Z).  This is not a bolted-on approximation: the spike
operator is itself the proximal operator of a sparsity penalty plus an
integer-grid constraint, so each layer still takes a prox step in the same
alternating scheme — and whether it still *works* is measured, not assumed
(per-layer coding rate + sparsity, below).  Unlike spikelm's SpikeAct, the
threshold here RECEIVES GRADIENT (the frozen-threshold finding).

Deviations from the strict derivation, stated plainly:
  - pre-LayerNorm before each block (the released CRATE code does this);
  - the MSSA step size kappa*p/(N*eps^2) is folded into a learnable
    per-layer output scale rather than kept as the derived constant;
  - causal masking restricts compression to past tokens - the derivation
    is for full attention; the causal JMLR variant makes the same move.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    n_layer: int = 12          # CRATE blocks are ~3d^2 params vs RWKV ~13d^2
    n_embd: int = 384
    n_head: int = 8
    ctx: int = 256
    tied: bool = True          # aggregate heads with U^T (the derivation)
    mssa_scale: float = 0.1    # init of the learnable step size — the derived
                               # kappa*p/(N*eps^2) is SMALL; 1.0 collapses L0
    ista_eta: float = 0.1
    ista_lambda: float = 0.1
    spike_prox: bool = False   # integer-quantizer prox instead of soft-thresh
    spike_levels: int = 4
    spike_init_threshold: float = 0.5


class SpikeProx(nn.Module):
    """prox of lambda*||a||_0 + integer-grid constraint: a hard quantizer.

    Forward: z = clamp(floor(v / thr), 0, L) * thr, per-channel learnable
    thr = exp(log_threshold).  Gradients: straight-through mask to v (as
    spikelm.spiking), and the threshold trains through the n * thr product
    (n detached) - the gradient path spikelm's SpikeAct lacked.
    """

    def __init__(self, dim, levels=4, init_threshold=0.5):
        super().__init__()
        self.levels = levels
        self.log_threshold = nn.Parameter(
            torch.full((dim,), float(init_threshold)).log())
        self.last_rate = None

    def forward(self, v):
        thr = self.log_threshold.exp()
        n = torch.clamp(torch.floor(v / thr), 0, self.levels)
        self.last_rate = (n.detach() != 0).float().mean()
        # value: n*thr;  grad wrt v: STE mask;  grad wrt thr: through n*thr
        mask = ((v > -0.5 * thr) & (v < thr * (self.levels + 0.5))).to(v.dtype)
        ste = v * mask
        return n.detach() * thr + (ste - ste.detach())


class MSSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, K = cfg.n_embd, cfg.n_head
        self.K, self.p = K, d // K
        self.U = nn.Linear(d, d, bias=False)          # rows = U* stacked
        self.tied = cfg.tied
        if not cfg.tied:
            self.out = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(cfg.mssa_scale))
        mask = torch.triu(torch.full((cfg.ctx, cfg.ctx), float('-inf')), 1)
        self.register_buffer('causal', mask)

    def forward(self, x):                              # (B, T, d)
        B, T, d = x.shape
        h = self.U(x).view(B, T, self.K, self.p).transpose(1, 2)  # B,K,T,p
        att = (h @ h.transpose(-2, -1)) / math.sqrt(self.p)
        att = torch.softmax(att + self.causal[:T, :T], dim=-1)
        ssa = (att @ h).transpose(1, 2).reshape(B, T, d)
        if self.tied:
            agg = F.linear(ssa, self.U.weight.t())     # [U_1..U_K] concat
        else:
            agg = self.out(ssa)
        return self.scale * agg


class ISTA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.n_embd
        self.D = nn.Parameter(torch.empty(d, d))
        nn.init.orthogonal_(self.D)                    # incoherent-ish init
        self.eta = cfg.ista_eta
        self.lam = cfg.ista_lambda
        self.prox = (SpikeProx(d, cfg.spike_levels, cfg.spike_init_threshold)
                     if cfg.spike_prox else None)
        self.last_rate = None

    def forward(self, x):                              # (B, T, d)
        Dx = x @ self.D.t()                            # D z  (rows)
        v = x + self.eta * ((x - Dx) @ self.D)         # z + eta D^T(z - Dz)
        if self.prox is not None:
            z = self.prox(v)
            self.last_rate = self.prox.last_rate
        else:
            z = torch.relu(v - self.eta * self.lam)    # soft-threshold
            self.last_rate = (z.detach() != 0).float().mean()
        return z


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = MSSA(cfg)
        self.ista = ISTA(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                 # compression step
        x = self.ista(self.ln2(x))                     # sparsification REPLACES
        return x


class CausalCRATE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.ctx, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        x = self.ln_out(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss

    # ---------------- white-box instrumentation ----------------

    @torch.no_grad()
    def layer_metrics(self, idx, eps_sq=0.5):
        """The white-box check: per layer, does the MSSA step reduce the
        compression term R^c(Z;U), and how sparse is the ISTA output?
        Returns [{layer, rc_before, rc_after, sparsity}] on one batch."""
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device))
        out = []
        for li, b in enumerate(self.blocks):
            z = b.ln1(x)
            rc_before = self._coding_rate(z, b.attn, eps_sq)
            x_mid = x + b.attn(z)
            rc_after = self._coding_rate(b.ln1(x_mid), b.attn, eps_sq)
            x = b.ista(b.ln2(x_mid))
            out.append(dict(layer=li,
                            rc_before=float(rc_before),
                            rc_after=float(rc_after),
                            sparsity=1.0 - float(b.ista.last_rate)))
        return out

    @staticmethod
    def _coding_rate(z, attn, eps_sq):
        """R^c = sum_k p/(2N) logdet(I + p/(N eps^2) H_k H_k^T), H_k = U_k^T Z."""
        B, T, d = z.shape
        K, p = attn.K, attn.p
        h = attn.U(z).view(B, T, K, p).transpose(1, 2)     # B,K,T,p
        g = h.transpose(-2, -1) @ h                        # B,K,p,p
        alpha = p / (T * eps_sq)
        eye = torch.eye(p, device=z.device)
        return (torch.logdet(eye + alpha * g).sum(dim=1) * p / (2 * T)).mean()
