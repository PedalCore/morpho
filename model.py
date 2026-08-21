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
    # ---- M2: spike-driven weight paths (see M2.md) ----
    m2: str = ''               # '' | 'a' (codes into U, D, head) | 'b' (+ ternary error codes)
    m2_identity: bool = False  # control: reordered wiring, identity quantizer
    attn: str = 'mssa'         # 'mssa' | 'crsa' | 'dval' ('tssa' legacy)
    window: int = 0            # mssa only: banded causal mask (0 = full)
    dict_expand: int = 1       # >1: overcomplete dictionary fork (DICTIONARY.md)


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
        q = n.detach() * thr + (ste - ste.detach())
        a = getattr(self, 'blend', 1.0)    # calibrated conversion: 0 -> 1
        return q if a >= 1.0 else (1.0 - a) * v + a * q


class SignedProx(nn.Module):
    """Ternary-code prox for the reconstruction error (M2b):
    sign(r) * clamp(floor(|r|/thr), 0, L) * thr, trainable per-channel thr."""

    def __init__(self, dim, levels=4, init_threshold=0.5):
        super().__init__()
        self.levels = levels
        self.log_threshold = nn.Parameter(
            torch.full((dim,), float(init_threshold)).log())
        self.last_rate = None

    def forward(self, r):
        thr = self.log_threshold.exp()
        n = torch.clamp(torch.floor(r.abs() / thr), 0, self.levels)
        s = torch.sign(r)
        self.last_rate = (n.detach() != 0).float().mean()
        mask = (r.abs() < thr * (self.levels + 0.5)).to(r.dtype)
        ste = r * mask
        q = (s * n).detach() * thr + (ste - ste.detach())
        a = getattr(self, 'blend', 1.0)    # same blend contract as SpikeProx
        return q if a >= 1.0 else (1.0 - a) * r + a * q


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
        if cfg.window:                     # sliding-window arm (probe suite)
            mask = mask + torch.tril(
                torch.full((cfg.ctx, cfg.ctx), float('-inf')), -cfg.window)
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
            # ALIGNED substep (post-autopsy convention): z vs z + attn(z),
            # same basis and scaling — the LN'd form logged before the
            # autopsy commit is labeled as the old convention
            rc_after = self._coding_rate(z + b.attn(z), b.attn, eps_sq)
            x = b.ista(b.ln2(x_mid))
            out.append(dict(layer=li,
                            rc_before=float(rc_before),
                            rc_after=float(rc_after),
                            r_total=float(self._expansion_rate(x, eps_sq)),
                            sparsity=1.0 - float(b.ista.last_rate)))
        return out

    @staticmethod
    def _code_stats(z, prox):
        """Spike-code health: rate, level entropy, mean |code|."""
        if prox is None:
            nz = (z.detach() != 0).float().mean()
            return dict(rate=float(nz), entropy=None,
                        mag=float(z.detach().abs().mean()))
        thr = prox.log_threshold.exp()
        n = torch.round(z.detach() / thr).clamp(0, prox.levels).long()
        counts = torch.bincount(n.reshape(-1), minlength=prox.levels + 1).float()
        p = counts / counts.sum()
        ent = float(-(p[p > 0] * p[p > 0].log2()).sum())
        return dict(rate=float((n != 0).float().mean()), entropy=round(ent, 3),
                    mag=float(z.detach().abs().mean()))

    @staticmethod
    def _expansion_rate(z, eps_sq):
        """R(Z) — the diversity half of the objective. Deep compression with
        collapsing R means degenerate token collapse, not good structure."""
        B, T, d = z.shape
        g = z.transpose(-2, -1) @ z                        # B,d,d
        alpha = d / (T * eps_sq)
        eye = torch.eye(d, device=z.device)
        return (_chol_logdet(eye + alpha * g) / 2).mean().float()

    @staticmethod
    def _coding_rate(z, attn, eps_sq):
        return _coding_rate_impl(z, attn, eps_sq)


def _chol_logdet(M):
    """Overflow-robust logdet: float64, symmetrized, Cholesky
    (logdet = 2 sum log L_ii), escalating diagnostic jitter.
    Computed on CPU — MPS has no float64; these matrices are small."""
    M = M.detach().cpu().double()
    M = 0.5 * (M + M.transpose(-2, -1))
    eye = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
    for jitter in (0.0, 1e-8, 1e-6, 1e-4):
        try:
            L = torch.linalg.cholesky(M + jitter * eye)
            return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        except Exception:
            continue
    return torch.full(M.shape[:-2], float('nan'), dtype=M.dtype)


def _coding_rate_impl(z, attn, eps_sq):
    """R^c = sum_k p/(2N) logdet(I + p/(N eps^2) H_k H_k^T), H_k = U_k^T Z.
    (autograd path used by autopsy keeps torch.logdet; the robust Cholesky
    path serves the no-grad metrics)."""
    B, T, d = z.shape
    K, p = attn.K, attn.p
    h = attn.U(z).view(B, T, K, p).transpose(1, 2)     # B,K,T,p
    g = h.transpose(-2, -1) @ h                        # B,K,p,p
    alpha = p / (T * eps_sq)
    eye = torch.eye(p, device=z.device)
    if torch.is_grad_enabled() and z.requires_grad:
        return (torch.logdet(eye + alpha * g).sum(dim=1) * p / (2 * T)).mean()
    ld = _chol_logdet(eye + alpha * g)
    return (ld.sum(dim=1) * p / (2 * T)).mean().float()


class CRSA(nn.Module):
    """CRSA — Causal Rate-Statistics Attention (name provisional;
    "TSSA" is occupied by Decision SpikeFormer, CVPR 2025).
    M3 statistics attention (M3.md): no token pairs, no softmax.
    Per head, a causally decayed activity statistic gates the projected
    coordinates by their marginal coding price:

        c_{h,t} = rho_h c_{h,t-1} + h_{h,t}^2      (decaying counters)
        d = f'(c) = 1/(1+c)                        (f = log1p, continuous
                                                    control version; the
                                                    comparator staircase is
                                                    M3-step-rate)
        out = -scale * U^T [d ⊙ h]                 (descent form; training
                                                    decides what to do with
                                                    the sign — the aligned
                                                    dR^c metric will tell us)

    Heads carry a fixed dyadic ladder of horizons m ∈ {3,4,5,6}
    (time constants 8/16/32/64 tokens — the RWKV timescale lesson).
    Chunked parallel scan: exact cumsum inside 32-token chunks (bounded
    exponents, float32-safe even at m=3), recurrent carry across chunks."""

    CHUNK = 32

    def __init__(self, cfg):
        super().__init__()
        d, K = cfg.n_embd, cfg.n_head
        self.K, self.p = K, d // K
        self.U = nn.Linear(d, d, bias=False)
        self.tied = cfg.tied
        if not cfg.tied:
            self.out = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(cfg.mssa_scale))
        ms = [3 + (k % 4) for k in range(K)]
        self.register_buffer('rho', torch.tensor(
            [1.0 - 2.0 ** (-m) for m in ms]).float())

    def forward(self, x):                          # (B, T, d)
        B, T, d = x.shape
        h = self.U(x).view(B, T, self.K, self.p).permute(0, 2, 1, 3)
        rho = self.rho.view(1, self.K, 1, 1)
        C = self.CHUNK
        n_chunks = (T + C - 1) // C
        cs = []
        carry = torch.zeros(B, self.K, 1, self.p, device=x.device,
                            dtype=h.dtype)
        for ci in range(n_chunks):
            hc = h[:, :, ci * C:(ci + 1) * C]
            L = hc.shape[2]
            t = torch.arange(L, device=x.device, dtype=h.dtype).view(1, 1, L, 1)
            down = rho ** (t + 1)                  # rho^{t+1} for the carry
            local = (rho ** t) * torch.cumsum(hc * hc * (rho ** (-t)), dim=2)
            c = down * carry + local
            cs.append(c)
            carry = c[:, :, -1:]
        c = torch.cat(cs, dim=2)
        dcoef = 1.0 / (1.0 + c)
        agg = (dcoef * h).permute(0, 2, 1, 3).reshape(B, T, d)
        out = (F.linear(agg, self.U.weight.t()) if self.tied
               else self.out(agg))
        return -self.scale * out


class DecayedValue(nn.Module):
    """Probe-suite contrast arm: same params and horizon ladder as CRSA,
    but the state RETRIEVES past values (s_t = rho s_{t-1} + (1-rho) h_t)
    instead of pricing current activity by past statistics — the
    RWKV-style linear-recurrence baseline at comparable rho."""

    CHUNK = 32

    def __init__(self, cfg):
        super().__init__()
        d, K = cfg.n_embd, cfg.n_head
        self.K, self.p = K, d // K
        self.U = nn.Linear(d, d, bias=False)
        self.tied = cfg.tied
        if not cfg.tied:
            self.out = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(cfg.mssa_scale))
        ms = [3 + (k % 4) for k in range(K)]
        self.register_buffer('rho', torch.tensor(
            [1.0 - 2.0 ** (-m) for m in ms]).float())

    def forward(self, x):
        B, T, d = x.shape
        h = self.U(x).view(B, T, self.K, self.p).permute(0, 2, 1, 3)
        rho = self.rho.view(1, self.K, 1, 1)
        C = self.CHUNK
        cs = []
        carry = torch.zeros(B, self.K, 1, self.p, device=x.device,
                            dtype=h.dtype)
        for ci in range((T + C - 1) // C):
            hc = h[:, :, ci * C:(ci + 1) * C]
            t = torch.arange(hc.shape[2], device=x.device,
                             dtype=h.dtype).view(1, 1, -1, 1)
            s = ((rho ** (t + 1)) * carry +
                 (rho ** t) * torch.cumsum((1 - rho) * hc * (rho ** (-t)),
                                           dim=2))
            cs.append(s)
            carry = s[:, :, -1:]
        s = torch.cat(cs, dim=2)
        agg = s.permute(0, 2, 1, 3).reshape(B, T, d)
        out = (F.linear(agg, self.U.weight.t()) if self.tied
               else self.out(agg))
        return self.scale * out


# ------------------------------------------------------- M2: spike-driven paths

class BlockM2(nn.Module):
    """M2 block (M2.md eq.): the layer state is a CODE vector; U and D
    consume it raw (no LN in between), the ISTA step unrolls from the
    previous code, and (M2b) the reconstruction error is ternary-quantized
    so D^T consumes codes too."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.n_embd
        self.attn = ({'crsa': CRSA, 'tssa': CRSA, 'dval': DecayedValue}
                     .get(cfg.attn, MSSA))(cfg)
        self.ln = nn.LayerNorm(d)
        self.D = nn.Parameter(torch.empty(d, d))
        nn.init.orthogonal_(self.D)
        self.eta = cfg.ista_eta
        self.eprox = (SignedProx(d, cfg.spike_levels, cfg.spike_init_threshold)
                      if cfg.m2 == 'b' and not cfg.m2_identity else None)
        self.prox = (None if cfg.m2_identity else
                     SpikeProx(d, cfg.spike_levels, cfg.spike_init_threshold))

    def forward(self, z):
        x = z + self.attn(z)               # U consumes codes directly
        u = self.ln(x)
        r = u - z @ self.D.t()             # D consumes codes
        if self.eprox is not None:
            r = self.eprox(r)              # ternary error codes (M2b)
        v = z + self.eta * (r @ self.D)    # D^T consumes r (codes in M2b)
        return self.prox(v) if self.prox is not None else v


class BlockOD(nn.Module):
    """Overcomplete-dictionary block (DICTIONARY.md): state pair (z, a).
    z (d) feeds attention; the wide sparse code a (n = expand*d) unrolls
    across layers; active soft-threshold prox; decode continues the
    d-stream."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.n_embd
        n = cfg.dict_expand * d
        self.attn = ({'crsa': CRSA, 'tssa': CRSA, 'dval': DecayedValue}
                     .get(cfg.attn, MSSA))(cfg)
        self.ln = nn.LayerNorm(d)
        D = torch.randn(d, n)
        D = D / D.norm(dim=0, keepdim=True)          # column-normalized init
        self.D = nn.Parameter(D)
        self.eta = cfg.ista_eta
        self.lam = cfg.ista_lambda
        self.last_rate = None

    def forward(self, z, a):
        x = z + self.attn(z)
        u = self.ln(x)
        r = u - a @ self.D.t()                       # signal residual (d)
        pre = a + self.eta * (r @ self.D)            # ISTA step on the code
        a2 = torch.relu(pre - self.eta * self.lam)   # ACTIVE prox
        self.last_rate = (a2.detach() != 0).float().mean()
        return a2 @ self.D.t(), a2                   # decode, carry code


class CausalCRATEM2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.m2 in ('a', 'b')
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.ctx, cfg.n_embd)
        self.ln_in = nn.LayerNorm(cfg.n_embd)
        self.prox_in = (None if cfg.m2_identity else
                        SpikeProx(cfg.n_embd, cfg.spike_levels,
                                  cfg.spike_init_threshold))
        blk = BlockOD if cfg.dict_expand > 1 else BlockM2
        self.blocks = nn.ModuleList(blk(cfg) for _ in range(cfg.n_layer))
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight     # head consumes codes directly
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _embed(self, idx):
        T = idx.shape[1]
        x = self.ln_in(self.emb(idx) + self.pos(torch.arange(T, device=idx.device)))
        return self.prox_in(x) if self.prox_in is not None else x

    def forward(self, idx, targets=None):
        z = self._embed(idx)
        if self.cfg.dict_expand > 1:
            a = torch.zeros(*z.shape[:-1],
                            self.cfg.dict_expand * self.cfg.n_embd,
                            device=z.device, dtype=z.dtype)
            for b in self.blocks:
                z, a = b(z, a)
        else:
            for b in self.blocks:
                z = b(z)
        logits = self.head(z)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def layer_metrics(self, idx, eps_sq=0.5):
        """ALIGNED attention substep: dR^c = R^c(z + attn(z); U) − R^c(z; U),
        same basis, same scaling — no LayerNorm between the two sides (a
        LN'd comparison can flip sign without the substep changing)."""
        z = self._embed(idx)
        if self.cfg.dict_expand > 1:
            a = torch.zeros(*z.shape[:-1],
                            self.cfg.dict_expand * self.cfg.n_embd,
                            device=z.device, dtype=z.dtype)
            out = []
            for li, b in enumerate(self.blocks):
                rc_before = _coding_rate_impl(z, b.attn, eps_sq)
                rc_after = _coding_rate_impl(z + b.attn(z), b.attn, eps_sq)
                z, a = b(z, a)
                out.append(dict(layer=li, rc_before=float(rc_before),
                                rc_after=float(rc_after),
                                r_total=float(
                                    CausalCRATE._expansion_rate(z, eps_sq)),
                                sparsity=1.0 - float(b.last_rate),
                                entropy=None, mag=float(a.abs().mean()),
                                zmax=round(float(z.abs().max()), 2),
                                err_rate=None))
            return out
        out = []
        for li, b in enumerate(self.blocks):
            rc_before = _coding_rate_impl(z, b.attn, eps_sq)
            x = z + b.attn(z)
            rc_after = _coding_rate_impl(x, b.attn, eps_sq)
            z = b(z)
            stats = CausalCRATE._code_stats(z, b.prox)
            out.append(dict(layer=li,
                            rc_before=float(rc_before),
                            rc_after=float(rc_after),
                            r_total=float(CausalCRATE._expansion_rate(z, eps_sq)),
                            sparsity=1.0 - stats['rate'],
                            entropy=stats['entropy'],
                            mag=round(stats['mag'], 4),
                            zmax=round(float(z.detach().abs().max()), 2),
                            err_rate=(float(b.eprox.last_rate)
                                      if b.eprox is not None and
                                      b.eprox.last_rate is not None else None)))
        return out

    def set_levels(self, levels):
        """Annealing hook: 4 -> 2 -> 1 during training."""
        for m in self.modules():
            if isinstance(m, (SpikeProx, SignedProx)):
                m.levels = levels


TSSA = CRSA   # legacy alias (pre-rename imports)
