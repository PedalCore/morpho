"""M7 — DNA classification models (M7-DNA.md).

Three matched arms differing ONLY in the sequence mixer:
  'cnn'      — local motif stem + pooling (no recurrence)
  'counter'  — RC-tied bidirectional CRSA counters, DNA horizons
  'longhorn' — faithful (no-Wv) diagonal delta memory, bidirectional

Shared everywhere: single-base tokens (A C G T N), motif conv stem
(width 11), MLP feature block, mean+max pooling, classifier head.
RC equivariance: logits = 1/2 [f(x) + f(RC(x))] with SHARED params —
exact reverse-complement invariance for classification.
"""

import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC_MAP = torch.tensor([3, 2, 1, 0, 4])          # A<->T, C<->G, N->N
DNA_MS = (4, 6, 8, 10)                          # half-lives ~11/44/177/710 b


def encode(seq):
    return torch.tensor([BASES.get(c, 4) for c in seq.upper()],
                        dtype=torch.long)


def rc(tokens):
    """Reverse complement in token space."""
    return RC_MAP.to(tokens.device)[tokens].flip(-1)


class BiCounter(nn.Module):
    """Bidirectional multiscale counters: forward + backward decayed
    event statistics per coordinate, DNA dyadic ladder, price-gated
    read (the CRSA rule, no causal restriction needed for
    classification)."""

    def __init__(self, d):
        super().__init__()
        H = len(DNA_MS)
        assert d % H == 0
        self.H, self.p = H, d // H
        self.U = nn.Linear(d, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer('rho', torch.tensor(
            [1.0 - 2.0 ** (-m) for m in DNA_MS]).float())

    def _scan(self, v, rho):                     # causal decayed cumsum
        C, T = 128, v.shape[2]
        ys, carry = [], torch.zeros_like(v[:, :, :1])
        for ci in range((T + C - 1) // C):
            vc = v[:, :, ci*C:(ci+1)*C]
            L = vc.shape[2]
            t = torch.arange(L, device=v.device, dtype=v.dtype).view(1, 1, L, 1)
            y = (rho ** (t + 1)) * carry + \
                (rho ** t) * torch.cumsum(vc * (rho ** (-t)), dim=2)
            ys.append(y)
            carry = y[:, :, -1:]
        return torch.cat(ys, dim=2)

    def forward(self, x):                        # (B, T, d)
        B, T, d = x.shape
        h = self.U(x).view(B, T, self.H, self.p).permute(0, 2, 1, 3)
        rho = self.rho.view(1, self.H, 1, 1)
        cf = self._scan(h * h, rho)              # forward counters
        cb = self._scan((h * h).flip(2), rho).flip(2)   # backward
        c = 0.5 * (cf + cb)
        read = (1.0 / (1.0 + c)) * h
        agg = read.permute(0, 2, 1, 3).reshape(B, T, d)
        return self.scale * F.linear(agg, self.U.weight.t())


class BiDelta(nn.Module):
    """Faithful (no-Wv) diagonal delta memory, run in both directions
    with shared params; outputs averaged."""

    def __init__(self, d):
        super().__init__()
        from whitebox.model import Config, LonghornMem
        cfg = Config(n_embd=d, n_head=4, ctx=8192, attn='longhorn',
                     lh_no_wv=True)
        self.mem = LonghornMem(cfg)

    def forward(self, x):
        # Both directions stacked into the batch: one scan call instead of
        # two (MPS dispatch overhead dominates at this width).
        B = x.shape[0]
        y = self.mem(torch.cat([x, x.flip(1)], 0))
        return 0.5 * (y[:B] + y[B:].flip(1))


class DNAClassifier(nn.Module):
    def __init__(self, arm='counter', d=128, n_layer=4, n_classes=2,
                 rc=True):
        super().__init__()
        self.rc = rc
        self.emb = nn.Embedding(5, d)
        self.stem = nn.Conv1d(d, d, 11, padding=5)   # motif detector
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            mixer = (nn.Identity() if arm == 'cnn' else
                     BiCounter(d) if arm == 'counter' else BiDelta(d))
            self.blocks.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d), mixer=mixer,
                ln2=nn.LayerNorm(d),
                mlp=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                  nn.Linear(4 * d, d)))))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(2 * d, n_classes)

    def _trunk(self, tokens):                    # (B, T) -> logits
        x = self.emb(tokens)
        x = x + F.gelu(self.stem(x.transpose(1, 2))).transpose(1, 2)
        for b in self.blocks:
            m = b['mixer']
            if not isinstance(m, nn.Identity):
                x = x + m(b['ln1'](x))
            x = x + b['mlp'](b['ln2'](x))
        x = self.ln_f(x)
        pooled = torch.cat([x.mean(1), x.max(1).values], dim=-1)
        return self.head(pooled)

    def forward(self, tokens):
        """RC-equivariant: average logits over strand and its RC
        (batched as one trunk call). rc=False: single strand —
        strand identity PRESERVED (directional tasks, e.g. splice)."""
        if not self.rc:
            return self._trunk(tokens)
        B = tokens.shape[0]
        logits = self._trunk(torch.cat([tokens, rc(tokens)], 0))
        return 0.5 * (logits[:B] + logits[B:])
