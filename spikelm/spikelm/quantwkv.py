"""Fixed-point emulation of the RWKV wkv recurrence — the circuit numerics.

Answers, for a trained model rather than in the abstract: what does the
proposed hardware do to perplexity and to generated text?

Emulated exactly as the circuit would compute it:
  * base-2 reparameterization (w2 = w·log2e etc), so exp becomes 2^(-x)
  * Qm.n signed fixed-point registers with SATURATION (counted, not hidden)
  * 2^(-x) as barrel shift (integer part) + L-entry LUT on the fractional
    part, with optional linear interpolation
  * optional restoring (shift-subtract) division on the output path

Every exponent argument is <= 0 by construction (x - max(.., x)), so the
LUT only ever covers the decaying half and underflow to zero is correct.
"""

import math

import torch

LOG2E = math.log2(math.e)


class QConfig:
    def __init__(self, m=8, n=8, lut=32, interp=True, exact_exp=False,
                 exact_div=True, div_bits=24, name=None):
        self.m, self.n = m, n              # Qm.n signed: m int bits (incl sign)
        self.lut, self.interp = lut, interp
        self.exact_exp, self.exact_div = exact_exp, exact_div
        self.div_bits = div_bits
        self.name = name or f"Q{m}.{n}"
        self.hi = 2.0 ** (m - 1) - 2.0 ** -n
        self.lo = -(2.0 ** (m - 1))
        self.res = 2.0 ** -n


class Stats:
    def __init__(self):
        self.sat = 0
        self.n = 0
        self.err_sq = 0.0
        self.err_max = 0.0
        self.per_step = []                  # RMS error vs float, per timestep


def quantize(x, cfg, stats=None):
    q = torch.round(x / cfg.res) * cfg.res
    if stats is not None:
        stats.sat += int(((q > cfg.hi) | (q < cfg.lo)).sum())
        stats.n += q.numel()
    return q.clamp(cfg.lo, cfg.hi)


def make_lut(L):
    i = torch.arange(L + 1, dtype=torch.float64)
    return torch.pow(2.0, -i / L)           # 2^(-f), f in [0,1]


def exp2_neg(x, cfg, lut):
    """2^x for x <= 0: barrel shift by the integer part, LUT on the rest."""
    if cfg.exact_exp:
        return torch.pow(2.0, x)
    a = (-x).clamp(min=0)
    i = torch.floor(a)
    f = a - i
    pos = f * cfg.lut
    idx = torch.floor(pos).clamp(0, cfg.lut - 1).long()
    v0 = lut[idx]
    if cfg.interp:
        v = v0 + (pos - idx.to(pos.dtype)) * (lut[idx + 1] - v0)
    else:
        v = v0
    shift = torch.pow(2.0, -i.clamp(max=40))   # exact in binary: a shift
    return v * shift


def restoring_div(num, den, cfg):
    """Shift-subtract division, sign handled separately (den > 0 always)."""
    s = torch.sign(num)
    N = torch.round(num.abs() / cfg.res)
    D = torch.round(den / cfg.res).clamp(min=1)
    rem = torch.zeros_like(N)
    quo = torch.zeros_like(N)
    bits = cfg.div_bits
    Nsh = N * (2.0 ** cfg.n)                   # numerator pre-shifted
    for i in range(bits - 1, -1, -1):
        bit = torch.floor(Nsh / (2.0 ** i)) % 2
        rem = rem * 2 + bit
        ge = (rem >= D).to(rem.dtype)
        rem = rem - ge * D
        quo = quo * 2 + ge
    return s * quo * cfg.res


def wkv(k, v, w, u, cfg=None, stats=None, ref=None):
    """(B,T,C) k,v and (C,) w,u -> (B,T,C). cfg None = float64 reference."""
    B, T, C = k.shape
    float_mode = cfg is None
    if not float_mode:
        lut = make_lut(cfg.lut).to(k.dtype)
        k = quantize(k * LOG2E, cfg, stats)
        w = quantize(w * LOG2E, cfg, stats)
        u = quantize(u * LOG2E, cfg, stats)
        v = quantize(v, cfg, stats)
        Q = lambda x: quantize(x, cfg, stats)
        E = lambda x: exp2_neg(x, cfg, lut)
    else:
        k = k * LOG2E; w = w * LOG2E; u = u * LOG2E
        Q = lambda x: x
        E = lambda x: torch.pow(2.0, x)

    aa = torch.zeros(B, C, dtype=k.dtype)
    bb = torch.zeros(B, C, dtype=k.dtype)
    pp = torch.full((B, C), -60.0, dtype=k.dtype)
    out = torch.empty(B, T, C, dtype=k.dtype)
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        ww = Q(u + kt)
        p = torch.maximum(pp, ww)
        e1, e2 = E(pp - p), E(ww - p)
        num = Q(e1 * aa + e2 * vt)
        den = Q(e1 * bb + e2 + 1e-4)
        if float_mode or cfg.exact_div:
            out[:, t] = num / den
        else:
            out[:, t] = restoring_div(num, den, cfg)
        ww2 = Q(pp + w)
        p2 = torch.maximum(ww2, kt)
        e1, e2 = E(ww2 - p2), E(kt - p2)
        aa = Q(e1 * aa + e2 * vt)
        bb = Q(e1 * bb + e2)
        pp = p2
        if ref is not None and stats is not None:
            d = (out[:, t] - ref[:, t])
            stats.per_step.append(float(torch.sqrt((d * d).mean())))
            stats.err_sq += float((d * d).sum())
            stats.err_max = max(stats.err_max, float(d.abs().max()))
    if ref is not None and stats is not None:
        stats.rms = math.sqrt(stats.err_sq / (B * T * C))
    return out
