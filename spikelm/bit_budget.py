"""How many bits does each part of a language model actually need?

Motivation. Spiking and other amplitude-free codes fail on language while
succeeding on MNIST, and the usual explanation ("language is harder") does
not tell you where to spend a precision budget. This measures it directly:
quantise ONE tensor family at a time to b bits, leave everything else in
float32, and watch held-out perplexity.

The hypothesis under test, written before running it: the tensors that
CARRY content — the residual stream, the logits, the value path — will be
unforgiving, while the tensors that ROUTE — the receptance gate, the decay
constants — will be cheap, because routing only needs the ordering of its
decisions to survive, not their exact values. If that holds, it says where
an event-driven or spiking mechanism could be substituted without paying
for it, and where it could not.

Every arm is the same trained checkpoint on the same fixed batches, so the
only thing moving is the numeric format of one tensor family.

    python bit_budget.py --ckpt runs/base-rwkv-d384L6-s42/ckpt.pt
"""

import argparse
import json
import math

import numpy as np
import torch

from spikelm.data import get_tokenizer, load_split
from spikelm.model import Config, RWKVMini

BITS = [1, 2, 3, 4, 5, 6, 8, 12]


# ------------------------------------------------------------- quantisation

def q_sym(x, bits):
    """Symmetric uniform quantisation, scale from the tensor's own range.

    b bits spans [-2^(b-1), 2^(b-1)-1]. At b=1 this is sign(x)*max|x|, i.e.
    a binary code, which is the amplitude-free limit we care about.
    """
    if bits is None:
        return x
    n = 2 ** (bits - 1)
    s = x.abs().amax().clamp_min(1e-12) / max(n - 1, 1)
    return torch.round(x / s).clamp(-n, n - 1) * s


def q_unit(x, bits):
    """For tensors already in [0,1] (gates): unsigned, full range used."""
    if bits is None:
        return x
    n = 2 ** bits - 1
    return torch.round(x.clamp(0, 1) * n) / n


# ------------------------------------------------------------------ patching

class Probe:
    """Which tensor family is being quantised, and to how many bits."""

    def __init__(self):
        self.target, self.bits = None, None

    def set(self, target, bits):
        self.target, self.bits = target, bits

    def on(self, name):
        return self.bits if self.target == name else None


P = Probe()


def timemix_forward(self, x):
    """TimeMix with a quantisation point on every internal tensor.

    Mirrors spikelm/model.py TimeMix.forward exactly; the only additions are
    the q_* calls, each active only when that family is the one under test.
    """
    xs = self.shift(x)
    k = self.key(x * self.mix_k + xs * (1 - self.mix_k))
    v = self.value(x * self.mix_v + xs * (1 - self.mix_v))
    r = torch.sigmoid(self.receptance(x * self.mix_r + xs * (1 - self.mix_r)))

    k = q_sym(k, P.on("key"))
    v = q_sym(v, P.on("value"))
    r = q_unit(r, P.on("gate_r"))

    B, T, C = x.shape
    w = -torch.exp(q_sym(self.time_decay, P.on("decay")))
    u = self.time_first
    aa = torch.zeros(B, C, device=x.device)
    bb = torch.zeros(B, C, device=x.device)
    pp = torch.full((B, C), -1e38, device=x.device)
    out = torch.empty(B, T, C, device=x.device)
    sbits = P.on("state")
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
        if sbits is not None:
            # the recurrent memory itself, requantised every step so error
            # accumulates through the recurrence rather than being applied once
            aa, bb = q_sym(aa, sbits), q_sym(bb, sbits)
    return self.output(r * q_sym(out, P.on("wkv_out")))


def block_forward(self, x):
    """Residual stream quantised where each sublayer reads and writes it."""
    x = q_sym(x, P.on("residual"))
    x = x + self.tm(self.ln1(x))
    x = q_sym(x, P.on("residual"))
    x = x + self.cm(self.ln2(x))
    return x


def model_forward(self, idx, targets=None):
    import torch.nn.functional as F
    x = self.ln_in(self.emb(idx))
    x = q_sym(x, P.on("embedding"))
    for b in self.blocks:
        x = b(x)
    x = self.ln_out(x)
    if self.sync is not None:
        x = x + self.sync(x)
    logits = self.head(x)
    logits = q_sym(logits, P.on("logits"))
    if targets is None:
        return logits, None
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return logits, loss


def install(model):
    import types
    from spikelm.model import TimeMix, Block
    for m in model.modules():
        if isinstance(m, TimeMix):
            m.forward = types.MethodType(timemix_forward, m)
        elif isinstance(m, Block):
            m.forward = types.MethodType(block_forward, m)
    model.forward = types.MethodType(model_forward, model)


# ------------------------------------------------------------------ weights

def quantise_weights(model, bits, which):
    """Weights are a separate arm: quantised once, per output row.

    Per-row rather than per-tensor because that is what the export path and
    the crossbar circuits actually do — one scale per array row.
    """
    import torch.nn as nn
    orig = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if which == "weights_head" and "head" not in name:
            continue
        if which == "weights_body" and "head" in name:
            continue
        W = m.weight.data
        orig[name] = W.clone()
        n = 2 ** (bits - 1)
        s = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / max(n - 1, 1)
        m.weight.data = torch.round(W / s).clamp(-n, n - 1) * s
    return orig


def restore_weights(model, orig):
    for name, m in model.named_modules():
        if name in orig:
            m.weight.data = orig[name]


# --------------------------------------------------------------------- eval

@torch.no_grad()
def perplexity(model, batches):
    tot = ntok = 0
    for x, y in batches:
        _, loss = model(x, y)
        tot += float(loss) * y.numel()
        ntok += y.numel()
    return math.exp(tot / ntok)


ACTIVATION_ARMS = [
    ("residual", "residual stream (the workspace)"),
    ("logits", "logits (output distribution)"),
    ("embedding", "embedding output"),
    ("value", "value path (content)"),
    ("key", "key path (content)"),
    ("wkv_out", "wkv output (mixed content)"),
    ("state", "recurrent state aa/bb (memory)"),
    ("gate_r", "receptance gate r (routing)"),
    ("decay", "time decay w (routing)"),
]
WEIGHT_ARMS = [("weights_body", "weights, all but head"), ("weights_head", "weights, head only")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base-rwkv-d384L6-s42/ckpt.pt")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--out", default="bit-budget.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size)
    model = RWKVMini(cfg)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
    model.eval().to(a.device)
    install(model)

    data = load_split("valid")
    rng = np.random.default_rng(7)
    B, T = 8, cfg.ctx
    nb = max(1, a.tokens // (B * T))
    batches = []
    for _ in range(nb):
        ix = rng.integers(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64)
        batches.append((torch.from_numpy(x).to(a.device),
                        torch.from_numpy(y).to(a.device)))

    P.set(None, None)
    ref = perplexity(model, batches)
    print(f"{a.ckpt}\n{nb} batches x {B}x{T} = {nb*B*T} tokens on {a.device}")
    print(f"float32 reference perplexity {ref:.4f}\n")

    hdr = "  " + f"{'tensor family':<34}" + "".join(f"{b:>8}b" for b in BITS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    results = {"reference_ppl": ref, "bits": BITS, "arms": {}}

    def row(key, label, run):
        vals = []
        for b in BITS:
            ppl = run(b)
            vals.append(ppl)
            print(f"{100*(ppl/ref-1):>8.1f}" if ppl < 1e4 else f"{'>1e4':>8}",
                  end="", flush=True)
        print()
        results["arms"][key] = {"label": label, "ppl": vals,
                                "delta_pct": [100 * (v / ref - 1) for v in vals]}

    print("  (each cell is % perplexity increase vs float32; lower is cheaper)\n")
    for key, label in ACTIVATION_ARMS:
        print(f"  {label:<34}", end="", flush=True)
        def run(b, key=key):
            P.set(key, b)
            p = perplexity(model, batches)
            P.set(None, None)
            return p
        row(key, label, run)

    for key, label in WEIGHT_ARMS:
        print(f"  {label:<34}", end="", flush=True)
        def run(b, key=key):
            orig = quantise_weights(model, b, key)
            p = perplexity(model, batches)
            restore_weights(model, orig)
            return p
        row(key, label, run)

    # the headline: fewest bits each family tolerates at a 1% and 5% budget
    print("\n  minimum bits to stay within a perplexity budget:")
    print(f"    {'tensor family':<34}{'<1%':>6}{'<5%':>6}")
    summary = {}
    for key, r in results["arms"].items():
        got = {}
        for lim in (1.0, 5.0):
            ok = [b for b, d in zip(BITS, r["delta_pct"]) if d < lim]
            got[f"bits_{int(lim)}pct"] = min(ok) if ok else None
        summary[key] = got
        f = lambda v: str(v) if v else ">12"
        print(f"    {r['label']:<34}{f(got['bits_1pct']):>6}{f(got['bits_5pct']):>6}")
    results["summary"] = summary

    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
