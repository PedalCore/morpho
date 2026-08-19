"""Train a model small enough to become a circuit.

Task: delayed recall — emit the symbol seen `delay` steps ago. Chosen because
a tiny recurrence can solve it EXACTLY, so a compiled netlist's output is
verifiable rather than merely plausible, and because it is the task the
reservoir campaign used to probe memory.

Architecture: exactly the primitive morpho_lm.py describes.

    u = E[token]                       embedding, d channels
    h = a * h + b * u                  the diagonal recurrence
    y = W h + c                        linear readout over the vocabulary

with `a` a per-channel constant (the RWKV configuration). No attention, no
gating, no layer norm — every operation has a cell in morpho_lm.

    python train_toy.py [--d 8] [--delay 3] [--vocab 4]

Writes toy-export/{weights.json,report.json}: float weights, their Q4.4
fixed-point quantization, and the accuracy of both.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


class ToyRecur(nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        # decays spread across channels: distinct memory timescales, which the
        # readout needs to separate recent inputs from older ones
        self.decay_raw = nn.Parameter(torch.linspace(-3, 3, d))
        self.b = nn.Parameter(torch.ones(d) * 0.5)
        self.out = nn.Linear(d, vocab)
        self.d = d

    def forward(self, idx):                                # (B, T) -> (B, T, V)
        B, T = idx.shape
        u = self.emb(idx)
        a = torch.sigmoid(self.decay_raw)
        h = torch.zeros(B, self.d, device=idx.device)
        ys = []
        for t in range(T):
            h = a * h + self.b * u[:, t]
            ys.append(self.out(h))
        return torch.stack(ys, 1)


def batch(rng, vocab, delay, B, T):
    x = torch.from_numpy(rng.integers(0, vocab, size=(B, T)).astype(np.int64))
    y = torch.full_like(x, -100)
    y[:, delay:] = x[:, :-delay]                           # target = symbol from `delay` ago
    return x, y


def accuracy(model, rng, cfg, n=64):
    x, y = batch(rng, cfg.vocab, cfg.delay, n, cfg.T)
    with torch.no_grad():
        p = model(x).argmax(-1)
    m = y >= 0
    return float((p[m] == y[m]).float().mean())


def quantize(model, bits=8):
    """Q(bits/2).(bits/2) unsigned decay, signed everything else."""
    frac = bits // 2
    def q(t, signed=True):
        v = np.round(np.asarray(t, dtype=np.float64) * (1 << frac))
        lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        return np.clip(v, lo, hi).astype(np.int64)
    a = torch.sigmoid(model.decay_raw).detach().numpy()
    return {
        "bits": bits, "frac": frac,
        "a": q(a, signed=False).tolist(),
        "b": q(model.b.detach().numpy()).tolist(),
        "emb": q(model.emb.weight.detach().numpy()).tolist(),
        "w_out": q(model.out.weight.detach().numpy()).tolist(),
        "b_out": q(model.out.bias.detach().numpy()).tolist(),
    }


def fixed_point_accuracy(qw, rng, cfg, n=256):
    """Run the quantized model in pure integer arithmetic — what the circuit does."""
    bits, frac = qw["bits"], qw["frac"]
    mask = (1 << bits) - 1
    a = np.array(qw["a"]); b = np.array(qw["b"])
    emb = np.array(qw["emb"]); W = np.array(qw["w_out"]); bo = np.array(qw["b_out"])
    def smul(x, y):
        return np.sign(x) * np.sign(y) * ((np.abs(x) * np.abs(y)) >> frac)
    x, y = batch(rng, cfg.vocab, cfg.delay, n, cfg.T)
    x, y = x.numpy(), y.numpy()
    h = np.zeros((n, len(a)), dtype=np.int64)
    correct = total = 0
    for t in range(cfg.T):
        u = emb[x[:, t]]
        h = smul(a[None, :], h) + smul(b[None, :], u)
        h = ((h + (1 << (bits - 1))) & mask) - (1 << (bits - 1))     # wrap
        logits = smul(h[:, None, :], W[None]).sum(-1) + bo
        m = y[:, t] >= 0
        correct += int((logits.argmax(-1)[m] == y[:, t][m]).sum()); total += int(m.sum())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=4)
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bits", type=int, default=12)   # Q6.6; Q4.4 is too coarse (0.28 acc)
    cfg = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = ToyRecur(cfg.vocab, cfg.d)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    print(f"toy recurrence: d={cfg.d}, vocab={cfg.vocab}, delay={cfg.delay}, "
          f"{sum(p.numel() for p in model.parameters())} parameters")
    for step in range(cfg.steps):
        x, y = batch(rng, cfg.vocab, cfg.delay, 64, cfg.T)
        loss = lossf(model(x).reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1:5d}  loss {loss.item():.4f}  "
                  f"acc {accuracy(model, rng, cfg):.4f}")

    acc_f = accuracy(model, rng, cfg, n=256)
    qw = quantize(model, cfg.bits)
    acc_q = fixed_point_accuracy(qw, np.random.default_rng(7), cfg)
    print(f"\nfloat accuracy      {acc_f:.4f}")
    print(f"Q{cfg.bits//2}.{cfg.bits//2} fixed point   {acc_q:.4f}   <- what the circuit computes")

    out = os.path.join(os.path.dirname(__file__), "toy-export")
    os.makedirs(out, exist_ok=True)
    qw["config"] = vars(cfg)
    qw["accuracy_float"] = acc_f
    qw["accuracy_fixed"] = acc_q
    json.dump(qw, open(os.path.join(out, "weights.json"), "w"), indent=1)
    print(f"\nwrote {out}/weights.json")


if __name__ == "__main__":
    main()
