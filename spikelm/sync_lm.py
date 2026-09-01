"""A spiking-synchronisation tick head on a language model trunk.

This is the architecture the campaign's surviving results point at, built
now that the silicon constraint is relaxed: spikes and synchronisation are
kept because they are the research question; exact accumulators and full-
precision projections are kept because nothing forces us to cripple them.

What goes in, and the measurement behind each piece:

  * the CTM tick loop            +34.4 points on parity, per-tick trace
                                 visibly computing (ctm_parity.py)
  * SPIKING sync                 z thresholded to {0,1} before the pairwise
                                 product matched or beat real-valued sync
                                 (ctm_spiking_sync.py)
  * small D, many pairs          512 pairs of 32 neurons beat 128 raw
                                 activations at every K (sync_width_task.py)
                                 [param-matched act/512 control in flight]
  * exact accumulators           counter-width results were invalidated by
                                 parity's training bimodality; with the
                                 hardware constraint dropped we simply use
                                 exact and the question dissolves
  * factored readout P -> d      the scaling analysis: a direct P -> vocab
                                 projection costs more than the block it
                                 rides on; going through d reuses the tied
                                 embedding as every other head here does

Shape: an unchanged RWKV trunk produces hidden states h. Per token, a
small CTM runs T internal ticks: state z (D neurons, seeded from h_i),
neuron-level models over the tick history, spiking synchronisation over P
pairs, and a sync-driven query that cross-attends CAUSALLY over the
trunk's hidden states. The final tick's sync representation is projected
back to d and added to h as a zero-initialised residual, so step 0 is
exactly the trunk and any gain must be learned. This is the loop our
first sync attempt (spike-sync-rwkv, worse than its baseline) was
missing; the earlier failure is the control for this design.

Two arms, identical trunk, identical data order:

    python sync_lm.py --sync 0 --tag base      # trunk alone
    python sync_lm.py --sync 1 --tag sync      # trunk + sync-tick head

Language is the right test now and not before because of the width
result: a vocabulary head wants many independent directions, which is
exactly what sync measurably provides and what parity never asked for.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikelm.data import get_tokenizer, load_split
from spikelm.model import Config, RWKVMini
from ctm import NeuronLevelModels, Synchronisation


def spike(z):
    """{0,1} threshold, straight-through backward — as in ctm_spiking_sync."""
    hard = (z > 0).to(z.dtype)
    return z + (hard - z).detach()


class SyncTickHead(nn.Module):
    def __init__(self, d, D=64, M=8, ticks=4, pairs=512, d_head=64,
                 nlm_hidden=8, spike_sync=True):
        super().__init__()
        self.D, self.M, self.ticks, self.d_head = D, M, ticks, d_head
        self.spike_sync = spike_sync
        self.seed = nn.Linear(d, D)
        self.synapse = nn.Sequential(
            nn.Linear(D + d_head, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.ln = nn.LayerNorm(D)
        self.nlm = NeuronLevelModels(D, M, nlm_hidden)
        self.sync_out = Synchronisation(D, pairs, seed=1)
        self.sync_act = Synchronisation(D, pairs, seed=2)
        self.w_in = nn.Linear(pairs, d_head, bias=False)
        self.k = nn.Linear(d, d_head, bias=False)
        self.v = nn.Linear(d, d_head, bias=False)
        self.w_out = nn.Linear(pairs, d, bias=False)
        nn.init.zeros_(self.w_out.weight)      # exact no-op at step 0
        self.scale = d_head ** -0.5

    def forward(self, h):                       # h: (B, N, d) trunk states
        B, N, _ = h.shape
        z = torch.tanh(self.seed(h))                          # (B, N, D)
        A = h.new_zeros(B, N, self.D, self.M)
        o = h.new_zeros(B, N, self.d_head)
        K, V = self.k(h), self.v(h)
        mask = torch.full((N, N), float("-inf"), device=h.device).triu(1)
        no, do = self.sync_out.reset(B * N, h.device, h.dtype)
        na, da = self.sync_act.reset(B * N, h.device, h.dtype)

        s_out = None
        for _ in range(self.ticks):
            a = self.synapse(torch.cat([z, o], dim=-1))
            A = torch.cat([A[..., 1:], a.unsqueeze(-1)], dim=-1)
            z = self.ln(self.nlm(A.reshape(B * N, self.D, self.M))
                        ).reshape(B, N, self.D)
            zz = spike(z) if self.spike_sync else z
            zf = zz.reshape(B * N, self.D)
            s_out, no, do = self.sync_out.step(zf, no, do)
            s_act, na, da = self.sync_act.step(zf, na, da)
            q = self.w_in(s_act).reshape(B, N, self.d_head)
            att = torch.softmax(
                q @ K.transpose(1, 2) * self.scale + mask, dim=-1)
            o = att @ V                                       # causal lookup
        return self.w_out(s_out.reshape(B, N, -1))


class SyncLM(nn.Module):
    def __init__(self, cfg, use_sync, **head_kw):
        super().__init__()
        self.core = RWKVMini(cfg)
        self.sync_head = SyncTickHead(cfg.n_embd, **head_kw) if use_sync else None

    def forward(self, idx, targets=None):
        m = self.core
        x = m.ln_in(m.emb(idx))
        for b in m.blocks:
            x = b(x)
        x = m.ln_out(x)
        if self.sync_head is not None:
            x = x + self.sync_head(x)
        logits = m.head(x)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def batches_from(data, n_tokens, B, T, seed, device):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(max(1, n_tokens // (B * T))):
        ix = rng.integers(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1:i + T + 1] for i in ix]).astype(np.int64)
        out.append((torch.from_numpy(x).to(device),
                    torch.from_numpy(y).to(device)))
    return out


@torch.no_grad()
def val_ppl(model, batches):
    model.eval()
    tot = ntok = 0
    for x, y in batches:
        _, loss = model(x, y)
        tot += float(loss) * y.numel()
        ntok += y.numel()
    model.train()
    return math.exp(tot / ntok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", type=int, default=1)
    ap.add_argument("--spike-sync", type=int, default=1)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size, n_embd=a.d, n_layer=a.layers)
    torch.manual_seed(a.seed)
    model = SyncLM(cfg, use_sync=bool(a.sync), D=a.D, ticks=a.ticks,
                   pairs=a.pairs, spike_sync=bool(a.spike_sync)).to(dev)
    n_all = sum(p.numel() for p in model.parameters())
    n_head = (sum(p.numel() for p in model.sync_head.parameters())
              if model.sync_head else 0)
    print(f"{'sync-tick head' if a.sync else 'trunk only':<16} d={a.d} "
          f"L={a.layers} D={a.D} T={a.ticks} P={a.pairs} · {dev}")
    print(f"params {n_all:,} (head {n_head:,}, "
          f"{100 * n_head / n_all:.1f}%)")

    train = load_split("train")
    valid = load_split("valid")
    vb = batches_from(valid, 16384, 8, cfg.ctx, 7, dev)   # matches bit_budget
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    os.makedirs("runs-lm", exist_ok=True)
    log = open(f"runs-lm/{a.tag}.jsonl", "a")
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    model.train()
    for step in range(1, a.steps + 1):
        ix = rng.integers(0, len(train) - cfg.ctx - 1, size=a.batch)
        x = torch.from_numpy(np.stack([train[i:i + cfg.ctx] for i in ix])
                             .astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([train[i + 1:i + cfg.ctx + 1] for i in ix])
                             .astype(np.int64)).to(dev)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 250 == 0 or step == a.steps:
            ppl = val_ppl(model, vb)
            rec = dict(step=step, loss=round(float(loss), 4),
                       val_ppl=round(ppl, 4),
                       mins=round((time.time() - t0) / 60, 1))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()

    torch.save({"model": model.state_dict(), "args": vars(a)},
               f"runs-lm/{a.tag}.pt")
    print(f"done · runs-lm/{a.tag}.jsonl · runs-lm/{a.tag}.pt")


if __name__ == "__main__":
    main()
