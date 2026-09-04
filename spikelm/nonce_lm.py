"""v3 — the nonce-LM: can the model derive a stable memory ADDRESS from
content itself?

v2c's verdict: the memory stores and retrieves compressed candidates, but
free-form learned allocation fails (8.4%) where pinned identity
addressing succeeds (46.7%). Language supplies no identity markers, so
the bridge capability - the ONE new thing this experiment tests - is
content-derived addressing:

    k = f_key(h_entity),   p(slot j) = softmax(k . a_j)

with everything else held at v2b/v2c known-good settings.

TASK. Tiny Shakespeare is carrier text only. Each sequence plants N facts
"\nZEVRIN carries the amber ring.\n" - nonce names are random 6-letter
strings regenerated per sequence (unmemorisable in weights); color/item
come from 8x8 fixed-length lists (64 combos, 6 bits/fact). A carrier gap
of 140 chars (> window 128) separates the last fact from the query
"\nZEVRIN carries the " whose 10-char answer is the loss region. The
queried entity is random, so with N>1 every fact must be stored.

SIDE-CHANNEL GUARD, extended from v2c: the writer GRU is itself a
recurrent memory that could carry facts across the gap and bypass the
slots. Its input therefore ENDS at the last fact - it never sees the gap
or the query - so the K x d_slot x q slot state (v2b sweet spot: 8 slots
x 16 dims x 8 bits = 1024 persistent bits) is provably the only bridge.

ARMS (write machinery identical; only the addressing differs):

  local     no slots: the floor (fact is outside the window at query).
  uniform   gate=1, uniform address: storage without addressing.
  gated     free-form learned write: v2c's mechanism, expected weak at
            N>1 per its 8.4% there.
  oracleid  address pinned to fact index (fact j -> slot j): the
            structural upper bound, v2c's kvaddr ported.
  keyaddr   THE HYPOTHESIS: address = softmax(f_key(h) . slot_keys),
            keys learned end to end from content alone.

DIAGNOSTICS beyond answer accuracy, per review:
  * collision rate: P(same argmax slot | two DIFFERENT entities in one
    sequence) - keyaddr must drive this low without ever seeing an
    entity ID;
  * read-key match (keyaddr, probe only): recompute the key from the
    query-time name with a fresh writer pass over the query prompt; does
    it argmax to the same slot the entity was written to? A_same >>
    A_diff is the language analogue of the v2a gate trace - and it is
    emergent, since nothing trains the read side of the key.
  * seed spread: v2c showed structure removes the basin lottery; if
    keyaddr gets high recall AND tight seeds, addressing structure was
    learned, not lucked into.

PRE-REGISTERED:
  1. local ~ chance on the answer (1/64 exact-fact = 1.6%).
  2. oracleid strong at N=1 and N=4 (the read path is v2c-proven).
  3. THE TEST: keyaddr approaching oracleid with low collisions beats
     gated substantially. keyaddr ~ gated means content-keys are as hard
     as free allocation and the coordinate-structure question stays open.
  4. gated degrades from N=1 to N=4 faster than keyaddr/oracleid.

    python nonce_lm.py --entities 1
    python nonce_lm.py --entities 4
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = " \n.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
STOI = {c: i for i, c in enumerate(CHARS)}
V = len(CHARS)
COLORS = ["amber", "azure", "coral", "ebony", "frost", "olive", "ivory", "sable"]
ITEMS = ["ring", "lamp", "mask", "rope", "harp", "veil", "drum", "bell"]
NAME_L, W = 6, 128
# THE GAP MUST BEAT THE COMPOSED RECEPTIVE FIELD, not the single-layer
# window: with L layers of causal window W, information relays hop-by-hop
# through gap positions up to L*(W-1) = 381 chars. Our first run used
# gap=140 and the no-memory control scored 100% at N=1 - the leak was
# caught by the control before any claim shipped. v2c never tripped this
# because 96 random bits are too much to relay learnably; a 6-bit
# templated fact is not.
GAP = 420
FACT_L = 1 + NAME_L + 13 + 5 + 1 + 4 + 2          # "\nNAME carries the color item.\n"
QUERY_L = 1 + NAME_L + 13                          # "\nNAME carries the "
ANS_L = 10                                         # "color item"
PREGAP = 40                                        # carrier before/between facts


def load_carrier(path="data/tinyshakespeare.txt"):
    txt = open(path).read()
    return np.array([STOI.get(c, STOI[" "]) for c in txt], dtype=np.int64)


def enc(s):
    return [STOI[c] for c in s]


def make_batch(B, N, carrier, rng, device):
    SEQ = N * FACT_L + PREGAP + GAP + QUERY_L + ANS_L
    xs = np.zeros((B, SEQ), dtype=np.int64)
    fact_spans = np.zeros((B, N, 2), dtype=np.int64)
    tgt = rng.integers(0, N, size=B)
    for b in range(B):
        names = set()
        while len(names) < N:
            names.add("".join(chr(65 + c) for c in rng.integers(0, 26, NAME_L)))
        names = list(names)
        combos = [(COLORS[rng.integers(8)], ITEMS[rng.integers(8)])
                  for _ in range(N)]
        gaps = rng.multinomial(PREGAP, [1 / N] * N)
        p = 0
        for j in range(N):
            g = gaps[j]
            off = rng.integers(0, len(carrier) - g - 1)
            xs[b, p:p + g] = carrier[off:off + g]; p += g
            fact = f"\n{names[j]} carries the {combos[j][0]} {combos[j][1]}.\n"
            xs[b, p:p + FACT_L] = enc(fact)
            fact_spans[b, j] = (p, p + FACT_L); p += FACT_L
        off = rng.integers(0, len(carrier) - GAP - 1)
        xs[b, p:p + GAP] = carrier[off:off + GAP]; p += GAP
        t = int(tgt[b])
        xs[b, p:p + QUERY_L] = enc(f"\n{names[t]} carries the "); p += QUERY_L
        xs[b, p:p + ANS_L] = enc(f"{combos[t][0]} {combos[t][1]}")
    x = torch.from_numpy(xs[:, :-1]).to(device)
    y = torch.full((B, SEQ - 1), -100, dtype=torch.long)
    y[:, SEQ - ANS_L - 1:] = torch.from_numpy(xs[:, SEQ - ANS_L:])
    return (x, y.to(device), torch.from_numpy(fact_spans).to(device),
            torch.from_numpy(tgt).to(device), SEQ)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.ln2(x))


def quantise_fixed(s, qbits):
    if qbits is None:
        return s
    levels = 2 ** qbits
    q = torch.round((s + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
    return s + (q - s).detach()


class NonceLM(nn.Module):
    def __init__(self, mode, N, K=8, d=128, d_slot=16, qbits=8, dk=16,
                 layers=3, heads=4, window=W, gru=64, max_seq=1024):
        super().__init__()
        self.mode, self.K, self.window, self.qbits = mode, K, window, qbits
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if mode != "local":
            self.writer = nn.GRU(d, gru, batch_first=True)
            self.w_gate = nn.Linear(gru, 1)
            self.w_val = nn.Linear(gru, d_slot)
            self.slot_up = nn.Linear(d_slot, d, bias=False)
            if mode == "gated":
                self.w_addr = nn.Linear(gru, K)
            if mode == "keyaddr":
                self.w_key = nn.Linear(gru, dk)
                self.slot_keys = nn.Parameter(torch.randn(K, dk) * 0.5)
                self.scale = dk ** -0.5

    def write_region(self, spans):
        return int(spans[:, -1, 1].max())        # writer input ends here

    def addresses(self, h, pre, spans):
        B, T, _ = h.shape
        if self.mode == "uniform":
            return torch.full((B, T, self.K), 1.0 / self.K, device=h.device)
        if self.mode == "gated":
            return torch.softmax(self.w_addr(h), dim=-1)
        if self.mode == "keyaddr":
            k = self.w_key(h)
            return torch.softmax(k @ self.slot_keys.t() * self.scale, dim=-1)
        # oracleid: positions inside fact j -> slot j, else uniform
        a = torch.full((B, T, self.K), 1.0 / self.K, device=h.device)
        for b in range(B):
            for j in range(spans.shape[1]):
                lo, hi = int(spans[b, j, 0]), int(spans[b, j, 1])
                hot = torch.zeros(self.K, device=h.device)
                hot[j % self.K] = 1.0
                a[b, lo:hi] = hot
        return a

    def build_slots(self, x, spans):
        if self.mode == "local":
            return None, None, None
        end = self.write_region(spans)
        pre = x[:, :end]
        e = self.emb(pre) + self.pos[:end]
        h, _ = self.writer(e)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)
        if self.mode == "uniform":
            g = torch.ones_like(g)
        a = self.addresses(h, pre, spans)
        w = g.unsqueeze(-1) * a
        num = torch.einsum("btk,btd->bkd", w, self.w_val(h))
        den = w.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        s = quantise_fixed(torch.tanh(num / den), self.qbits)
        return self.slot_up(s), g, a

    def forward(self, x, spans):
        B, Nx = x.shape
        slots, g, a = self.build_slots(x, spans)
        h = self.emb(x) + self.pos[:Nx]
        K = 0 if slots is None else slots.shape[1]
        if slots is not None:
            h = torch.cat([slots, h], dim=1)
        L = K + Nx
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        for i in range(Nx):
            lo = max(0, i - self.window + 1)
            mask[K + i, K + lo:K + i + 1] = False
            mask[K + i, :K] = False
        mask[:K, :K] = False
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_out(h[:, K:])), a


def diagnostics(model, a, spans, tgt, x):
    """Write-address collisions; read-key match for keyaddr (probe only)."""
    out = {}
    if a is None:
        return out
    B, N = spans.shape[0], spans.shape[1]
    slot_of = np.zeros((B, N), dtype=int)
    for b in range(B):
        for j in range(N):
            lo, hi = int(spans[b, j, 0]), int(spans[b, j, 1])
            slot_of[b, j] = int(a[b, lo:hi].mean(0).argmax())
    if N > 1:
        same = sum(slot_of[b, i] == slot_of[b, j]
                   for b in range(B) for i in range(N) for j in range(i + 1, N))
        pairs = B * N * (N - 1) // 2
        out["collision"] = same / pairs
    if model.mode == "keyaddr":
        with torch.no_grad():
            q0 = x.shape[1] - (QUERY_L + ANS_L - 1)
            qe = model.emb(x[:, q0:q0 + QUERY_L]) + model.pos[:QUERY_L]
            hq, _ = model.writer(qe)                  # fresh state: probe
            kq = model.w_key(hq[:, 1 + NAME_L])       # end of the name
            read_slot = (kq @ model.slot_keys.t()).argmax(-1).cpu().numpy()
        match_t = np.mean([read_slot[b] == slot_of[b, int(tgt[b])]
                           for b in range(B)])
        others = [read_slot[b] == slot_of[b, j] for b in range(B)
                  for j in range(N) if j != int(tgt[b])]
        out["read_match_tgt"] = float(match_t)
        out["read_match_other"] = float(np.mean(others)) if others else None
    return out


def run(mode, N, carrier, steps, seed, device, B=32, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = NonceLM(mode, N).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y, spans, tgt, _ = make_batch(B, N, carrier, rng, device)
        logits, _ = m(x, spans)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(99)
    fact_ok = fact_n = 0
    diags = []
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, tgt, SEQ = make_batch(64, N, carrier, ev, device)
            logits, a = m(x, spans)
            pred = logits.argmax(-1)
            sel = y != -100
            hit = ((pred == y) | ~sel).all(dim=1)     # whole answer exact
            fact_ok += int(hit.sum()); fact_n += y.shape[0]
            d = diagnostics(m, None if a is None else a.cpu(), spans.cpu(),
                            tgt.cpu(), x)
            if d: diags.append(d)
    agg = {}
    for k in (diags[0] if diags else {}):
        vals = [d[k] for d in diags if d.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    return fact_ok / fact_n, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--arms", nargs="+",
                    default=["local", "uniform", "gated", "oracleid", "keyaddr"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()
    out = a.out or f"nonce-n{a.entities}.json"

    print(f"nonce-LM · N={a.entities} facts · window {W} · gap {GAP} · "
          f"budget 8x16x8 = 1024b · exact-fact chance {1/64:.1%}")
    print(f"{a.steps} steps · {a.seeds} seeds · {dev}\n")
    print(f"  {'arm':<10}{'fact med':>10}{'conv':>6}{'collide':>9}"
          f"{'readT':>7}{'readO':>7}   per-seed")
    print("  " + "-" * 70)
    res = {}
    for mode in a.arms:
        accs, dg = [], []
        for s in range(a.seeds):
            acc, d = run(mode, a.entities, carrier, a.steps, s, dev)
            accs.append(acc); dg.append(d)
        med = float(np.median(accs))
        conv = sum(v >= 0.20 for v in accs)
        def m_(k):
            vs = [d[k] for d in dg if d.get(k) is not None]
            return f"{np.mean(vs):.2f}" if vs else "—"
        seeds_s = " ".join(f"{v:.2f}" for v in accs)
        print(f"  {mode:<10}{med:>9.1%}{conv:>4}/{a.seeds}{m_('collision'):>9}"
              f"{m_('read_match_tgt'):>7}{m_('read_match_other'):>7}   {seeds_s}",
              flush=True)
        res[mode] = dict(acc=accs, median=med, converged=conv, diags=dg)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
