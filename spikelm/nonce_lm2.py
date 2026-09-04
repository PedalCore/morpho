"""v3.1 — a proper key/value associative memory, with write and read
stages isolated.

v25 decomposed the failure: the writer's key router collapses (91%
collisions), AND perfect write addressing only reaches 27.8% because the
reader cannot tell which slot holds the queried name. Two failures, two
fixes, and a 2x2 grid to isolate them (design frozen in review):

    slot i stores  M_i = (k_i, v_i)
    write:  k = f_key(h), v = f_value(h), slot chosen by a learned
            router with a LOAD-BALANCING loss (its only job: stop the
            one-bucket collapse)
    read:   q = f_query(queried name),  a = softmax(q . k_i),
            r = sum_i a_i v_i  - injected into the reader as a
            zero-init residual over the query region

Slot identity now means nothing: ZEVRIN may live in slot 3 today and
slot 1 tomorrow; the stored key beside the fact is what finds it. The
transformer never sees slots as positions - retrieval happens ONLY
through the explicit associative read, so the mechanism under test is
not confounded with positional attention over a prefix.

ARMS (the 2x2, plus the floor):

  local   no memory                       harness floor (must be ~1.6%)
  oo      oracle write / oracle read      task ceiling
  oc      oracle write / learned read     can the reader associate?
  lo      learned write / oracle read     can the writer allocate?
  ll      learned write / learned read    the full hypothesis

Oracle write = slot j for fact j (appearance order). Oracle read = the
slot the writer ACTUALLY used most for the target's span (fair for
learned writers). Learned arms carry:
  L = L_task + 0.01 * L_balance          (learned-write arms)
            + 0.1  * L_keymatch          (learned-read arms)
L_balance is the coefficient-of-variation loss on router importance (the
MoE remedy for one-bucket collapse). L_keymatch is InfoNCE over the
sequence's own entities: pull q(query name) toward the write-time key of
the SAME name, push from the other entities' keys - content-derived
throughout; no entity IDs exist anywhere.

Persistent ledger: K x (dk + dv) x q = 8 x 32 x 8 = 2048 bits, keys and
values both quantised (fixed grid, tanh-bounded).

DIAGNOSTICS: write collision rate; read attention accuracy
P(argmax a = target's slot); fact recall; per-seed distributions.

PRE-REGISTERED:
  1. local ~ 1.6%; oo well above v25's oracleid 27.8% (the explicit read
     removes the association bottleneck) - if oo is not strong, the
     injection pathway is broken and nothing else is interpretable.
  2. oc isolates the reader: oc >> 27.8% means learned association
     works given clean writes.
  3. lo isolates the writer: collision rate well below 0.91 means the
     balance loss defeats the collapse.
  4. THE TEST: ll approaching oo with low collisions and high read
     accuracy = learned content-derived association, the empty box in
     the chain. ll ~ v25's 17.6% despite 1-3 passing = the failures
     interact, and that interaction becomes the finding.

    python nonce_lm2.py --entities 4
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nonce_lm import (CHARS, V, W, GAP, NAME_L, FACT_L, QUERY_L, ANS_L,
                      load_carrier, make_batch, Block, quantise_fixed)


class KVNonceLM(nn.Module):
    def __init__(self, write_mode, read_mode, N, K=8, d=128, dk=16, dv=16,
                 qbits=8, layers=3, heads=4, window=W, gru=64, max_seq=1024):
        super().__init__()
        self.write_mode, self.read_mode = write_mode, read_mode
        self.K, self.window, self.qbits = K, window, qbits
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        if write_mode != "none":
            self.writer = nn.GRU(d, gru, batch_first=True)
            self.w_gate = nn.Linear(gru, 1)
            self.w_key = nn.Linear(gru, dk)
            self.w_val = nn.Linear(gru, dv)
            self.w_route = nn.Linear(gru, K)
            self.q_gru = nn.GRU(d, gru, batch_first=True)
            self.q_key = nn.Linear(gru, dk)
            self.r_up = nn.Linear(dv, d, bias=False)
            nn.init.zeros_(self.r_up.weight)     # exact no-op at step 0
            self.scale = dk ** -0.5

    def write(self, x, spans):
        """Returns slot keys (B,K,dk), values (B,K,dv), router probs, gate."""
        end = int(spans[:, -1, 1].max())
        e = self.emb(x[:, :end]) + self.pos[:end]
        h, _ = self.writer(e)
        g = torch.sigmoid(self.w_gate(h)).squeeze(-1)          # (B,T)
        route = torch.softmax(self.w_route(h), dim=-1)          # (B,T,K)
        if self.write_mode == "oracle":
            B, T = g.shape
            route = torch.full_like(route, 1.0 / self.K)
            for b in range(B):
                for j in range(spans.shape[1]):
                    lo, hi = int(spans[b, j, 0]), int(spans[b, j, 1])
                    hot = torch.zeros(self.K, device=x.device)
                    hot[j % self.K] = 1.0
                    route[b, lo:hi] = hot
        w = g.unsqueeze(-1) * route                             # (B,T,K)
        den = w.sum(1).unsqueeze(-1).clamp_min(1e-6)
        keys = torch.einsum("btk,bte->bke", w, self.w_key(h)) / den
        vals = torch.einsum("btk,bte->bke", w, self.w_val(h)) / den
        keys = quantise_fixed(torch.tanh(keys), self.qbits)
        vals = quantise_fixed(torch.tanh(vals), self.qbits)
        return keys, vals, route, g, h

    def query_key(self, x, q0):
        qe = self.emb(x[:, q0:q0 + QUERY_L]) + self.pos[:QUERY_L]
        hq, _ = self.q_gru(qe)
        return self.q_key(hq[:, 1 + NAME_L])                    # end of name

    def forward(self, x, spans, tgt):
        B, Nx = x.shape
        aux = {"balance": x.new_zeros(()).float(),
               "keymatch": x.new_zeros(()).float()}
        h = self.emb(x) + self.pos[:Nx]
        r_att = None
        if self.write_mode != "none":
            keys, vals, route, g, hw = self.write(x, spans)
            q0 = Nx - (QUERY_L + ANS_L - 1)
            # which slot did the writer actually use for each fact?
            B_, N_ = spans.shape[0], spans.shape[1]
            used = torch.zeros(B_, N_, dtype=torch.long, device=x.device)
            for b in range(B_):
                for j in range(N_):
                    lo, hi = int(spans[b, j, 0]), int(spans[b, j, 1])
                    used[b, j] = route[b, lo:hi].mean(0).argmax()
            if self.read_mode == "oracle":
                a = F.one_hot(used[torch.arange(B_), tgt], self.K).float()
            else:
                q = self.query_key(x, q0)
                a = torch.softmax(
                    torch.einsum("be,bke->bk", q, keys) * self.scale, dim=-1)
                # key-match InfoNCE over this sequence's own entities:
                # per-entity write keys from the writer's hidden states
                ent_k = torch.zeros(B_, N_, q.shape[-1], device=x.device)
                for b in range(B_):
                    for j in range(N_):
                        lo, hi = int(spans[b, j, 0]), int(spans[b, j, 1])
                        ent_k[b, j] = self.w_key(hw[b, lo:hi]).mean(0)
                logits_kc = torch.einsum("be,bne->bn", q, ent_k) * self.scale
                aux["keymatch"] = F.cross_entropy(logits_kc, tgt)
            if self.write_mode == "learned":
                imp = route.mean(dim=(0, 1))                    # (K,)
                aux["balance"] = self.K * (imp ** 2).sum() - 1.0
            r = torch.einsum("bk,bke->be", a, vals)
            h = h.clone()
            h[:, q0:] = h[:, q0:] + self.r_up(r).unsqueeze(1)
            r_att = a
        mask = torch.ones(Nx, Nx, dtype=torch.bool, device=x.device)
        for i in range(Nx):
            lo = max(0, i - self.window + 1)
            mask[i, lo:i + 1] = False
        for blk in self.blocks:
            h = blk(h, mask)
        extra = None
        if self.write_mode != "none":
            extra = dict(route=route, used=used, read_att=r_att)
        return self.head(self.ln_out(h)), aux, extra


ARMS = [("local", "none", "none"),
        ("oo", "oracle", "oracle"),
        ("oc", "oracle", "learned"),
        ("lo", "learned", "oracle"),
        ("ll", "learned", "learned")]


def run(write_mode, read_mode, N, carrier, steps, seed, device, B=32, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = KVNonceLM(write_mode, read_mode, N).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y, spans, tgt, _ = make_batch(B, N, carrier, rng, device)
        logits, aux, _ = m(x, spans, tgt)
        loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                ignore_index=-100)
                + 0.01 * aux["balance"] + 0.1 * aux["keymatch"])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    ev = np.random.default_rng(99)
    ok = n = 0
    coll, ratt = [], []
    m.eval()
    with torch.no_grad():
        for _ in range(8):
            x, y, spans, tgt, _ = make_batch(64, N, carrier, ev, device)
            logits, _, extra = m(x, spans, tgt)
            pred = logits.argmax(-1)
            sel = y != -100
            ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
            if extra is not None and N > 1:
                u = extra["used"].cpu().numpy()
                same = sum(u[b, i] == u[b, j] for b in range(len(u))
                           for i in range(N) for j in range(i + 1, N))
                coll.append(same / (len(u) * N * (N - 1) // 2))
                if extra["read_att"] is not None:
                    a = extra["read_att"]
                    hit = (a.argmax(-1) ==
                           extra["used"][torch.arange(len(u)), tgt]).float()
                    ratt.append(float(hit.mean()))
    return (ok / n,
            float(np.mean(coll)) if coll else None,
            float(np.mean(ratt)) if ratt else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()
    out = a.out or f"noncekv-n{a.entities}.json"

    print(f"KV nonce-LM · N={a.entities} · gap {GAP} · ledger 8x(16+16)x8 = "
          f"2048b · chance {1/64:.1%} · {a.steps} steps · {a.seeds} seeds · {dev}\n")
    print(f"  {'arm':<7}{'fact med':>10}{'conv':>6}{'collide':>9}{'readacc':>9}"
          f"   per-seed")
    print("  " + "-" * 62)
    res = {}
    for name, wm, rm in ARMS:
        accs, cs, rs = [], [], []
        for s in range(a.seeds):
            acc, c, r = run(wm, rm, a.entities, carrier, a.steps, s, dev)
            accs.append(acc)
            if c is not None: cs.append(c)
            if r is not None: rs.append(r)
        med = float(np.median(accs))
        conv = sum(v >= 0.20 for v in accs)
        c_s = f"{np.mean(cs):.2f}" if cs else "—"
        r_s = f"{np.mean(rs):.2f}" if rs else "—"
        seeds_s = " ".join(f"{v:.2f}" for v in accs)
        print(f"  {name:<7}{med:>9.1%}{conv:>4}/{a.seeds}{c_s:>9}{r_s:>9}"
              f"   {seeds_s}", flush=True)
        res[name] = dict(acc=accs, median=med, converged=conv,
                         collision=cs, read_acc=rs)
    json.dump(res, open(out, "w"), indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
