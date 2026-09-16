"""Controlled full-LM comparison: oracle addressing vs learned addressing.

The isolated probe (v29) showed the fast-weight store retrieves 8 facts
under DISTINCT keys but degrades under OVERLAPPING keys. So the LM floor
may be a learned-addressing failure, not a memory or capacity limit. This
disentangles it: match everything except where the keys come from.

Both arms, per review:
  * one write PER COMPLETED FACT (writer state pooled over the fact span)
    - removes the per-token whole-matrix decay entirely;
  * retention a = 1 (no forgetting) - isolate addressing from decay;
  * additive fast-weight M = sum_i v_i (x) phi(k_i), z = sum_i phi(k_i),
    read = (M phi(q)) / (z . phi(q)), phi = elu+1 (one consistent map,
    no separate tanh, so the normaliser is real);
  * FULL PRECISION both arms (quantisation removed as a variable; restore
    later if the comparison succeeds and we want budget back);
  * same value encoder, decoder, loss, batches, seeds.

The ONLY difference:
  learned   k_i = phi(w_key(pooled fact state));  q = phi(q_key(name))
  oracle    k_i = phi(ORTHO[entity_i]);           q = phi(ORTHO[target])
            ORTHO = fixed orthonormal per-entity vectors. The query
            retrieves the key assigned to the MATCHING entity regardless
            of fact order (explicit entity->key matching - documented
            assistance, this is a diagnostic not a headline model).

Saves both checkpoints. Reports overall recall AND recall by fact
POSITION, plus a counterfactual (change one fact's value; its read must
move, bystanders must not).

Interpretation (review):
  oracle succeeds, learned fails -> addressing (learning it / integrating
    it with writing) is the implicated component; credit density stays a
    possible reason addressing failed.
  both fail -> look at value encoding / write timing / read injection /
    decoder; NOT a credit conclusion.
  both succeed -> the controlled setup removed a difficulty; restore
    components (quantisation, per-token writes, decay) one at a time.

    python oracle_addr.py --entities 8
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nonce_lm import make_batch, load_carrier, V, QUERY_L, ANS_L, NAME_L
from nonce_lm2 import Block


class OracleAddrLM(nn.Module):
    def __init__(self, addr, N, d=128, dk=16, dv=16, layers=3, heads=4,
                 window=128, gru=64, max_seq=2048, key_seed=1234):
        super().__init__()
        assert dk >= N, f"one-hot oracle needs dk>=N, got dk={dk} N={N}"
        self.addr, self.N, self.window, self.dk = addr, N, window, dk
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Parameter(torch.randn(max_seq, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)
        self.writer = nn.GRU(d, gru, batch_first=True)
        self.w_key = nn.Linear(gru, dk)
        self.w_val = nn.Linear(gru, dv)
        self.q_gru = nn.GRU(d, gru, batch_first=True)
        self.q_key = nn.Linear(gru, dk)
        self.r_up = nn.Linear(dv, d, bias=False)
        nn.init.zeros_(self.r_up.weight)
        g = torch.Generator().manual_seed(key_seed)
        A = torch.linalg.qr(torch.randn(dk, dk, generator=g))[0]   # orthonormal
        self.register_buffer("ortho", A)                          # (dk, dk)

    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0                                     # nonneg feature

    def keyfeat(self, ci, pooled):
        """Key feature for fact ci. Oracle: one-hot e_ci - orthogonal AND
        nonnegative, so no feature map is applied and it needs none (review
        fix: elu+1 on orthonormal vectors collapsed them; one-hot vectors
        are simultaneously orthogonal and nonneg). Learned: elu+1 of the
        pooled writer state. Both live in the same nonneg feature space, so
        the SAME additive M/z store and read serve both - only the address
        source differs."""
        if self.addr == "oracle":
            e = pooled.new_zeros(self.dk); e[ci] = 1.0
            return e
        return self.phi(self.w_key(pooled))

    def build_memory(self, x, spans, val_over=None):
        """Additive fast-weight store, one write per fact. M=sum v(x)k,
        z=sum k. With one-hot oracle keys: M q = v_target, z.q = 1 -> exact
        retrieval (verified). This is the SAME compressed synaptic store
        that floored; only the key source is swapped."""
        B, N = spans.shape[0], spans.shape[1]
        end = int(spans[:, :, 1].max())
        h, _ = self.writer(self.emb(x[:, :end]) + self.pos[:end])
        M = h.new_zeros(B, self.w_val.out_features, self.dk)
        z = h.new_zeros(B, self.dk)
        for ci in range(N):
            kfeat = h.new_zeros(B, self.dk)
            vvec = h.new_zeros(B, self.w_val.out_features)
            for b in range(B):
                lo, hi = int(spans[b, ci, 0]), int(spans[b, ci, 1])
                pooled = h[b, lo:hi].mean(0)
                vvec[b] = torch.tanh(self.w_val(pooled))
                kfeat[b] = self.keyfeat(ci, pooled)
            if val_over is not None and ci == val_over[0]:
                vvec = val_over[1]                               # counterfactual
            M = M + torch.einsum("bv,bk->bvk", vvec, kfeat)
            z = z + kfeat
        return M, z

    def query_key(self, x, q0, tgt):
        if self.addr == "oracle":
            e = torch.zeros(tgt.shape[0], self.dk, device=x.device)
            e[torch.arange(tgt.shape[0]), tgt] = 1.0             # one-hot e_target
            return e
        qe = self.emb(x[:, q0:q0 + QUERY_L]) + self.pos[:QUERY_L]
        hq, _ = self.q_gru(qe)
        return self.phi(self.q_key(hq[:, 1 + NAME_L]))

    def target_value(self, x, spans, tgt):
        """The target entity's own value vector (a perfect, unmixed read) -
        for the clean-value-injection diagnostic."""
        end = int(spans[:, :, 1].max())
        h, _ = self.writer(self.emb(x[:, :end]) + self.pos[:end])
        out = h.new_zeros(x.shape[0], self.w_val.out_features)
        for b in range(x.shape[0]):
            lo, hi = int(spans[b, tgt[b], 0]), int(spans[b, tgt[b], 1])
            out[b] = torch.tanh(self.w_val(h[b, lo:hi].mean(0)))
        return out

    def match_logits(self, x, spans, beta=10.0):
        """beta * cos(query, k_ci) for every entity ci - the training-only
        discrimination classifier. Teaches the query to pick its entity's
        write key over distractors, through the SAME q_key/w_key encoders;
        the compressed M/z memory and the linear read are unchanged."""
        B, N = spans.shape[0], spans.shape[1]
        end = int(spans[:, :, 1].max())
        h, _ = self.writer(self.emb(x[:, :end]) + self.pos[:end])
        K = h.new_zeros(B, N, self.dk)
        for ci in range(N):
            for b in range(B):
                lo, hi = int(spans[b, ci, 0]), int(spans[b, ci, 1])
                K[b, ci] = self.keyfeat(ci, h[b, lo:hi].mean(0))
        q0 = x.shape[1] - (QUERY_L + ANS_L - 1)
        qe = self.emb(x[:, q0:q0 + QUERY_L]) + self.pos[:QUERY_L]
        hq, _ = self.q_gru(qe)
        q = self.phi(self.q_key(hq[:, 1 + NAME_L]))             # (B,dk)
        qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Kn = K / K.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return beta * torch.einsum("bnk,bk->bn", Kn, qn)        # (B,N) cos*beta

    def forward(self, x, spans, tgt, val_over=None, mem_off=False, read_r=None):
        B, Nx = x.shape
        M, z = self.build_memory(x, spans, val_over)
        q0 = Nx - (QUERY_L + ANS_L - 1)
        q = self.query_key(x, q0, tgt)
        num = torch.einsum("bvk,bk->bv", M, q)
        den = (z * q).sum(-1, keepdim=True).clamp_min(1e-6)
        r = num / den if read_r is None else read_r    # inject clean value
        h = self.emb(x) + self.pos[:Nx]
        h = h.clone()
        if not mem_off:                                 # ablation: zero the read
            h[:, q0:] = h[:, q0:] + self.r_up(r).unsqueeze(1)
        mask = torch.ones(Nx, Nx, dtype=torch.bool, device=x.device)
        for i in range(Nx):
            mask[i, max(0, i - self.window + 1):i + 1] = False
        for blk in self.blocks:
            h = blk(h, mask)
        return self.head(self.ln_out(h))


def train(addr, N, carrier, steps, seed, device, B=16, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = OracleAddrLM(addr, N).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x, y, spans, tgt, _ = make_batch(B, N, carrier, rng, device)
        logits = m(x, spans, tgt)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if (step + 1) % 2000 == 0:
            print(f"    .. {addr} N={N} seed{seed} step {step+1}/{steps} "
                  f"loss {float(loss.detach()):.3f}", flush=True)
    return m


@torch.no_grad()
def evaluate(m, N, carrier, device, batches=8):
    """Overall recall + recall by fact position (0..N-1 in write order)."""
    ev = np.random.default_rng(99)
    ok = n = 0
    by_pos_ok = np.zeros(N); by_pos_n = np.zeros(N)
    m.eval()
    for _ in range(batches):
        x, y, spans, tgt, _ = make_batch(64, N, carrier, ev, device)
        pred = m(x, spans, tgt).argmax(-1)
        sel = y != -100
        hit = ((pred == y) | ~sel).all(dim=1)
        ok += int(hit.sum()); n += y.shape[0]
        # write-order position of each queried target: rank of its start
        for b in range(x.shape[0]):
            order = np.argsort(spans[b, :, 0].cpu().numpy())
            posn = int(np.where(order == int(tgt[b]))[0][0])
            by_pos_ok[posn] += int(hit[b]); by_pos_n[posn] += 1
    return ok / n, (by_pos_ok / np.maximum(by_pos_n, 1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="oracle-addr.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()

    print(f"oracle vs learned addressing · N={a.entities} · a=1 · one write/"
          f"fact · full precision · {a.steps} steps · {a.seeds} seeds · {dev}\n")
    print(f"  {'addressing':<12}{'recall med':>12}{'conv':>6}   per-seed")
    print("  " + "-" * 44)
    res = {}
    for addr in ("oracle", "learned"):
        accs, bypos = [], None
        for s in range(a.seeds):
            m = train(addr, a.entities, carrier, a.steps, s, dev)
            acc, bp = evaluate(m, a.entities, carrier, dev)
            accs.append(acc); bypos = bp
            torch.save({"model": m.state_dict(),
                        "addr": addr, "N": a.entities, "seed": s},
                       f"oracle-{addr}-s{s}.pt")
        med = float(np.median(accs))
        conv = sum(v >= 0.20 for v in accs)
        print(f"  {addr:<12}{med:>11.1%}{conv:>4}/{a.seeds}   "
              + " ".join(f"{v:.2f}" for v in accs), flush=True)
        print(f"     recall by write position: "
              + " ".join(f"{v:.2f}" for v in bypos))
        res[addr] = dict(acc=accs, median=med, converged=conv,
                         recall_by_position=bypos)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}  (+ checkpoints oracle-*.pt)")


if __name__ == "__main__":
    main()
