"""Localise WHERE learned addressing breaks (review's three causes).

If the learned arm floors while oracle is 100%, the defensible statement
is "ideal addresses rescue recall in this controlled setup." This probe
then says WHICH part of learned addressing failed, from a saved
checkpoint - making the next intervention specific:

  1. write-key overlap : different entities receive similar keys
     -> mean off-diagonal cosine among the N per-fact write keys.
        High = collisions (the v25 router-collapse failure mode).
  2. query mismatch     : the query key fails to match its entity's key
     -> cosine(query key, target write key) vs cosine(query, distractors).
        Low target / high distractor = the name->key map is inconsistent
        between write and read.
  3. read interference  : matching exists but distractors contaminate
     -> normalised retrieval weight w_i = (k_i . q)/sum_j(k_j . q); report
        mass on target vs mean distractor. Target ~ 1/N = uniform
        contamination even if matching is fine.

Also reports per-position recall WITH sample counts (review: even recall
needs counts), and the counterfactual (change target value; its read must
move, a bystander must not).

    python addr_diag.py --ckpt oracle-learned-s0.pt
"""

import argparse
import json

import numpy as np
import torch

from nonce_lm import make_batch, load_carrier, V, QUERY_L, ANS_L, NAME_L
from oracle_addr import OracleAddrLM


def cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)


@torch.no_grad()
def diagnose(m, N, carrier, device, batches=8):
    ev = np.random.default_rng(99)
    overlap, qt, qd, wt, wd, rank, margin, sens_self, sens_cross = (
        [], [], [], [], [], [], [], [], [])
    ok = 0; n = 0
    pos_ok = np.zeros(N); pos_n = np.zeros(N)
    m.eval()
    for _ in range(batches):
        x, y, spans, tgt, _ = make_batch(64, N, carrier, ev, device)
        B, Nx = x.shape
        end = int(spans[:, :, 1].max())
        h, _ = m.writer(m.emb(x[:, :end]) + m.pos[:end])
        # per-fact write keys/values THROUGH THE MODEL'S OWN keyfeat, so an
        # oracle checkpoint reads as a positive control for the diagnostic
        Kf = h.new_zeros(B, N, m.dk); Vf = h.new_zeros(B, N, m.w_val.out_features)
        for ci in range(N):
            for b in range(B):
                lo, hi = int(spans[b, ci, 0]), int(spans[b, ci, 1])
                p = h[b, lo:hi].mean(0)
                Kf[b, ci] = m.keyfeat(ci, p); Vf[b, ci] = torch.tanh(m.w_val(p))
        q0 = Nx - (QUERY_L + ANS_L - 1)
        q = m.query_key(x, q0, tgt)                             # model's own query
        ar = torch.arange(B, device=device)
        dmask = torch.ones(B, N, device=device); dmask[ar, tgt] = 0
        # 1. write-key overlap (mean off-diagonal cosine)
        Kn = Kf.norm(dim=-1, keepdim=True)
        Gc = torch.einsum("bik,bjk->bij", Kf, Kf) / (Kn * Kn.transpose(1, 2) + 1e-9)
        overlap += ((Gc.sum((1, 2)) - N) / (N * (N - 1))).tolist()
        # 2. query vs target/distractor write key (cosine)
        qt += cos(q, Kf[ar, tgt]).tolist()
        qd += ((cos(q.unsqueeze(1).expand(-1, N, -1), Kf) * dmask).sum(1)
               / (N - 1)).tolist()
        # raw dot products (WITH magnitude): rank + margin catch the single
        # distractor that wins, which averaged cosines hide
        raw = torch.einsum("bnk,bk->bn", Kf, q)                 # k_i . q
        rt = raw[ar, tgt]
        rank += (raw > rt.unsqueeze(1)).sum(1).tolist()         # 0 = target wins
        top_d = (raw + (1 - dmask) * -1e9).max(1).values        # best distractor
        margin += (rt - top_d).tolist()
        # 3. normalised retrieval weights (contamination)
        w = raw / raw.sum(-1, keepdim=True).clamp_min(1e-6)
        wt += w[ar, tgt].tolist()
        wd += ((w * dmask).sum(1) / (N - 1)).tolist()
        # recall + per position
        pred = m(x, spans, tgt).argmax(-1); sel = y != -100
        hit = ((pred == y) | ~sel).all(dim=1)
        ok += int(hit.sum()); n += B
        for b in range(B):
            order = np.argsort(spans[b, :, 0].cpu().numpy())
            pp = int(np.where(order == int(tgt[b]))[0][0])
            pos_ok[pp] += int(hit[b]); pos_n[pp] += 1
        # counterfactual (same target query, magnitude-normalised ||dr||):
        #  self  - swap the TARGET's value: target read SHOULD move (~w_tgt)
        #  cross - swap a DISTRACTOR's value: target read should NOT move
        #          (~w_distractor); nonzero = read interference
        r0 = torch.einsum("bnv,bn->bv", Vf, w)
        dv = torch.tanh(torch.randn_like(Vf[ar, tgt]))
        pert = (dv - Vf[ar, tgt]).norm(dim=-1).clamp_min(1e-6)
        Vs = Vf.clone(); Vs[ar, tgt] = dv
        sens_self += ((torch.einsum("bnv,bn->bv", Vs, w) - r0).norm(dim=-1)
                      / pert).tolist()
        dj = (tgt + 1) % N
        Vc = Vf.clone(); Vc[ar, dj] = torch.tanh(torch.randn_like(Vf[ar, dj]))
        pert_c = (Vc[ar, dj] - Vf[ar, dj]).norm(dim=-1).clamp_min(1e-6)
        sens_cross += ((torch.einsum("bnv,bn->bv", Vc, w) - r0).norm(dim=-1)
                       / pert_c).tolist()
    return dict(
        recall=ok / n,
        write_overlap=float(np.mean(overlap)),
        query_target=float(np.mean(qt)), query_distractor=float(np.mean(qd)),
        target_rank=float(np.mean(rank)), target_margin=float(np.mean(margin)),
        read_mass_target=float(np.mean(wt)),
        read_mass_distractor=float(np.mean(wd)),
        sens_self=float(np.mean(sens_self)),
        sens_cross=float(np.mean(sens_cross)),
        pos_recall=(pos_ok / np.maximum(pos_n, 1)).tolist(),
        pos_counts=pos_n.astype(int).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()
    ck = torch.load(a.ckpt, map_location=dev)
    N = ck["N"]
    m = OracleAddrLM(ck["addr"], N).to(dev)
    m.load_state_dict(ck["model"])
    d = diagnose(m, N, carrier, dev)

    print(f"{a.ckpt} · addr={ck['addr']} N={N} · recall {d['recall']:.1%}\n")
    print("  symptoms of learned addressing (related, not independent causes):")
    print(f"    1. write-key overlap (off-diag cos):  {d['write_overlap']:+.3f}"
          f"   (high -> entities collide)")
    print(f"    2. query match  target {d['query_target']:+.3f}"
          f"  vs distractor {d['query_distractor']:+.3f}   ·  "
          f"target rank {d['target_rank']:.2f}/{N-1}  margin {d['target_margin']:+.3f}")
    print(f"    3. read mass    target {d['read_mass_target']:.3f}"
          f"  vs distractor {d['read_mass_distractor']:.3f}"
          f"   (target~{1/N:.2f} -> uniform contamination)")
    print(f"\n  counterfactual (||dr||/||dv||): self {d['sens_self']:.3f}"
          f"  cross {d['sens_cross']:.3f}"
          f"   (clean = self high, cross ~0)")
    print("  per-position recall (count):")
    print("    " + "  ".join(f"p{i}:{r:.2f}({c})"
                             for i, (r, c) in enumerate(
                                 zip(d['pos_recall'], d['pos_counts']))))
    out = a.out or a.ckpt.replace(".pt", "-diag.json")
    json.dump(d, open(out, "w"), indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
