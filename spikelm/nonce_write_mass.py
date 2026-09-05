"""v27b — was the lo arm's oracle defined in the wrong coordinates?

The temperature ablation killed the soft-pooling story (~2 points). The
remaining suspect for the ll-over-lo gap is the ORACLE itself: it
one-hots to the slot with the target's argmax ROUTE mass, but writes are
soft, so the target's VALUE mass (gate x route weighted) can land mostly
elsewhere. This instruments that directly rather than narrating it.

Four reads on the SAME trained memory (probe asserted bit-identical to
the model's own forward for the native case):

  route-oracle   one-hot argmax of mean route over the target span
                 (v3.1's lo/read-acc reference)
  mass-oracle    one-hot argmax of target VALUE MASS received per slot
                 (sum of gate x route over the target span)
  learned soft   the model's own read
  learned hard   one-hot argmax of the model's own read

If mass-oracle ~ learned >> route-oracle, the lo deficit was an
instrumentation artifact: an oracle in the wrong coordinates for a
soft-write system, and v3.1's 0.62 "read accuracy" measured agreement
with a non-canonical reference.

Also reported, replacing the retired read-accuracy metric:
  * mass share: fraction of the target's write mass in its top slot
    (how soft the writes actually are);
  * counterfactual Delta-L: answer loss with the target's write
    contribution zeroed vs full memory - does the retrieved content
    CARRY the fact, regardless of which slot anyone designates.

Same harness as the pressure sweep (B=16); levels comparable within this
file only.

    python nonce_write_mass.py
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from nonce_lm import make_batch, load_carrier, V, QUERY_L, ANS_L
from nonce_pressure import train_ll


@torch.no_grad()
def probe(m, x, spans, tgt, read="native", zero_target=False):
    """Mirror of KVNonceLM.forward with read override + write ablation."""
    B, Nx = x.shape
    end = int(spans[:, -1, 1].max())
    e = m.emb(x[:, :end]) + m.pos[:end]
    hw, _ = m.writer(e)
    g = torch.sigmoid(m.w_gate(hw)).squeeze(-1)
    route = torch.softmax(m.w_route(hw), dim=-1)
    w = g.unsqueeze(-1) * route
    if zero_target:
        for b in range(B):
            lo, hi = int(spans[b, tgt[b], 0]), int(spans[b, tgt[b], 1])
            w[b, lo:hi] = 0.0
    den = w.sum(1).unsqueeze(-1).clamp_min(1e-6)
    keys = torch.einsum("btk,bte->bke", w, m.w_key(hw)) / den
    vals = torch.einsum("btk,bte->bke", w, m.w_val(hw)) / den
    from nonce_lm2 import quantise_fixed
    keys = quantise_fixed(torch.tanh(keys), m.qbits)
    vals = quantise_fixed(torch.tanh(vals), m.qbits)

    q0 = Nx - (QUERY_L + ANS_L - 1)
    mass = torch.zeros(B, m.K, device=x.device)
    routm = torch.zeros(B, m.K, device=x.device)
    for b in range(B):
        lo, hi = int(spans[b, tgt[b], 0]), int(spans[b, tgt[b], 1])
        mass[b] = w[b, lo:hi].sum(0)
        routm[b] = route[b, lo:hi].mean(0)
    if read == "route_oracle":
        a = F.one_hot(routm.argmax(-1), m.K).float()
    elif read == "mass_oracle":
        a = F.one_hot(mass.argmax(-1), m.K).float()
    else:
        q = m.query_key(x, q0)
        a = torch.softmax(torch.einsum("be,bke->bk", q, keys) * m.scale, -1)
        if read == "hard":
            a = F.one_hot(a.argmax(-1), m.K).float()
    r = torch.einsum("bk,bke->be", a, vals)
    h = m.emb(x) + m.pos[:Nx]
    h = h.clone()
    h[:, q0:] = h[:, q0:] + m.r_up(r).unsqueeze(1)
    mask = torch.ones(Nx, Nx, dtype=torch.bool, device=x.device)
    for i in range(Nx):
        lo = max(0, i - m.window + 1)
        mask[i, lo:i + 1] = False
    for blk in m.blocks:
        h = blk(h, mask)
    logits = m.head(m.ln_out(h))
    share = (mass.max(-1).values / mass.sum(-1).clamp_min(1e-9)).mean()
    return logits, float(share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="nonce-write-mass.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()

    print(f"write-mass oracle test · N=4 · {a.steps} steps · {a.seeds} seeds "
          f"· {dev} · B=16 harness\n")
    models = [train_ll(4, carrier, a.steps, s, dev) for s in range(a.seeds)]

    # probe integrity: native probe must equal the model's own forward
    ev = np.random.default_rng(7)
    x, y, spans, tgt, _ = make_batch(8, 4, carrier, ev, dev)
    ref, _, _ = models[0](x, spans, tgt)
    got, _ = probe(models[0], x, spans, tgt, read="native")
    d = float((ref - got).abs().max())
    print(f"probe vs forward: max diff {d:.2e} (must be 0)\n")
    assert d < 1e-5

    reads = ["route_oracle", "mass_oracle", "native", "hard"]
    print(f"  {'read':<14}" + "".join(f"{f'seed{s}':>8}" for s in range(a.seeds))
          + f"{'median':>9}")
    print("  " + "-" * (14 + 8 * a.seeds + 9))
    res = {}
    ev = np.random.default_rng(99)
    batches = [make_batch(64, 4, carrier, ev, dev) for _ in range(8)]
    shares = []
    for read in reads:
        accs = []
        for m in models:
            ok = n = 0
            for x, y, spans, tgt, _ in batches:
                logits, share = probe(m, x, spans, tgt, read=read)
                pred = logits.argmax(-1)
                sel = y != -100
                ok += int((((pred == y) | ~sel).all(dim=1)).sum())
                n += y.shape[0]
                if read == "native":
                    shares.append(share)
            accs.append(ok / n)
        med = float(np.median(accs))
        print(f"  {read:<14}" + "".join(f"{v:>8.2f}" for v in accs)
              + f"{med:>8.1%}", flush=True)
        res[read] = {"acc": accs, "median": med}

    # counterfactual: remove the target's write contribution entirely
    accs = []
    for m in models:
        ok = n = 0
        for x, y, spans, tgt, _ in batches:
            logits, _ = probe(m, x, spans, tgt, read="native", zero_target=True)
            pred = logits.argmax(-1)
            sel = y != -100
            ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
        accs.append(ok / n)
    med = float(np.median(accs))
    print(f"  {'target-removed':<14}" + "".join(f"{v:>8.2f}" for v in accs)
          + f"{med:>8.1%}   <- counterfactual: memory must carry the fact")
    res["target_removed"] = {"acc": accs, "median": med}
    res["target_mass_top_slot_share"] = float(np.mean(shares))
    print(f"\n  target write mass in its top slot: {np.mean(shares):.2f} "
          f"(1.0 = writes are effectively hard)")
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
