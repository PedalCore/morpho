"""v3.2 — read ablation + entity-count pressure on the fixed-size memory.

PART A (the mechanism nail-down, per review): the ll arm's soft read
scored 88.4% while its argmax-slot accuracy was 0.62 - the interpretation
was that SOFT retrieval pools evidence across keys and reads through
collisions the hard oracle read cannot. Demonstrated, not just argued, by
evaluating the SAME trained memory under a read-temperature sweep:

    a = softmax(q.k / tau)      tau: 1.0 (native) -> 0.25 -> 0.1 -> hard

If softness is what rescues collisions, recall must fall monotonically as
the read approaches hard argmax - on identical weights, identical data.

PART B (the load-bearing question): hold the memory FIXED (K=8 slots x
(16+16) dims x 8 bits = 2048 persistent bits) and sweep semantic demand:

    N in {1, 2, 4, 8, 16, 32}   (32 facts into 8 slots: 4x oversubscribed)

tracking recall, collision rate, slots actually used, read entropy, and
the ledger's bits per fact. The question is HOW the system fails as
demand exceeds capacity:

    smooth degradation  -> a bounded associative cache (graceful)
    sudden collapse     -> brittle routing dynamics
    recall above what disjoint slots could give at N >> K -> sharing/
                           compression across entities: the doorway to
                           hierarchy

The frozen scope from review holds throughout: this is a toy language
setting; the claim is content-derived writes and reads for nonce
entities beyond the audited receptive field - not KV-cache replacement.

    python nonce_pressure.py --part a
    python nonce_pressure.py --part b
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from nonce_lm import make_batch, load_carrier, V
from nonce_lm2 import KVNonceLM


def train_ll(N, carrier, steps, seed, device, B=16, lr=1e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = KVNonceLM("learned", "learned", N, max_seq=2048).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x, y, spans, tgt, _ = make_batch(B, N, carrier, rng, device)
        logits, aux, _ = m(x, spans, tgt)
        loss = (F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                ignore_index=-100)
                + 0.01 * aux["balance"] + 0.1 * aux["keymatch"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    return m


@torch.no_grad()
def evaluate(m, N, carrier, device, batches=8):
    ev = np.random.default_rng(99)
    ok = n = 0
    coll, ents, used_n = [], [], []
    for _ in range(batches):
        x, y, spans, tgt, _ = make_batch(64, N, carrier, ev, device)
        logits, _, extra = m(x, spans, tgt)
        pred = logits.argmax(-1)
        sel = y != -100
        ok += int((((pred == y) | ~sel).all(dim=1)).sum()); n += y.shape[0]
        if extra is not None:
            u = extra["used"].cpu().numpy()
            if N > 1:
                same = sum(u[b, i] == u[b, j] for b in range(len(u))
                           for i in range(N) for j in range(i + 1, N))
                coll.append(same / (len(u) * N * (N - 1) // 2))
            used_n.append(np.mean([len(set(u[b])) for b in range(len(u))]))
            a = extra["read_att"]
            if a is not None:
                ent = -(a * a.clamp_min(1e-9).log()).sum(-1).mean()
                ents.append(float(ent))
    return (ok / n,
            float(np.mean(coll)) if coll else 0.0,
            float(np.mean(used_n)) if used_n else 0.0,
            float(np.mean(ents)) if ents else 0.0)


def part_a(carrier, device, steps, seeds):
    print("PART A — same trained memory, read softness swept\n")
    print(f"  {'read':<16}" + "".join(f"{f'seed{s}':>8}" for s in range(seeds))
          + f"{'median':>9}")
    print("  " + "-" * (16 + 8 * seeds + 9))
    models = [train_ll(4, carrier, steps, s, device) for s in range(seeds)]
    res = {}
    settings = [("soft tau=1.0", 1.0, False), ("soft tau=0.5", 0.5, False),
                ("soft tau=0.25", 0.25, False), ("soft tau=0.1", 0.1, False),
                ("HARD argmax", 1.0, True)]
    for label, tau, hard in settings:
        accs = []
        for m in models:
            m.read_tau, m.read_hard = tau, hard
            acc, _, _, _ = evaluate(m, 4, carrier, device)
            accs.append(acc)
            m.read_tau, m.read_hard = 1.0, False
        med = float(np.median(accs))
        print(f"  {label:<16}" + "".join(f"{a:>8.2f}" for a in accs)
              + f"{med:>8.1%}", flush=True)
        res[label] = {"acc": accs, "median": med}
    return res


def part_b(carrier, device, steps, seeds):
    print("\nPART B — fixed 2048-bit memory, semantic demand swept\n")
    print(f"  {'N':>4}{'bits/fact':>11}{'recall med':>12}{'collide':>9}"
          f"{'slots used':>12}{'read H':>8}   per-seed")
    print("  " + "-" * 72)
    res = {}
    for N in (1, 2, 4, 8, 16, 32):
        accs, cs, us, hs = [], [], [], []
        for s in range(seeds):
            m = train_ll(N, carrier, steps, s, device)
            acc, c, u, h = evaluate(m, N, carrier, device)
            accs.append(acc); cs.append(c); us.append(u); hs.append(h)
            del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        med = float(np.median(accs))
        print(f"  {N:>4}{2048 // N:>10}b{med:>11.1%}{np.mean(cs):>9.2f}"
              f"{np.mean(us):>12.1f}{np.mean(hs):>8.2f}   "
              + " ".join(f"{a:.2f}" for a in accs), flush=True)
        res[str(N)] = dict(acc=accs, median=med, collision=float(np.mean(cs)),
                           slots_used=float(np.mean(us)),
                           read_entropy=float(np.mean(hs)))
    # spot-check the floor at maximum pressure: no-memory arm at N=32
    accs = []
    for s in range(min(2, seeds)):
        torch.manual_seed(s)
        rng = np.random.default_rng(s)
        m = KVNonceLM("none", "none", 32, max_seq=2048).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            x, y, spans, tgt, _ = make_batch(16, 32, carrier, rng, device)
            logits, aux, _ = m(x, spans, tgt)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
        acc, _, _, _ = evaluate(m, 32, carrier, device)
        accs.append(acc); del m
    print(f"  floor (no memory) at N=32: " + " ".join(f"{a:.2f}" for a in accs))
    res["floor_n32"] = accs
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default="nonce-pressure.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    carrier = load_carrier()
    print(f"read ablation + pressure sweep · {a.steps} steps · {a.seeds} seeds "
          f"· {dev} · memory fixed at 2048 bits\n")
    res = {}
    if a.part in ("a", "both"):
        res["read_ablation"] = part_a(carrier, dev, a.steps, a.seeds)
    if a.part in ("b", "both"):
        res["pressure"] = part_b(carrier, dev, a.steps, a.seeds)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
