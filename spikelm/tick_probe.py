"""What does each internal tick actually do in the trained SyncLM?

The tick loop's claim is iterative retrieval: sync state -> query ->
attend -> refined state -> NEW query. If that is happening, three things
are measurable on the trained checkpoint, and if none of them move, the
head learned a one-shot lookup wearing a loop costume.

  1. DOES THE QUERY MOVE?  Total-variation distance between consecutive
     ticks' attention distributions, per position. A loop doing multi-hop
     retrieval attends somewhere new each tick; a degenerate one repeats
     the same lookup (TV ~ 0 after tick 1).

  2. DOES THE PREDICTION IMPROVE?  Read the sync representation out at
     EVERY tick, not just the last, and score val perplexity through the
     same w_out and head. Parity showed chance-chance-then-solved across
     ticks; the language analogue is ppl falling tick over tick.

  3. DO THE SPIKES CHANGE?  Fraction of neurons whose spike bit flips
     between consecutive ticks. Frozen spikes mean the recurrent state
     converged instantly and the ticks are idle.

Mirrors SyncTickHead.forward exactly (asserted against it), instrumented.

    python tick_probe.py --ckpt runs-lm/sync.pt
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from spikelm.data import get_tokenizer, load_split
from spikelm.model import Config
from sync_lm import SyncLM, spike, batches_from


@torch.no_grad()
def probe_forward(model, x):
    """SyncTickHead.forward with per-tick instrumentation."""
    m = model.core
    h = m.ln_in(m.emb(x))
    for b in m.blocks:
        h = b(h)
    h = m.ln_out(h)                                        # trunk states

    hd = model.sync_head
    B, N, _ = h.shape
    z = torch.tanh(hd.seed(h))
    A = h.new_zeros(B, N, hd.D, hd.M)
    o = h.new_zeros(B, N, hd.d_head)
    K, V = hd.k(h), hd.v(h)
    mask = torch.full((N, N), float("-inf"), device=h.device).triu(1)
    no, do = hd.sync_out.reset(B * N, h.device, h.dtype)
    na, da = hd.sync_act.reset(B * N, h.device, h.dtype)

    per_tick = {"logits": [], "att": [], "spikes": []}
    for _ in range(hd.ticks):
        a = hd.synapse(torch.cat([z, o], dim=-1))
        A = torch.cat([A[..., 1:], a.unsqueeze(-1)], dim=-1)
        z = hd.ln(hd.nlm(A.reshape(B * N, hd.D, hd.M))).reshape(B, N, hd.D)
        zz = spike(z)
        zf = zz.reshape(B * N, hd.D)
        s_out, no, do = hd.sync_out.step(zf, no, do)
        s_act, na, da = hd.sync_act.step(zf, na, da)
        q = hd.w_in(s_act).reshape(B, N, hd.d_head)
        att = torch.softmax(q @ K.transpose(1, 2) * hd.scale + mask, dim=-1)
        o = att @ V

        per_tick["att"].append(att)
        per_tick["spikes"].append(zz)
        per_tick["logits"].append(
            m.head(h + hd.w_out(s_out.reshape(B, N, -1))))
    return per_tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs-lm/sync.pt")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--out", default="tick-probe.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu")
    ca = ck["args"]
    tok = get_tokenizer()
    cfg = Config(vocab_size=tok.vocab_size, n_embd=ca["d"], n_layer=ca["layers"])
    model = SyncLM(cfg, use_sync=True, D=ca["D"], ticks=ca["ticks"],
                   pairs=ca["pairs"], spike_sync=bool(ca["spike_sync"]))
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    T = ca["ticks"]

    valid = load_split("valid")
    vb = batches_from(valid, a.tokens, 8, cfg.ctx, 7, dev)  # same 16k tokens

    loss_t = torch.zeros(T)
    tv = torch.zeros(T - 1)          # attention movement between ticks
    flip = torch.zeros(T - 1)        # spike-bit flips between ticks
    rate = torch.zeros(T)            # spike rate per tick
    ent = torch.zeros(T)             # attention entropy per tick
    ntok = 0
    for x, y in vb:
        pt = probe_forward(model, x)
        n = y.numel(); ntok += n
        for t in range(T):
            loss_t[t] += float(F.cross_entropy(
                pt["logits"][t].reshape(-1, cfg.vocab_size),
                y.reshape(-1))) * n
            att = pt["att"][t]
            ent[t] += float(-(att * att.clamp_min(1e-9).log()).sum(-1).mean()) * n
            rate[t] += float(pt["spikes"][t].mean()) * n
            if t:
                tv[t - 1] += float(
                    0.5 * (att - pt["att"][t - 1]).abs().sum(-1).mean()) * n
                flip[t - 1] += float(
                    (pt["spikes"][t] != pt["spikes"][t - 1]).float().mean()) * n

    loss_t, tv, flip = loss_t / ntok, tv / ntok, flip / ntok
    rate, ent = rate / ntok, ent / ntok
    ppl = [math.exp(v) for v in loss_t.tolist()]

    print(f"{a.ckpt} · T={T} ticks · {ntok} val tokens\n")
    print("  1. does the prediction improve across ticks?")
    print("     " + "  ".join(f"tick{t+1} {p:8.3f}" for t, p in enumerate(ppl)))
    print(f"     (final-tick ppl is the training readout; earlier ticks show "
          f"whether the loop refines)")
    print("\n  2. does the query move?  (total variation between consecutive "
          "ticks' attention, 0=frozen, 2=disjoint)")
    print("     " + "  ".join(f"t{t+1}->t{t+2} {v:.3f}" for t, v in enumerate(tv.tolist())))
    print("     attention entropy per tick: "
          + " ".join(f"{e:.2f}" for e in ent.tolist())
          + f"  (uniform would be {math.log(cfg.ctx):.2f})")
    print("\n  3. do the spikes change?  (fraction of bits flipping per tick)")
    print("     " + "  ".join(f"t{t+1}->t{t+2} {v:.3f}" for t, v in enumerate(flip.tolist())))
    print("     spike rate per tick: " + " ".join(f"{r:.3f}" for r in rate.tolist()))

    json.dump({"ppl_per_tick": ppl, "att_tv": tv.tolist(),
               "att_entropy": ent.tolist(), "spike_flip": flip.tolist(),
               "spike_rate": rate.tolist()}, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
