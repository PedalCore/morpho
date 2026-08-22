"""Recurrent (constant-time-per-token) inference for CRSA models.

Training uses the chunked parallel scan (model.py CRSA.forward); this
module runs the SAME model token-by-token, carrying only the per-layer
counter state c (B, K, p) — the recurrence c_t = rho * c_{t-1} + h_t^2
that the scan computes in parallel. Everything else in the block stack
(LN, dictionary / MLP, head) is per-token already.

Correctness contract (the campaign's rule — equivalence, not vibes):
`verify()` runs the same random token sequence through the parallel
forward and the recurrent stepper and reports the max logit difference.
Expected: ~1e-5 float32 roundoff (the chunked scan itself carries
<=6.5e-7 vs a serial scan; LN/matmul reassociation adds the rest).

Scope: attn='crsa' (and 'tost', same recurrence with rho=uniform mean —
handled via its rho buffer) with dict_local or mlp blocks, m2_identity.
Position embeddings exist only up to cfg.ctx: beyond that the position
index CLAMPS to ctx-1 (the counters keep integrating — state is
unbounded-context; the positional table is not). Documented divergence
from the sliding-window naive sampler beyond ctx tokens.
"""

import sys
import torch
import torch.nn.functional as F

from whitebox.model import Config, CausalCRATEM2, CRSA, BlockODLocal, BlockMLP


class RecurrentCRSA:
    """Step-wise executor over a trained CausalCRATEM2 (CRSA attention)."""

    def __init__(self, model, batch=1):
        cfg = model.cfg
        assert cfg.attn in ('crsa', 'tssa'), 'recurrent path: CRSA only'
        assert cfg.m2_identity and cfg.m2 in ('a', 'b')
        assert cfg.dict_local or cfg.mlp, 'dict_local or mlp blocks'
        self.model = model
        self.cfg = cfg
        self.B = batch
        self.reset()

    def reset(self):
        m, cfg = self.model, self.cfg
        dev = m.emb.weight.device
        K, p = cfg.n_head, cfg.n_embd // cfg.n_head
        self.c = [torch.zeros(self.B, K, p, device=dev)
                  for _ in m.blocks]
        self.t = 0

    @torch.no_grad()
    def step(self, tok):
        """tok: (B,) long. Returns logits (B, vocab). Advances state."""
        m, cfg = self.model, self.cfg
        pos = min(self.t, cfg.ctx - 1)
        z = m.ln_in(m.emb(tok) + m.pos.weight[pos])
        for li, b in enumerate(m.blocks):
            a = b.attn
            h = a.U(z).view(self.B, a.K, a.p)
            self.c[li] = a.rho.view(1, a.K, 1) * self.c[li] + h * h
            dcoef = 1.0 / (1.0 + self.c[li])
            agg = (dcoef * h).reshape(self.B, cfg.n_embd)
            attn_out = -a.scale * (F.linear(agg, a.U.weight.t())
                                   if a.tied else a.out(agg))
            if isinstance(b, BlockODLocal):
                x = b.ln(z + attn_out)
                pre = b.eta * (x @ b.D)
                code = pre if b.identity else torch.relu(pre - b.eta * b.lam)
                xhat = code @ b.D.t()
                z = x + b.gamma * (xhat - x)
            else:                                      # BlockMLP
                x = z + attn_out
                z = x + b.w2(F.gelu(b.w1(b.ln(x))))
        self.t += 1
        return m.head(z)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=100, temperature=0.8, seed=None):
        """prompt_ids: (T,) long. Returns full id list."""
        g = torch.Generator(device='cpu')
        if seed is not None:
            g.manual_seed(seed)
        self.reset()
        ids = list(prompt_ids.tolist())
        logits = None
        for t in prompt_ids:
            logits = self.step(t.view(self.B))
        for _ in range(max_new):
            probs = torch.softmax(logits[0] / temperature, -1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=g).item()
            ids.append(nxt)
            logits = self.step(torch.tensor([nxt] * self.B,
                                            device=logits.device))
        return ids


@torch.no_grad()
def verify(model, T=256, seed=0):
    """Max |logit| gap: parallel forward vs recurrent stepper, T tokens."""
    torch.manual_seed(seed)
    idx = torch.randint(0, model.cfg.vocab_size, (1, T),
                        device=model.emb.weight.device)
    par, _ = model(idx)
    rec = RecurrentCRSA(model)
    outs = [rec.step(idx[:, t]) for t in range(T)]
    return float((par[0] - torch.cat(outs)).abs().max())


if __name__ == '__main__':
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--prompt', default='Once upon a time, there was a '
                                        'little girl named')
    ap.add_argument('--max-new', type=int, default=100)
    ap.add_argument('--bench', action='store_true')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    model = CausalCRATEM2(Config(**ck['cfg']))
    model.load_state_dict(ck['model'])
    model.eval()

    err = verify(model)
    print(f'parallel-vs-recurrent max logit diff over 256 tokens: {err:.2e}')

    sys.path.insert(0, '/Users/marycarrigan/coding/morpho-snn/spikelm')
    from spikelm.data import get_tokenizer
    tok = get_tokenizer()
    enc = tok.encode(args.prompt)
    ids = torch.tensor(enc if isinstance(enc, list) else enc.ids)

    rec = RecurrentCRSA(model)
    t0 = time.time()
    out = rec.generate(ids, max_new=args.max_new, seed=1000)
    dt = time.time() - t0
    print(f'recurrent sampling: {args.max_new} tokens in {dt:.2f}s '
          f'= {args.max_new / dt:.1f} tok/s (CPU)')
    print('sample:', tok.decode(out)[:400])

    if args.bench:
        rec.reset()
        for t in ids:                                  # warm state + code
            rec.step(t.view(1))
        t0 = time.time()
        cur = torch.tensor([ids[-1]])
        for _ in range(200):
            logits = rec.step(cur)
            cur = logits[0].argmax().view(1)
        dt = time.time() - t0
        print(f'pure step rate (greedy, no sampling overhead): '
              f'{200 / dt:.1f} tok/s')
