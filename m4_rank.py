"""Rank-compression of the query specialization (M4, no training).

At the successful q-kv checkpoint (Q separate, K=V tied — binding
~1.00), form per layer Delta = W_q - W_kv, SVD-truncate to rank r, and
evaluate with W_q^(r) = W_kv + Delta_r. Answers: is query
specialization intrinsically full-rank, or a small preconditioner that
translates retrieval contexts into the tied memory basis?

White-box interpretation if low rank suffices: the model keeps ONE
derived memory basis and learns a cheap query-side correction.
Cost at d=448, L=12: rank 8 = 86k, 16 = 172k, 32 = 344k params
vs full 2.41M.

python3 -m whitebox.m4_rank whitebox/runs/m4probes/q-kv-s0.pt
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from whitebox.model import Config, CausalCRATEM2       # noqa: E402
from whitebox.m4_probe_train import eval_binding       # noqa: E402
from whitebox.probe_train import eval_grid             # noqa: E402

RANKS = (0, 1, 2, 4, 8, 16, 32, 64, 128)   # 0 = W_q := W_kv (no untying)


def main():
    path = sys.argv[1]
    ck = torch.load(path, map_location='cpu')
    cfg = Config(**ck['cfg'])
    device = 'cpu'
    model = CausalCRATEM2(cfg)
    model.load_state_dict(ck['model'])
    model.to(device).eval()

    # stash full-rank deltas
    deltas = []
    for b in model.blocks:
        a = b.attn
        deltas.append((a, a.q.weight.data.clone(), a.k.weight.data.clone()))

    results = {}
    for r in RANKS + ('full',):
        for a, wq, wk in deltas:
            if r == 'full':
                a.q.weight.data.copy_(wq)
            elif r == 0:
                a.q.weight.data.copy_(wk)
            else:
                d = wq - wk
                U, S, Vh = torch.linalg.svd(d)
                dr = (U[:, :r] * S[:r]) @ Vh[:r]
                a.q.weight.data.copy_(wk + dr)
        b = eval_binding(model, device)
        m3 = eval_grid(model, device)
        ind = float(np.mean([v for k, v in m3.items()
                             if k.startswith('induction')]))
        sel = float(np.mean([v for k, v in m3.items()
                             if k.startswith('selective')]))
        bm = float(np.mean(list(b.values())))
        results[str(r)] = dict(bind_mean=round(bm, 3),
                               induction=round(ind, 3),
                               selective=round(sel, 3),
                               facts=[b[f'bind-facts@{f}']
                                      for f in (2, 4, 8, 16)])
        print(f'rank {str(r):>4}: bind-mean {bm:.3f}  '
              f'facts {results[str(r)]["facts"]}  '
              f'induction {ind:.3f}  selective {sel:.3f}', flush=True)

    out = pathlib.Path(__file__).parent / 'runs' / 'm4rank.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
