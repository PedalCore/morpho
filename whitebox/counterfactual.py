"""Locked counterfactual binding suite for TRAINED LM checkpoints (M4).

Answers, without retraining: does the 14M model already have PARTIAL
binding? Paired natural-language prompts differing ONLY in which color
belongs to which name:

  A: " Tom has the red ball. Tim has the blue ball. [filler]
      Later, Tom picks up the"          -> ' red'
  B: same with red/blue exchanged       -> ' blue'

Metrics per cell (collaborator spec, preregistered):
  acc  — candidate-restricted top-1 over the stored colors, both twins
  S    — swap sensitivity: mean of (m_A + m_B)/2 where
         m_A = logit(correct_A) - logit(swapped alternative | A)
         positive S = predictions track the assignment, not the tokens
  rec  — recency-follow rate: how often argmax = most recently stored
         color (the masquerading heuristic the controls must expose)

Outcome map (recorded in advance): strong binding = high acc, S >> 0,
survives facts/distance; partial = above chance on easy cells, S > 0,
declining with facts/delay; latent = chance top-1 but S slightly > 0;
none = S ~ 0, flat; systematic misbinding (recency rule) = S < 0.

Counterbalanced: queried entity first-vs-last stored; names, colors,
assignment order randomized per example; single-token candidates only.
Data seed locked: 20260823. Distance via neutral filler sentences (no
names/colors). Facts {2,4,8} x distance {short ~0, long ~120 tok} x
query position {first, last}. n=50 pairs per cell.

python3 -m whitebox.counterfactual runs/plateau-dict-ext20k/ckpt.pt ...
"""

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
SPIKELM = '/Users/marycarrigan/coding/morpho-snn/spikelm'
sys.path.insert(0, SPIKELM)

from spikelm.data import get_tokenizer                       # noqa: E402
from whitebox.model import Config, CausalCRATE, CausalCRATEM2  # noqa: E402

NAMES = ['Tom', 'Tim', 'Lily', 'Ben', 'Sam', 'Anna', 'Max', 'Sue',
         'Mia', 'Jack', 'Amy', 'Joe', 'Lucy', 'Sara', 'Emma', 'Bob']
COLORS = ['red', 'blue', 'green', 'black', 'white', 'brown', 'pink',
          'gray', 'yellow', 'purple']
FILLER = [' The sun was warm.', ' It was a nice day.',
          ' The birds sang songs.', ' The wind blew softly.',
          ' The trees were tall.', ' The sky was clear.']
SEED = 20260823
CELLS = [(f, d, q) for f in (2, 4, 8) for d in ('short', 'long')
         for q in ('first', 'last')]
N_PAIRS = 50


def build_cell(rng, tok, facts, dist, qpos):
    def ids(s):
        e = tok.encode(s)
        return e if isinstance(e, list) else e.ids
    pairs = []
    for _ in range(N_PAIRS):
        ns = rng.choice(len(NAMES), facts, replace=False)
        cs = rng.choice(len(COLORS), facts, replace=False)
        qi = 0 if qpos == 'first' else facts - 1
        oj = int(rng.choice([j for j in range(facts) if j != qi]))
        filler = ''
        if dist == 'long':
            while len(ids(filler)) < 110:
                filler += FILLER[int(rng.integers(len(FILLER)))]
        twins = []
        for swap in (False, True):
            assign = list(cs)
            if swap:
                assign[qi], assign[oj] = assign[oj], assign[qi]
            body = ''.join(f' {NAMES[ns[k]]} has the '
                           f'{COLORS[assign[k]]} ball.'
                           for k in range(facts))
            prompt = body + filler + f' Later, {NAMES[ns[qi]]} picks up the'
            correct = ids(' ' + COLORS[assign[qi]])[0]
            alt = ids(' ' + COLORS[assign[oj]])[0]
            cand = [ids(' ' + COLORS[c])[0] for c in cs]
            # most recent stored color in THIS twin
            recent = ids(' ' + COLORS[assign[facts - 1]])[0]
            twins.append(dict(ids=ids(prompt), correct=correct, alt=alt,
                              cand=cand, recent=recent))
        pairs.append(twins)
    return pairs


@torch.no_grad()
def last_logits(model, batch_ids, device):
    L = max(len(x) for x in batch_ids)
    x = torch.zeros(len(batch_ids), L, dtype=torch.long)
    for i, s in enumerate(batch_ids):
        x[i, :len(s)] = torch.tensor(s)
    logits, _ = model(x.to(device))
    return torch.stack([logits[i, len(s) - 1]
                        for i, s in enumerate(batch_ids)]).float().cpu()


def eval_checkpoint(path, tok, device='cpu'):
    ck = torch.load(path, map_location='cpu')
    cfg = Config(**ck['cfg'])
    cls = CausalCRATEM2 if cfg.m2 else CausalCRATE
    model = cls(cfg)
    model.load_state_dict(ck['model'])
    model.to(device).eval()
    rng = np.random.default_rng(SEED)
    out = {}
    for facts, dist, qpos in CELLS:
        pairs = build_cell(rng, tok, facts, dist, qpos)
        flat = [t for p in pairs for t in p]
        logits = []
        for i in range(0, len(flat), 32):
            logits.append(last_logits(model, [t['ids'] for t in flat[i:i+32]],
                                      device))
        logits = torch.cat(logits)
        acc = rec = 0.0
        S = []
        for pi, p in enumerate(pairs):
            ms = []
            for ti, t in enumerate(p):
                lg = logits[2 * pi + ti]
                cl = lg[t['cand']]
                top = t['cand'][int(cl.argmax())]
                acc += float(top == t['correct'])
                rec += float(top == t['recent'])
                ms.append(float(lg[t['correct']] - lg[t['alt']]))
            S.append(0.5 * (ms[0] + ms[1]))
        n = 2 * len(pairs)
        out[f'f{facts}-{dist}-{qpos}'] = dict(
            acc=round(acc / n, 3), S=round(float(np.mean(S)), 3),
            rec=round(rec / n, 3), chance=round(1.0 / facts, 3))
    return out


def main():
    tok = get_tokenizer()
    results = {}
    for path in sys.argv[1:]:
        name = pathlib.Path(path).parent.name
        print(f'== {name}')
        r = eval_checkpoint(path, tok)
        results[name] = r
        for cell, v in r.items():
            print(f'  {cell:16s} acc {v["acc"]:.3f} (chance {v["chance"]})'
                  f'  S {v["S"]:+.3f}  recency {v["rec"]:.3f}')
    outp = pathlib.Path(__file__).parent / 'runs' / 'counterfactual.json'
    with open(outp, 'w') as f:
        json.dump(results, f, indent=1)
    print('saved', outp)


if __name__ == '__main__':
    main()
