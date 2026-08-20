"""The locked M3 probe generator — deterministic from PROBES.md seeds.

Vocabulary (64): 0..15 keys, 16..31 values, 32..47 fillers, controls:
  48 CUE (reproduce), 49 QUERY, 50 MARK, 51 SEP, 52 PAD.
Any change after M3-control starts training is a protocol amendment.

python3 -m whitebox.probes   # prints samples + oracle sanity check
"""

import numpy as np

KEYS = np.arange(0, 16)
VALS = np.arange(16, 32)
FILL = np.arange(32, 48)
CUE, QUERY, MARK, SEP, PAD = 48, 49, 50, 51, 52
VOCAB = 64
CTX = 256
DELAY_GRID = (6, 11, 22, 44, 89)         # {.25,.5,1,2,4} x t_half (m=5)
TRAIN_DELAY = (4, 96)                    # sampled uniformly for training
DATA_SEED = 20260820


def _fill(rng, n):
    return rng.choice(FILL, n)


def make_copy(rng, delay, k=8):
    payload = rng.choice(VALS, k)
    seq = np.concatenate([payload, _fill(rng, delay), [CUE], payload])
    return seq, len(payload) + delay + 1, list(payload)   # answer span start


def make_assoc(rng, delay, pairs=4):
    ks = rng.choice(KEYS, pairs, replace=False)
    vs = rng.choice(VALS, pairs)
    kv = np.stack([ks, vs], 1).reshape(-1)
    qi = rng.integers(pairs)
    seq = np.concatenate([kv, _fill(rng, delay), [QUERY, ks[qi]], [vs[qi]]])
    return seq, len(kv) + delay + 2, [vs[qi]]


def make_induction(rng, delay):
    a, b = rng.choice(KEYS), rng.choice(VALS)
    pre = _fill(rng, rng.integers(2, 8))
    seq = np.concatenate([pre, [a, b], _fill(rng, delay), [a], [b]])
    return seq, len(pre) + 2 + delay + 1, [b]


def make_selective(rng, delay, distract=6):
    items = rng.choice(VALS, distract + 1, replace=False)
    mi = rng.integers(distract + 1)
    body = []
    for i, it in enumerate(items):
        if i == mi:
            body.append(MARK)
        body.append(it)
    seq = np.concatenate([body, _fill(rng, delay), [QUERY], [items[mi]]])
    return seq, len(body) + delay + 1, [items[mi]]


TASKS = {'copy': make_copy, 'assoc': make_assoc,
         'induction': make_induction, 'selective': make_selective}


def sample(rng, task=None, delay=None):
    """One example: (tokens, answer_start, answer). Loss/accuracy are
    evaluated ONLY on the answer span."""
    name = task or rng.choice(list(TASKS))
    d = delay if delay is not None else int(rng.integers(*TRAIN_DELAY))
    seq, start, ans = TASKS[name](rng, d)
    assert len(seq) <= CTX, (name, d, len(seq))
    return seq.astype(np.int64), start, ans, name, d


def train_stream(seed=DATA_SEED, batch=16):
    """Infinite deterministic training batches (padded to max len)."""
    rng = np.random.default_rng(seed)
    while True:
        ex = [sample(rng) for _ in range(batch)]
        L = max(len(e[0]) for e in ex)
        x = np.full((batch, L), PAD, dtype=np.int64)
        mask = np.zeros((batch, L), dtype=bool)
        for i, (seq, start, ans, _, _) in enumerate(ex):
            x[i, :len(seq)] = seq
            mask[i, start:start + len(ans)] = True
        yield x, mask


def eval_set(task, delay, n=200, seed=7):
    rng = np.random.default_rng(seed * 1000 + delay)
    return [sample(rng, task, delay) for _ in range(n)]


if __name__ == '__main__':
    rng = np.random.default_rng(DATA_SEED)
    for name in TASKS:
        seq, start, ans, _, d = sample(rng, name, 11)
        print(f'{name:10s} d=11 len={len(seq):3d} answer@{start}: '
              f'{ans} | tail: {seq[max(0, start - 4):start + len(ans)].tolist()}')
        assert list(seq[start:start + len(ans)]) == ans, 'oracle mismatch'
    # determinism check
    a = [s[0].sum() for s in eval_set('copy', 22, n=5)]
    b = [s[0].sum() for s in eval_set('copy', 22, n=5)]
    assert a == b, 'generator not deterministic'
    print('\noracle spans correct, generator deterministic — suite locked')
