"""M4 binding probes — preregistered BEFORE any repair trains (M4.md).

The locked M3 suite (probes.py — untouched, its own file forbids
amendment) already localized the deficit: copy flat, induction 1.00,
selective 0.56 vs KV 0.91, assoc weaker. The sharp new probe is
BINDING SWAPS: matched example pairs containing exactly the same
identities and attributes where only the PAIRING differs. A memory that
retains marginal statistics sees nearly the same evidence in both; a
binding-capable memory must distinguish them.

Axes are separated deliberately (M4.md table):
  delay      — NULL fillers only (pure elapsed time)      -> forgetting
  facts      — 2..16 stored pairs, gap fixed              -> collision
  distractors— random FILL activity, TOTAL GAP FIXED at 64
               (n FILL + (64-n) NULL)                     -> selective
                                                             retrieval
  in/out W   — gap 8 vs 48 around the cache oracle's 16   -> local cache
                                                             vs persistent
Names and attributes are randomized every example; assignments are
random permutations, so fixed name->attr relationships cannot be
memorized.

Vocabulary extends the locked map compatibly: 0..15 names, 16..31
attributes, 32..47 FILL, 48 CUE, 49 QUERY, 50 MARK, 51 SEP, 52 PAD,
53 NULL (quiet filler — separates elapsed time from interfering
activity).
"""

import numpy as np

from whitebox.probes import (KEYS, VALS, FILL, QUERY, PAD, VOCAB,  # noqa
                             TASKS as M3_TASKS, DATA_SEED)

NULL = 53
OF = 54                 # possessive marker: [attr, OF, name] patterns
CTX = 128
GAP_FIXED = 64          # distractor axis: total gap constant
DELAY_GRID = (8, 16, 32, 64, 96)
FACTS_GRID = (2, 4, 8, 16)          # 16 = full name vocabulary
DISTRACT_GRID = (0, 8, 24, 64)
WINDOW_GRID = (8, 48)               # inside / outside W=16
TRAIN_SEED = 20260822


def make_binding(rng, facts=2, gap=16, n_fill=0, perm=None, qi=None):
    """STORE k1 v1 ... kF vF | gap (n_fill FILL + rest NULL, shuffled) |
    QUERY ki -> vi.  Returns (seq, answer_start, [answer])."""
    ks = rng.choice(KEYS, facts, replace=False)
    vs = rng.choice(VALS, facts, replace=False)
    if perm is None:
        perm = rng.permutation(facts)
    kv = np.stack([ks, vs[perm]], 1).reshape(-1)
    filler = np.concatenate([rng.choice(FILL, n_fill),
                             np.full(gap - n_fill, NULL)])
    rng.shuffle(filler)
    if qi is None:
        qi = int(rng.integers(facts))
    seq = np.concatenate([kv, filler, [QUERY, ks[qi]], [vs[perm[qi]]]])
    assert len(seq) <= CTX
    return seq.astype(np.int64), len(kv) + gap + 2, [int(vs[perm[qi]])]


def swap_pair(rng, facts=2, gap=16, n_fill=0):
    """The binding-swap twin pair: identical names, attributes, gap and
    distractor positions; only the assignment permutation differs (and
    the query key is shared). Marginal statistics cannot separate them."""
    state = rng.bit_generator.state
    p1 = rng.permutation(facts)
    p2 = p1.copy()
    i, j = rng.choice(facts, 2, replace=False)
    p2[i], p2[j] = p2[j], p2[i]
    out = []
    for perm in (p1, p2):
        rng.bit_generator.state = state          # same names/attrs/filler
        # query a SWAPPED position: the twins' answers must differ
        ex = make_binding(rng, facts, gap, n_fill, perm=perm, qi=int(i))
        out.append(ex)
    rng.permutation(2 * facts + gap)             # advance the stream
    return out


def eval_sets(n=200, seed=7):
    """The preregistered grid: dict name -> list of examples."""
    out = {}
    def pairs(tag, **kw):
        rng = np.random.default_rng(seed + hash(tag) % 1000)
        ex = []
        for _ in range(n // 2):
            ex.extend(swap_pair(rng, **kw))
        out[tag] = ex
    for d in DELAY_GRID:
        pairs(f'bind-delay@{d}', facts=2, gap=d, n_fill=0)
    for f in FACTS_GRID:
        pairs(f'bind-facts@{f}', facts=f, gap=16, n_fill=0)
    for nd in DISTRACT_GRID:
        pairs(f'bind-distract@{nd}', facts=4, gap=GAP_FIXED, n_fill=nd)
    for g in WINDOW_GRID:
        pairs(f'bind-window@{g}', facts=4, gap=g, n_fill=g // 2)
    # round-2 amendment: STRICTLY in-window cells (earliest store within
    # 16 of the query) — the mixed grid's cells almost all exceed W=16
    pairs('bind-inwin@2', facts=2, gap=6, n_fill=0)    # max distance 10
    pairs('bind-inwin@4', facts=4, gap=4, n_fill=0)    # max distance 12
    return out


def make_update(rng, facts=4, gap=8, rebinds=1):
    """M5 stress: OVERWRITE fidelity. Store facts, then re-bind
    `rebinds` of the keys to NEW values after a gap; query a re-bound
    key — correct answer is the NEW value (the old one must be
    overwritten, not superposed)."""
    ks = rng.choice(KEYS, facts, replace=False)
    vs1 = rng.choice(VALS, facts, replace=False)
    body = list(np.stack([ks, vs1], 1).reshape(-1))
    mid = list(np.full(gap, NULL))
    rebinds = min(rebinds, facts)
    ri = rng.choice(facts, rebinds, replace=False)
    remaining = [v for v in VALS if v not in vs1]
    vs2 = rng.choice(remaining, rebinds, replace=False)
    body2 = list(np.stack([ks[ri], vs2], 1).reshape(-1))
    qi = int(rng.choice(ri))
    new_v = int(vs2[list(ri).index(qi)])
    seq = np.concatenate([body, mid, body2, np.full(gap, NULL),
                          [QUERY, ks[qi]], [new_v]]).astype(np.int64)
    assert len(seq) <= CTX
    return seq, len(seq) - 1, [new_v]


def eval_sets_stress(n=200, seed=13):
    """M5 capacity/fidelity stress grid (preregistered before the
    diagonal ran on it): fact counts approaching the per-head state
    dimension, plus overwrite cells."""
    out = {}
    def cells(tag, gen, **kw):
        rng = np.random.default_rng(seed + hash(tag) % 1000)
        out[tag] = [gen(rng, **kw) for _ in range(n)]
    for f in (8, 12, 16):
        cells(f'stress-facts@{f}', make_binding, facts=f, gap=8, n_fill=0)
    for rb in (1, 2, 4):
        cells(f'stress-update@{rb}', make_update, facts=4, gap=8, rebinds=rb)
    cells('stress-update-far', make_update, facts=4, gap=40, rebinds=2)
    return out





def make_binding_pat(rng, facts=2, gap=16, pattern='pre', offset=2,
                     perm=None, qi=None):
    """Offset-varied binding (gate 3): the owner is NOT always the
    previous token. Patterns per stored pair:
      pre : [name, attr]                (owner at t-1 — the oracle's case)
      post: [attr, OF, name]            (owner AFTER the attribute)
      far : [name, f1..f_offset, attr]  (owner offset+1 back)
    Query unchanged: QUERY name -> attr. The prev-token oracle fails
    post and far BY CONSTRUCTION."""
    ks = rng.choice(KEYS, facts, replace=False)
    vs = rng.choice(VALS, facts, replace=False)
    if perm is None:
        perm = rng.permutation(facts)
    body = []
    for i in range(facts):
        k, v = int(ks[i]), int(vs[perm[i]])
        if pattern == 'pre':
            body += [k, v]
        elif pattern == 'post':
            body += [v, OF, k]
        else:                                    # far
            body += [k] + list(rng.choice(FILL, offset)) + [v]
    filler = np.full(gap, NULL)
    if qi is None:
        qi = int(rng.integers(facts))
    seq = np.concatenate([body, filler, [QUERY, ks[qi]],
                          [vs[perm[qi]]]]).astype(np.int64)
    assert len(seq) <= CTX, (pattern, facts, offset, len(seq))
    return seq, len(body) + gap + 2, [int(vs[perm[qi]])]


def swap_pair_pat(rng, facts=2, gap=16, pattern='pre', offset=2):
    state = rng.bit_generator.state
    p1 = rng.permutation(facts)
    p2 = p1.copy()
    i, j = rng.choice(facts, 2, replace=False)
    p2[i], p2[j] = p2[j], p2[i]
    out = []
    for perm in (p1, p2):
        rng.bit_generator.state = state
        out.append(make_binding_pat(rng, facts, gap, pattern, offset,
                                    perm=perm, qi=int(i)))
    rng.permutation(2 * facts + gap)
    return out


def eval_sets_pat(n=200, seed=11):
    """Gate-3 grid: preregistered offset-varied cells."""
    out = {}
    def cells(tag, **kw):
        rng = np.random.default_rng(seed + hash(tag) % 1000)
        ex = []
        for _ in range(n // 2):
            ex.extend(swap_pair_pat(rng, **kw))
        out[tag] = ex
    for f in (2, 4):
        cells(f'pat-pre@{f}', facts=f, gap=16, pattern='pre')
        cells(f'pat-post@{f}', facts=f, gap=16, pattern='post')
        cells(f'pat-far2@{f}', facts=f, gap=16, pattern='far', offset=2)
    cells('pat-far4@2', facts=2, gap=16, pattern='far', offset=4)
    return out


def train_stream(seed=TRAIN_SEED, batch=16, reachable=0, patterns=False,
                 stress=False):
    """Mixed stream: the four locked M3 tasks + binding (equal weight).
    Binding params sampled across the sweep ranges so no eval cell is
    out-of-distribution. NOTE: this is a DIFFERENT training distribution
    from the locked M3 runs — M4 comparisons are between arms trained
    identically under THIS protocol, plus the absolute induction gate."""
    rng = np.random.default_rng(seed)
    names = list(M3_TASKS) + ['binding']
    while True:
        ex = []
        for _ in range(batch):
            t = names[int(rng.integers(len(names)))]
            if t == 'binding' and stress and rng.random() < 0.5:
                ex.append(make_update(rng, facts=int(rng.choice([2, 4])),
                                      gap=int(rng.integers(4, 41)),
                                      rebinds=int(rng.integers(1, 4))))
            elif t == 'binding' and patterns:
                pat = ['pre', 'post', 'far'][int(rng.integers(3))]
                facts = int(rng.choice([2, 4]))
                gap = int(rng.integers(4, 49))
                off = int(rng.integers(1, 5))
                ex.append(make_binding_pat(rng, facts, gap, pat, off))
            elif t == 'binding':
                if reachable:
                    # reachability-aware curriculum (round-2 amendment):
                    # a W-token branch cannot solve store->query gaps
                    # beyond its window; training on impossible examples
                    # adds irreducible conflicting gradients. Constrain
                    # earliest-store distance 2F+gap <= W-1.
                    facts = int(rng.choice([2, 4]))
                    gmax = reachable - 1 - 2 * facts
                    gap = int(rng.integers(1, max(2, gmax + 1)))
                else:
                    facts = int(rng.choice(FACTS_GRID))
                    gmax = min(96, CTX - 2 * facts - 3)
                    gap = int(rng.integers(4, gmax + 1))
                n_fill = int(rng.integers(0, min(gap, 64) + 1))
                ex.append(make_binding(rng, facts, gap, n_fill))
            else:
                d = int(rng.integers(4, 90))
                seq, start, ans = M3_TASKS[t](rng, d)
                seq = seq[:CTX]
                ex.append((seq.astype(np.int64), start, ans))
        L = max(len(e[0]) for e in ex)
        x = np.full((batch, L), PAD, dtype=np.int64)
        mask = np.zeros((batch, L), dtype=bool)
        for i, (seq, start, ans) in enumerate(ex):
            x[i, :len(seq)] = seq
            mask[i, start:start + len(ans)] = True
        yield x, mask


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    (s1, st1, a1), (s2, st2, a2) = swap_pair(rng, facts=4, gap=8, n_fill=4)
    print('twin 1:', s1.tolist(), '->', a1)
    print('twin 2:', s2.tolist(), '->', a2)
    assert (s1[:8] != s2[:8]).any() and a1 != a2
    same_gap = (s1[8:16] == s2[8:16]).all()
    print('gap identical across twins:', bool(same_gap))
    ev = eval_sets(n=20)
    print('eval cells:', list(ev)[:6], f'... ({len(ev)} total)')
