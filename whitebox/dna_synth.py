"""M7 synthetic positional ladder (PAPER-DNA.md 3b — REQUIRED for the
verdict): tasks where counting provably cannot suffice, in the
~/.genomic_benchmarks layout so whitebox.dna_train runs unchanged.

Rungs (500 bp, balanced binary, 20k train / 3k test):
  synth_count    motif count >= 6 vs <= 3          (counter-POSITIVE control)
  synth_spacing  A..B gap 20+/-2 vs gap elsewhere  (position)
  synth_order    A before B vs B before A          (order)
  synth_assoc    XOR(variant of A, presence of B)  (distant association;
                                                    marginals uninformative)
Predictions (preregistered): counters solve count only; longhorn
solves spacing/order/assoc; cnn fails all four beyond stem reach.

python3 -m whitebox.dna_synth
"""

import pathlib

import numpy as np

OUT = pathlib.Path.home() / '.genomic_benchmarks'
L, NTR, NTE = 500, 20000, 3000
A, Av, B = 'ACGTACGA', 'ACGAACGA', 'TTGCCGTT'   # Av = A with one flip
BASES = np.array(list('ACGT'))


def bg(rng, n):
    return rng.integers(0, 4, size=n)


def plant(seq, motif, pos):
    seq[pos:pos + len(motif)] = [('ACGT').index(c) for c in motif]


def gen_count(rng, label):
    s = bg(rng, L)
    k = rng.integers(6, 9) if label else rng.integers(1, 4)
    for p in rng.choice(np.arange(0, L - 8, 12), size=k, replace=False):
        plant(s, A, p)
    return s


def gen_spacing(rng, label):
    s = bg(rng, L)
    if label:
        gap = rng.integers(18, 23)
    else:
        gap = rng.choice(np.concatenate(
            [np.arange(5, 16), np.arange(26, 100)]))
    p = rng.integers(0, L - 16 - gap)
    plant(s, A, p)
    plant(s, B, p + 8 + gap)
    return s


def gen_order(rng, label):
    s = bg(rng, L)
    gap = rng.integers(10, 80)
    p = rng.integers(0, L - 16 - gap)
    first, second = (A, B) if label else (B, A)
    plant(s, first, p)
    plant(s, second, p + 8 + gap)
    return s


def gen_assoc(rng, label):
    s = bg(rng, L)
    var = rng.integers(0, 2)                    # which A variant
    hasb = var ^ (1 - label)                    # label = XOR(var, hasb)
    p = rng.integers(0, L - 300)
    plant(s, Av if var else A, p)
    if hasb:
        plant(s, B, p + rng.integers(50, 200))
    return s


def main():
    rng = np.random.default_rng(7)
    for name, gen in (('synth_count', gen_count),
                      ('synth_spacing', gen_spacing),
                      ('synth_order', gen_order),
                      ('synth_assoc', gen_assoc)):
        for split, n in (('train', NTR), ('test', NTE)):
            for lab in (0, 1):
                d = OUT / name / split / str(lab)
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n // 2):
                    seq = gen(rng, lab)
                    (d / f'{i}.txt').write_text(''.join(BASES[seq]))
        print(name, 'done', flush=True)


if __name__ == '__main__':
    main()
