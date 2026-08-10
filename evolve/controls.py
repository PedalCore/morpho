# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Final controls for Experiment 0.5.

  census  Exact enumeration of all 2^15 ICs at N=15 through the compiled
          Morpho pipeline, for the best genome of each chirality: exact
          accuracy vs margin, exact attractor taxonomy, exact transients,
          and a reflection-bijection check between the two chiralities.

  frozen  The frozen-law transfer: stop evolving, instantiate the discovered
          architecture (transport sea + particle-absorber + hole-absorber at
          separation N/2) directly at N = 15..511, and measure
          P(correct | majority excess Delta, N) at horizons T = 2N, 4N, 8N.
          Large lattices run on a frame-capturing numpy stepper verified
          bit-exact against the compiled Morpho pipeline at N <= 63.

Usage: python3 -m evolve.controls census runs/exp05_density_n15.jsonl
       python3 -m evolve.controls frozen
"""

import argparse
import json
from collections import Counter

import numpy as np

from tiny_morpho import unpack, pack
from .evaluate import run_trace
from .experiment05 import _load

# The discovered law, per chirality: one-bit edits of the transport sea that
# absorb surplus particles / surplus holes.
LAW = {226: {'particle': 224, 'hole': 234},   # left-moving traffic
       184: {'particle': 168, 'hole': 248}}   # right-moving mirror


#@MARK: exact census

def _margin_table(correct, ones, n):
    print(f"{'ones':>5} {'cases':>6} {'exact strict':>13}")
    for k in range(n + 1):
        sel = ones == k
        print(f"{k:5d} {sel.sum():6d} {correct[sel].mean():13.6f}")

def _attractors(trace, n, failed):
    codes = np.stack([pack(trace[:, t]) for t in range(trace.shape[1])])
    taxonomy, periods, transients, cycles = Counter(), Counter(), [], set()
    for k in failed:
        seen = {}
        for t, c in enumerate(codes[:, k]):
            c = int(c)
            if c in seen:
                p = t - seen[c]
                transients.append(seen[c])
                periods[p] += 1
                cycle = frozenset(int(x) for x in codes[seen[c]:t, k])
                cycles.add(cycle)
                uniform = all(x in (0, (1 << n) - 1) for x in cycle)
                taxonomy['uniform_wrong' if uniform else f'nonuniform_p{p}'] += 1
                break
            seen[c] = t
        else:
            taxonomy['no_cycle_in_window'] += 1
    return taxonomy, periods, transients, cycles

def _census_one(genome, n=15):
    ics = unpack(np.arange(1 << n), n).astype(np.int32)
    trace = run_trace(np.asarray(genome, dtype=np.uint8), ics, 8 * n)
    ones = ics.sum(0)
    target = (ones > n // 2).astype(np.int32)
    correct = (trace[:, 2 * n - 1] == target).all(0)
    return ics, trace, ones, correct

def census(path, n=15):
    recs = [r for r in _load(path) if r['cell_n'] == n]
    for sea in (226, 184):
        chiral = [r for r in recs if r['sea_rule'] == sea]
        if not chiral:
            print(f"(no {sea}-sea genomes in {path})")
            continue
        rec = max(chiral, key=lambda r: r['holdout_smooth'])
        genome = rec['genome']
        ics, trace, ones, correct = _census_one(genome, n)
        failed = (~correct).nonzero()[0]
        print(f"\n== exact census: sea {sea} (seed {rec['seed']}), all "
              f"{1 << n} ICs, judged at T=2N ==\ngenome {genome}")
        print(f"exact strict accuracy: {correct.mean():.6f} "
              f"({correct.sum()}/{1 << n}; {len(failed)} failures)")
        _margin_table(correct, ones, n)
        taxonomy, periods, transients, cycles = _attractors(trace, n, failed)
        print(f"failure taxonomy: {dict(taxonomy.most_common())}")
        print(f"failure periods: {dict(sorted(periods.items()))}   "
              f"distinct failure attractor cycles: {len(cycles)}")
        if transients:
            print(f"failure transients: mean {np.mean(transients):.2f}  "
                  f"max {max(transients)}")

    # Reflection bijection: reflecting the lattice and swapping chirality
    # must give exactly identical statistics.
    rec = max((r for r in recs if r['sea_rule'] in LAW),
              key=lambda r: r['holdout_smooth'])
    g = np.asarray(rec['genome'], dtype=np.uint8)
    mirror_rule = {226: 184, 224: 168, 234: 248, 184: 226, 168: 224, 248: 234}
    mg = np.array([mirror_rule[int(r)] for r in g[::-1]], dtype=np.uint8)
    c1 = _census_one(g, n)[3]
    c2 = _census_one(mg, n)[3]
    print(f"\nreflection check: evolved {int(c1.sum())} correct, "
          f"mirrored genome {int(c2.sum())} correct "
          f"({'exact match' if c1.sum() == c2.sum() else 'MISMATCH'})")


#@MARK: frozen-law transfer

def _law_rules(sea, n):
    rules = np.full(n, sea, dtype=np.int16)
    rules[0] = LAW[sea]['particle']
    rules[n // 2] = LAW[sea]['hole']
    return rules

def _step_ca(rules, state, total, capture_at):
    """Iterate the non-uniform ring CA keeping only requested frames."""
    r2, s, caps = rules[:, None], state, {}
    for t in range(1, total + 1):
        idx = 4 * np.roll(s, 1, 0) + 2 * s + np.roll(s, -1, 0)
        s = (r2 >> idx) & 1
        if t in capture_at:
            caps[t] = s.copy()
    return caps

def _exact_ones_ics(rng, n, k, case_n):
    """case_n random ICs with exactly k ones (columns of an argsort trick)."""
    return (rng.random((n, case_n)).argsort(0) < k).astype(np.int16)

def _verify_stepper(sea, n=63, case_n=200, step_n=126):
    from .genome import genome_to_cell
    from tiny_morpho_seq import compile_seq
    rules = _law_rules(sea, n)
    ics = np.random.default_rng(1).integers(2, size=(n, case_n)).astype(np.int32)
    sim = compile_seq(genome_to_cell(rules))
    ref = sim.run(step_n + 1, state0=ics, samples=case_n)[:, -1]
    got = _step_ca(rules, ics.astype(np.int16), step_n, {step_n})[step_n]
    assert (got == ref).all(), "stepper diverged from compiled Morpho pipeline"

def _delta_grid(n):
    return [d for d in (1, 3, 5, 9, 15, 25, 41, 67, 109, 177, 289, 471)
            if d <= n]

def frozen(sizes=(15, 31, 63, 127, 255, 511), cases_per_side=1000, seed=99):
    results = {}
    for sea in (226, 184):
        _verify_stepper(sea)
        print(f"\n== frozen law, sea {sea} + {LAW[sea]['particle']} (particle "
              f"absorber @0) + {LAW[sea]['hole']} (hole absorber @N/2) ==")
        print("P(correct | Delta, N) at T=8N   [T=2N in brackets]")
        deltas = _delta_grid(sizes[-1])
        print(f"{'N':>5} " + ' '.join(f"{f'D={d}':>15}" for d in deltas))
        for n in sizes:
            rng = np.random.default_rng(seed + n)
            rules = _law_rules(sea, n)
            row = {}
            for d in _delta_grid(n):
                ks = [(n - d) // 2, (n + d) // 2]           # minority / majority of ones
                ics = np.concatenate(
                    [_exact_ones_ics(rng, n, k, cases_per_side) for k in ks], 1)
                target = (ics.sum(0) > n // 2).astype(np.int16)
                caps = _step_ca(rules, ics, 8 * n, {2 * n, 8 * n})
                row[d] = {c: float((caps[c * n] == target).all(0).mean())
                          for c in (2, 8)}
            results[f'{sea}_{n}'] = row
            print(f"{n:>5} " + ' '.join(
                f"{row[d][8]:.4f} [{row[d][2]:.3f}]" if d in row else ' ' * 15
                for d in deltas))
    json.dump(results, open('runs/frozen_law.json', 'w'))
    print("\nfull results -> runs/frozen_law.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('census').add_argument('path')
    sub.add_parser('frozen')
    args = p.parse_args()
    if args.cmd == 'census':
        census(args.path)
    else:
        frozen()


if __name__ == '__main__':
    main()
