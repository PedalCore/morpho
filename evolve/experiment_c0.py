# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Experiment C0: automatic circuit choreography.

Given the 8-segment tentacle and a typed component library, automatically
construct a Morpho controller producing a desired actuator behaviour —
compositional synthesis instead of de-novo evolution. Targets are
generated from hand compositions in the same grammar (ground truth for
achievability and minimum size), but the synthesizer is never told them.

Search: deterministic level-wise typed enumeration with a beam — level k
holds programs of k components; expansions add one type-valid node; two
programs offering behaviourally identical node sets are collapsed
(behaviour caching). Score = cyclic-phase-aligned mismatch + size
tiebreak. No randomness anywhere.

Verification of every solution, in the programme's tradition:
  1. interpreter trace == compiled Morpho trace, bit-exact;
  2. exact cycle check — the closed system's full state cycle is found by
     hashing, and the output over the entire cycle must equal the target
     schedule; the verdict then holds for all time, not a sampled window.

Ladder after C0 (from the research plan): embodied targets -> joint
controller+morphology composition -> library learning (repeated useful
subgraphs promoted to new genes) -> recursive growth operators ->
frozen-program transfer across body sizes.

Usage (from repo root):
  python3 -m evolve.experiment_c0 selftest
  python3 -m evolve.experiment_c0 run
  python3 -m evolve.experiment_c0 show bounce
"""

import argparse
import itertools

import numpy as np

from .compose import (W, COMPONENTS, out_width, type_ok, interpret,
                      _init_states, _tick, state_key, morpho_trace, pretty)

WARM = 8
BEAM_FULL = 220
BEAM_PARTIAL = 120
MAX_SIZE = 5

HAND = {
    'travel': [('ring', (), ())],
    'inhale': [('johnson', (), ())],
    'twin':   [('ring', (), ()), ('rotate', (4,), (0,)),
               ('or_', (), (0, 1))],
    'gait':   [('toggle', (), ()), ('repeat8', (), (0,)),
               ('stripes', (), ()), ('xor_', (), (1, 2))],
    'bounce': [('ring', (), ()), ('reverse', (), (0,)),
               ('counter', (), ()), ('sel_ctr', (2,), (2,)),
               ('mux', (), (0, 1, 3))],
}
PERIOD = {'travel': 8, 'inhale': 16, 'twin': 8, 'gait': 2, 'bounce': 8}

def make_target(name):
    p = PERIOD[name]
    return interpret(HAND[name], WARM + p)[WARM:]


#@MARK: scoring

def score(prog, target):
    if out_width(prog, len(prog) - 1) != W:
        return 1.0
    p = len(target)
    tr = interpret(prog, WARM + 2 * p)[WARM:]
    best = 1.0
    for off in range(p):
        exp = target[(np.arange(2 * p) + off) % p]
        best = min(best, float((tr != exp).mean()))
        if best == 0.0:
            break
    return best


#@MARK: exact verification (finalists only)

def verify_exact(prog, target):
    """(1) compiled Morpho == interpreter, bit-exact; (2) the closed
    system's full state cycle reproduces the target schedule exactly."""
    p = len(target)
    steps = WARM + 4 * p
    itr = interpret(prog, steps)
    mtr = morpho_trace(prog, steps)
    if itr.shape != mtr.shape or (itr != mtr).any():
        return {'exact': False, 'reason': 'morpho/interpreter mismatch'}
    # find the state cycle by hashing full component state
    states = _init_states(prog)
    seen, outs_log = {}, []
    t = 0
    while True:
        k = state_key(states)
        if k in seen:
            t1, t2 = seen[k], t
            break
        seen[k] = t
        outs, states = _tick(prog, states)
        outs_log.append(outs[-1])
        t += 1
        if t > 4096:
            return {'exact': False, 'reason': 'state cycle too long'}
    cyc = np.stack(outs_log[t1:t2])
    if len(cyc) % p != 0:
        return {'exact': False, 'reason':
                f'cycle length {len(cyc)} not a multiple of target {p}'}
    for off in range(p):
        exp = target[(np.arange(len(cyc)) + off) % p]
        if (cyc == exp).all():
            return {'exact': True, 'transient': t1, 'cycle': len(cyc),
                    'phase': off}
    return {'exact': False, 'reason': 'cycle does not match target'}


#@MARK: deterministic typed beam enumeration

# C0's library is FROZEN to the original 18 autonomous components so its
# deterministic results stay reproducible after reactive extensions.
C0_LIBRARY = ('const0', 'const1', 'stripes', 'toggle', 'ring', 'johnson',
              'lfsr', 'counter', 'not_', 'xor_', 'or_', 'and_', 'mux',
              'rotate', 'reverse', 'delay', 'select', 'sel_ctr', 'repeat8')

def expansions(prog, library=C0_LIBRARY):
    n = len(prog)
    for comp in library:
        c = COMPONENTS[comp]
        arity = len(c['ins'])
        for ins in itertools.product(range(n), repeat=arity):
            if not type_ok(prog, comp, ins):
                continue
            for params in c['params']:
                yield prog + [(comp, params, ins)]

def signature(prog, p):
    traces = interpret(prog, WARM + p, all_nodes=True)
    return frozenset((t.shape[1], t.tobytes()) for t in traces)

def synthesize(name, max_size=MAX_SIZE, quiet=False):
    target = make_target(name)
    p = len(target)
    gens = [[(comp, params, ())]
            for comp in C0_LIBRARY if not COMPONENTS[comp]['ins']
            for params in COMPONENTS[comp]['params']]
    level = gens
    seen = set()
    evals = 0
    for size in range(1, max_size + 1):
        scored = []
        for prog in level:
            sig = signature(prog, p)
            if sig in seen:
                continue
            seen.add(sig)
            s = score(prog, target)
            evals += 1
            scored.append((s, prog))
            if s == 0.0:
                v = verify_exact(prog, target)
                if v['exact']:
                    if not quiet:
                        print(f"[{name}] solved at size {size} "
                              f"after {evals} evaluations "
                              f"(cycle {v['cycle']}, transient "
                              f"{v['transient']})")
                    return {'name': name, 'solved': True, 'size': size,
                            'evals': evals, 'prog': prog, 'verify': v}
        scored.sort(key=lambda t: (t[0], len(t[1]), pretty(t[1])))
        full = [pr for s, pr in scored if s < 1.0][:BEAM_FULL]
        partial = [pr for s, pr in scored if s >= 1.0][:BEAM_PARTIAL]
        kept = full + partial
        if not quiet:
            best = scored[0][0] if scored else 1.0
            print(f"[{name}] size {size}: {len(scored)} novel candidates, "
                  f"best err {best:.3f}, kept {len(kept)}")
        level = [q for prog in kept for q in expansions(prog)]
    return {'name': name, 'solved': False, 'evals': evals}


#@MARK: reporting

def waveform(trace, rows=None):
    return '\n'.join(''.join('█' if b else '·' for b in r)
                     for r in trace[:rows])

def run_all():
    results = []
    for name in HAND:
        results.append(synthesize(name))
        r = results[-1]
        if r['solved']:
            print(pretty(r['prog']))
            tr = interpret(r['prog'], WARM + 12)[WARM:]
            print(waveform(tr), '\n')
    print(f"{'target':<8} {'hand size':>10} {'found size':>11} "
          f"{'evals':>7} {'exact':>6}")
    for r in results:
        hand_n = len(HAND[r['name']])
        print(f"{r['name']:<8} {hand_n:>10} "
              f"{r.get('size', '-'):>11} {r['evals']:>7} "
              f"{str(r.get('verify', {}).get('exact', False)):>6}")

def show(name):
    print(f"target '{name}' (hand composition, size {len(HAND[name])}):")
    print(pretty(HAND[name]), '\n')
    print(waveform(make_target(name)))


def selftest():
    rng = np.random.default_rng(0)
    # 1. every component: interpreter == compiled Morpho, bit-exact,
    # exercised inside small typed programs
    probes = list(HAND.values()) + [
        [('lfsr', (), ()), ('not_', (), (0,)), ('delay', (2,), (1,))],
        [('counter', (), ()), ('sel_ctr', (1,), (0,)),
         ('repeat8', (), (1,)), ('ring', (), ()), ('and_', (), (2, 3,))],
        [('const1', (), ()), ('repeat8', (), (0,)), ('rotate', (2,), (1,)),
         ('reverse', (), (2,)), ('stripes', (), ()), ('xor_', (), (3, 4))],
        [('toggle', (), ()), ('const0', (), ()), ('or_', (), (0, 1)),
         ('delay', (1,), (2,))],
    ]
    used = set()
    for prog in probes:
        used.update(c for c, _, _ in prog)
        a, b = interpret(prog, 24), morpho_trace(prog, 24)
        assert a.shape == b.shape and (a == b).all(), pretty(prog)
    missing = set(C0_LIBRARY) - used - {'mux', 'select', 'xor_'}
    assert not missing, f'components not exercised: {missing}'
    print(f"1. interpreter == compiled Morpho, bit-exact, across "
          f"{len(probes)} typed programs covering the library")
    # mux + select coverage
    prog = [('ring', (), ()), ('reverse', (), (0,)), ('counter', (), ()),
            ('sel_ctr', (0,), (2,)), ('mux', (), (0, 1, 3)),
            ('select', (5,), (4,))]
    a, b = interpret(prog, 24), morpho_trace(prog, 24)
    assert (a == b).all()
    print("2. mux/select coverage program bit-exact")
    # 3. targets render and are periodic as declared
    for name in HAND:
        t = make_target(name)
        long = interpret(HAND[name], WARM + 3 * len(t))[WARM:]
        exp = t[(np.arange(len(long))) % len(t)]
        assert (long == exp).all(), name
    print("3. all five targets periodic with their declared periods")
    # 4. exact verifier accepts the hand programs, rejects a wrong one
    assert verify_exact(HAND['twin'], make_target('twin'))['exact']
    assert not verify_exact(HAND['travel'], make_target('inhale'))['exact']
    print("4. exact cycle verifier: accepts correct, rejects wrong")
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('selftest')
    sub.add_parser('run')
    s = sub.add_parser('show')
    s.add_argument('name', choices=list(HAND))
    args = ap.parse_args()
    if args.cmd == 'selftest':
        selftest()
    elif args.cmd == 'run':
        run_all()
    else:
        show(args.name)


if __name__ == '__main__':
    main()
