# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Exact equivalence checking of evolved circuits against reference FSMs.

Both the compiled circuit and the task reference are finite-state machines,
so correctness over every possible infinite input sequence is decidable.
Verification is a two-phase BFS of the product machine (circuit registers x
reference state) under both input values:

  phase 1  expand `warmup` layers without output checks (targets are
           undefined for t < warmup);
  phase 2  checked closure from the depth-warmup frontier — every state
           expanded here is reachable at some t >= warmup, and all its
           outgoing transitions must agree with the reference.

Verdicts: exact=True (every reachable behaviour matches), a concrete
counterexample transition, or aborted=True past the state cap.
"""

import numpy as np


def _expand(sim, ref, layer):
    """All transitions out of `layer`: [(child_key, y, input, parent_ref)].
    Input ints enumerate all 2^x_n input-wire combinations (bit i = wire i)."""
    x_n = len(sim.c.inputs[0])
    s_batch = np.stack([np.frombuffer(s, dtype=np.int32) for s, _ in layer],
                       axis=1)
    refs = [r for _, r in layer]
    transitions = []
    for b in range(1 << x_n):
        bits = [(b >> i) & 1 for i in range(x_n)]
        x = np.tile(np.array(bits, dtype=np.int32)[:, None, None],
                    (1, 1, len(layer)))
        y, new_s = sim.run(1, x, state0=s_batch, samples=len(layer),
                           return_state=True)
        y = np.asarray(y).reshape(-1)
        for k, r in enumerate(refs):
            transitions.append(((new_s[:, k].tobytes(), ref['step'](r, b)),
                                int(y[k]), b, r))
    return transitions

def verify(sim, ref, warmup, max_states=1 << 20):
    layer = {(np.zeros(len(sim.reg_idx), dtype=np.int32).tobytes(), 0)}
    for _ in range(warmup):
        layer = {child for child, _, _, _ in _expand(sim, ref, sorted(layer))}
        if len(layer) > max_states:
            return {'exact': False, 'aborted': True, 'states': len(layer)}

    seen, frontier = set(layer), sorted(layer)
    while frontier:
        nxt = []
        for child, y, b, r in _expand(sim, ref, frontier):
            if y != ref['out'](r, b):
                return {'exact': False, 'aborted': False, 'states': len(seen),
                        'counterexample': {'input': b, 'ref_state': r}}
            if child not in seen:
                seen.add(child)
                nxt.append(child)
        if len(seen) > max_states:
            return {'exact': False, 'aborted': True, 'states': len(seen)}
        frontier = sorted(nxt)
    return {'exact': True, 'aborted': False, 'states': len(seen)}
