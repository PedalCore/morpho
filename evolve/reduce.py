# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0. See repository LICENSE.

"""Exact post-hoc circuit reducer (handoff Phase 2).

CEGIS optimizes correctness alone, so its exact solutions are bloated
(sizes 10-16 vs minimal 2-4). This module shrinks them WITHOUT touching
the search objective: every proposed simplification must re-pass full
product-FSM verification before being accepted.

Moves: dead-node elimination (reachability from the output through node
inputs and fbk bindings) and node bypass (redirect all consumers of a
node — including fbk bindings — to another node of the same width, then
re-verify). Applied to a fixpoint. Reported sizes: found vs reduced."""

from .compose import COMPONENTS, out_width
from .experiment_sa0 import verify_ce


def _deps(prog, i):
    comp, params, ins = prog[i]
    d = list(ins)
    if comp == 'fbk' and params[0] >= 0:
        d.append(params[0])
    return d

def compact(prog):
    """Drop nodes unreachable from the output (last node); remap refs."""
    keep = set()
    stack = [len(prog) - 1]
    while stack:
        i = stack.pop()
        if i in keep:
            continue
        keep.add(i)
        stack.extend(_deps(prog, i))
    order = [i for i in range(len(prog)) if i in keep]
    remap = {old: new for new, old in enumerate(order)}
    out = []
    for old in order:
        comp, params, ins = prog[old]
        if comp == 'fbk' and params[0] >= 0:
            params = (remap[params[0]],)
        out.append((comp, params, tuple(remap[i] for i in ins)))
    return out

def _redirect(prog, victim, target):
    """All consumers of `victim` (inputs and fbk bindings) read `target`."""
    out = []
    for comp, params, ins in prog:
        if comp == 'fbk' and params[0] == victim:
            params = (target,)
        out.append((comp, params,
                    tuple(target if i == victim else i for i in ins)))
    return out

def reduce_prog(prog, ref, max_passes=6):
    """Verified-exact minimization to a fixpoint of the two moves."""
    cur = compact(prog)
    for _ in range(max_passes):
        improved = False
        for victim in range(len(cur) - 1):
            w = out_width(cur, victim)
            for target in range(len(cur) - 1):
                if target == victim or out_width(cur, target) != w:
                    continue
                cand = compact(_redirect(cur, victim, target))
                if len(cand) >= len(cur):
                    continue
                try:
                    if verify_ce(cand, ref).get('exact'):
                        cur, improved = cand, True
                        break
                except Exception:
                    continue
            if improved:
                break
        if not improved:
            return cur
    return cur
