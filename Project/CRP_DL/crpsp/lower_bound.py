"""A* lower bound h(n) via 2-cycle counting (paper Eq 27-30, after Tanaka & Voss 2019)."""
from __future__ import annotations

from typing import Sequence

from .instance import Instance


def must_precede_pairs(inst: Instance) -> frozenset[tuple[int, int]]:
    """(a, b) in result <=> a must be placed on the vessel before b.

    CRPSP: same-vessel-stack below->above, plus Eq-23 shoreline precedence
    (tier-k of a farther stack before tier-k of any nearer stack; notes #5).
    CRP mode: retrieval priority order.
    """
    pairs: set[tuple[int, int]] = set()
    if inst.mode == "crp":
        order = inst.priorities
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                pairs.add((order[i], order[j]))
        return frozenset(pairs)
    for stack in inst.stowage:
        for lo in range(len(stack)):
            for hi in range(lo + 1, len(stack)):
                pairs.add((stack[lo], stack[hi]))
    for far in range(inst.s_v):
        for near in range(far + 1, inst.s_v):
            depth = min(len(inst.stowage[far]), len(inst.stowage[near]))
            for k in range(depth):
                pairs.add((inst.stowage[far][k], inst.stowage[near][k]))
    return frozenset(pairs)


def lower_bound(yard: Sequence[Sequence[int]], precede: frozenset[tuple[int, int]]) -> int:
    """h(n) = remaining containers + number of 2-cycles (Eq 29).

    Yard edges use DIRECT adjacency (notes #4): container `above` directly on
    `below` forms a cycle iff `below` must be placed before `above` — forcing
    one relocation of `above`. Movers are distinct, so the count is admissible.
    The paper's 1/2 factor is absorbed by counting each unordered pair once.
    """
    n_l = sum(len(s) for s in yard)
    cycles = 0
    for stack in yard:
        for k in range(1, len(stack)):
            above, below = stack[k], stack[k - 1]
            if (below, above) in precede:
                cycles += 1
    return n_l + cycles
