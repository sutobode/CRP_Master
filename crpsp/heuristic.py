"""Forward-looking greedy heuristic (paper Section 5.7; notes #12).

"Selects the container with the fewest containers on top that can be directly
loaded onto the ship. It then moves all containers on top of this container to
a position that minimally impacts the loading process." — impact interpreted
as (number of containers in the destination that must precede the moved one,
resulting stack height), minimized lexicographically.
"""
from __future__ import annotations

from dataclasses import dataclass

from .instance import Instance, slot_map
from .lower_bound import must_precede_pairs
from .transfer import closure_crpsp


@dataclass
class HeuristicResult:
    total_ops: int
    relocations: int
    solved: bool


def solve_heuristic(inst: Instance, max_ops: int | None = None) -> HeuristicResult:
    yard = [list(s) for s in inst.yard]
    vessel: list[list[int]] = [[] for _ in range(inst.s_v)]
    slot_of = slot_map(inst)
    precede = must_precede_pairs(inst)
    cap = max_ops if max_ops is not None else 10 * inst.n
    n_rel = n_tra = 0

    def height_ok(vs: int) -> bool:
        h_new = len(vessel[vs]) + 1
        return all(len(vessel[j]) >= h_new for j in range(vs))

    def closure() -> None:
        nonlocal n_tra
        n_tra += closure_crpsp(yard, vessel, slot_of)

    closure()
    while any(yard) and n_rel + n_tra < cap:
        # candidates: containers whose designated slot is placeable right now
        cands = []
        for si, st in enumerate(yard):
            for pos, c in enumerate(st):
                vs, vt = slot_of[c]
                if len(vessel[vs]) == vt and height_ok(vs):
                    cands.append((len(st) - pos - 1, c, si))
        if not cands:
            break
        _, _, si = min(cands)                  # fewest blockers on top
        b = yard[si][-1]                       # relocate topmost blocker
        dests = [d for d in range(inst.s_y) if d != si and len(yard[d]) < inst.t_y]
        if not dests:
            break

        def impact(d: int) -> tuple[int, int]:
            bad = sum(1 for c in yard[d] if (c, b) in precede)  # b would bury c
            return (bad, len(yard[d]))

        d = min(dests, key=impact)
        yard[d].append(yard[si].pop())
        n_rel += 1
        closure()
    return HeuristicResult(n_rel + n_tra, n_rel, all(not s for s in yard))
