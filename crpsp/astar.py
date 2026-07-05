"""A* for the CRPSP (paper Section 3).

Nodes are yard states after transfer closure (a node never holds an
immediately transferable container). Edge weight = number of operations
(one relocation + any transfers it unlocks). g = ops so far, h = Eq-29
lower bound, so f = g + h estimates total operations; minimizing total
operations is equivalent to minimizing relocations (transfers are fixed = N).
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

from .instance import Instance, slot_map
from .lower_bound import lower_bound, must_precede_pairs
from .transfer import closure_crp, closure_crpsp

Yard = tuple[tuple[int, ...], ...]
VHeights = tuple[int, ...]


@dataclass
class AStarResult:
    total_ops: int
    relocations: int
    optimal: bool
    nodes_expanded: int
    trajectory: list[tuple[Yard, VHeights, tuple[int, int]]] = field(default_factory=list)


def solve_astar(inst: Instance, time_limit_s: float | None = None,
                node_limit: int | None = None) -> AStarResult:
    crp = inst.mode == "crp"
    slot_of = slot_map(inst) if not crp else None
    precede = must_precede_pairs(inst)
    s_y, s_v, t_y = inst.s_y, inst.s_v, inst.t_y

    def rebuild_vessel(vh: VHeights) -> list[list[int]]:
        # vessel stack contents are determined by the stowage plan prefix
        return [list(inst.stowage[i][:vh[i]]) for i in range(s_v)]

    def run_closure(yard: list[list[int]], vh: VHeights) -> tuple[int, VHeights]:
        """Apply transfer closure in the right mode.

        The auxiliary state `vh` is vessel heights (CRPSP) or a 1-tuple with
        the next retrieval priority (CRP)."""
        if crp:
            n, nxt = closure_crp(yard, vh[0])
            return n, (nxt,)
        vessel = rebuild_vessel(vh)
        n = closure_crpsp(yard, vessel, slot_of)
        return n, tuple(len(v) for v in vessel)

    yard0 = [list(s) for s in inst.yard]
    root_aux: VHeights = (1,) if crp else tuple(0 for _ in range(s_v))
    g0, root_vh = run_closure(yard0, root_aux)
    root_yard: Yard = tuple(tuple(s) for s in yard0)

    counter = itertools.count()
    start = time.perf_counter()
    heap = [(g0 + lower_bound(root_yard, precede), next(counter), g0, root_yard, root_vh)]
    best_g: dict = {(tuple(sorted(root_yard)), root_vh): g0}
    parent: dict = {(root_yard, root_vh): None}     # child -> (parent_state, action)
    expanded = 0

    while heap:
        if time_limit_s is not None and time.perf_counter() - start > time_limit_s:
            return AStarResult(-1, -1, False, expanded)
        if node_limit is not None and expanded > node_limit:
            return AStarResult(-1, -1, False, expanded)
        f, _, g, yard, vh = heapq.heappop(heap)
        if all(not s for s in yard):
            return AStarResult(g, g - inst.n, True, expanded,
                               _reconstruct(parent, (yard, vh)))
        key = (tuple(sorted(yard)), vh)
        if g > best_g.get(key, float("inf")):
            continue                                 # stale heap entry
        expanded += 1
        for s in range(s_y):
            if not yard[s]:
                continue
            for d in range(s_y):
                if d == s or len(yard[d]) >= t_y:
                    continue
                ny = [list(st) for st in yard]
                ny[d].append(ny[s].pop())
                moved, child_vh = run_closure(ny, vh)
                child_yard: Yard = tuple(tuple(st) for st in ny)
                ng = g + 1 + moved
                ckey = (tuple(sorted(child_yard)), child_vh)
                if ng < best_g.get(ckey, float("inf")):
                    best_g[ckey] = ng
                    parent[(child_yard, child_vh)] = ((yard, vh), (s, d))
                    heapq.heappush(heap, (ng + lower_bound(child_yard, precede),
                                          next(counter), ng, child_yard, child_vh))
    return AStarResult(-1, -1, False, expanded)


def _reconstruct(parent, goal):
    """(pre-state yard, pre-state vessel heights, relocation) along the optimal path."""
    traj = []
    cur = goal
    while parent.get(cur) is not None:
        prev, action = parent[cur]
        traj.append((prev[0], prev[1], action))
        cur = prev
    traj.reverse()
    return traj
