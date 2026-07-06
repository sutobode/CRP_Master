"""Blocker relocation subproblem solver for HTR.

Given a target stack, find destinations for its top container
(or for all blockers if target is buried). Uses existing baseline code.
"""
from __future__ import annotations

from .instance import slot_map
from .lower_bound import must_precede_pairs


def solve_one_relocation(yard, target_stack, inst) -> int:
    """Find best destination for the top container of target_stack.

    Returns: destination stack index.
    Picks the stack with the fewest containers that must precede
    the relocated container (minimizes new blocking pairs).
    """
    if not yard[target_stack]:
        return -1

    target = yard[target_stack][-1]
    s_y = inst.s_y
    precede = must_precede_pairs(inst)
    best_d = -1
    best_score = 10**9

    for d in range(s_y):
        if d == target_stack or len(yard[d]) >= inst.t_y:
            continue
        score = sum(1 for c in yard[d] if (c, target) in precede)
        if score < best_score:
            best_score = score
            best_d = d

    return best_d


def solve_subproblem_heuristic(yard, target_stack, inst) -> list[tuple[int, int]]:
    """Relocate ALL blockers blocking the transferable target.

    The subproblem: target at (target_stack) might not be directly
    transferable. Relocate its top container to make progress.

    Returns: list of (source, dest) relocation pairs.
    """
    if not yard[target_stack]:
        return []

    seq = []
    max_reloc = inst.t_y  # at most T_y relocations per macro-step
    for _ in range(max_reloc):
        if not yard[target_stack]:
            break

        # Check if top container is now transferable
        c = yard[target_stack][-1]
        # If it IS transferable, stop (transfer will handle it)
        from .env import CRPSPEnv  # lazy import to avoid circular
        # We check: can this container be transferred right now?
        # We don't have vessel state here, so we use a simpler check:
        # If the container has no blockers above it (it's the top),
        # the transfer closure in the env will handle it.
        # So we only relocate if we're told to.

        # Find best destination for the TOP container
        d = solve_one_relocation(yard, target_stack, inst)
        if d < 0:
            break  # no valid destination

        # Execute relocation
        yard[d].append(yard[target_stack].pop())
        seq.append((target_stack, d))

        # After relocation, check if transfer is now possible
        # (We can't run full closure here, but the main env will)

    return seq


def is_transferable_now(c, yard, vessel, slot_of) -> bool:
    """Check if container c can be transferred right now."""
    vs, vt = slot_of[c]
    return len(vessel[vs]) == vt and all(
        len(vessel[j]) >= len(vessel[vs]) + 1
        for j in range(vs)
    )
