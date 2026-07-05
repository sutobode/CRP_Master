"""Shared transfer-closure logic used by env, A*, and the heuristic (single
implementation so all solvers agree on the paper's State Transition rule)."""
from __future__ import annotations


def closure_crpsp(yard: list[list[int]], vessel: list[list[int]], slot_of) -> int:
    """CRPSP: repeatedly transfer any top container whose designated slot is the
    current top of its vessel stack, respecting Eq (23). Returns transfers done."""
    n = 0
    moved = True
    while moved:
        moved = False
        for stack in yard:
            if not stack:
                continue
            c = stack[-1]
            vs, vt = slot_of[c]
            h_new = len(vessel[vs]) + 1
            if len(vessel[vs]) == vt and all(len(vessel[j]) >= h_new for j in range(vs)):
                vessel[vs].append(stack.pop())
                n += 1
                moved = True
    return n


def closure_crp(yard: list[list[int]], next_priority: int) -> tuple[int, int]:
    """Standard CRP: repeatedly retrieve the top container equal to the next
    retrieval priority. Returns (retrievals done, new next_priority)."""
    n = 0
    moved = True
    while moved:
        moved = False
        for stack in yard:
            if stack and stack[-1] == next_priority:
                stack.pop()
                next_priority += 1
                n += 1
                moved = True
    return n, next_priority
