"""CP-SAT translation of the CRPSP MIP, Eq (1)-(26) of Wang et al. (2025).

Faithful boolean translation with 0-based tiers (the paper is 1-based).
Documented deviations (REPRODUCTION_NOTES.md):
  #6  Eq (11) is enforced only on steps where an operation occurs, so the
      horizon may exceed the exact operation count (idle steps allowed).
  A redundant gravity constraint is added as a solver aid (implicit in the
  paper via top-to-top moves from a gravity-consistent initial state).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .instance import Instance


@dataclass
class MipResult:
    status: str
    total_ops: int
    relocations: int
    target_reached: bool
    wall_time_s: float


def _and2(model: cp_model.CpModel, a, b, name: str):
    """z <=> a AND b for boolean literals."""
    z = model.NewBoolVar(name)
    model.AddBoolAnd([a, b]).OnlyEnforceIf(z)
    model.AddBoolOr([a.Not(), b.Not()]).OnlyEnforceIf(z.Not())
    return z


def solve_mip(inst: Instance, horizon: int | None = None,
              time_limit_s: float = 120.0, w1: int = 10, w2: int = 1) -> MipResult:
    n, s_y, s_v, t_y = inst.n, inst.s_y, inst.s_v, inst.t_y
    n_stacks = s_y + s_v                     # 0..s_y-1 yard, s_y.. vessel
    k_max = max(t_y, max((len(st) for st in inst.stowage), default=1))
    H = horizon if horizon is not None else 2 * n
    m = cp_model.CpModel()

    # s[i,k,c,t]: container c (id c+1) at stack i tier k before op t; t in 0..H
    s = {(i, k, c, t): m.NewBoolVar(f"s_{i}_{k}_{c}_{t}")
         for i in range(n_stacks) for k in range(k_max)
         for c in range(n) for t in range(H + 1)}
    # x[i,j,t]: move from stack i to stack j at step t
    x = {(i, j, t): m.NewBoolVar(f"x_{i}_{j}_{t}")
         for i in range(n_stacks) for j in range(n_stacks) for t in range(H)}

    for t in range(H):
        for i in range(n_stacks):
            m.Add(x[i, i, t] == 0)                                   # Eq (2)
            if i >= s_y:                                             # Eq (3)
                for j in range(n_stacks):
                    m.Add(x[i, j, t] == 0)

    # yard stacks never exceed t_y tiers
    for i in range(s_y):
        for k in range(t_y, k_max):
            for c in range(n):
                for t in range(H + 1):
                    m.Add(s[i, k, c, t] == 0)

    for c in range(n):                                               # Eq (4)
        for t in range(H + 1):
            m.AddExactlyOne(s[i, k, c, t] for i in range(n_stacks) for k in range(k_max))
    for i in range(n_stacks):                                        # Eq (5)
        for k in range(k_max):
            for t in range(H + 1):
                m.AddAtMostOne(s[i, k, c, t] for c in range(n))
    # gravity (redundant solver aid)
    for i in range(n_stacks):
        for k in range(1, k_max):
            for t in range(H + 1):
                m.Add(sum(s[i, k, c, t] for c in range(n))
                      <= sum(s[i, k - 1, c, t] for c in range(n)))

    op, o_s, o_e, height = {}, {}, {}, {}
    for t in range(H):
        m.Add(sum(x[i, j, t] for i in range(n_stacks)
                  for j in range(n_stacks)) <= 1)                    # Eq (6)
        op[t] = m.NewBoolVar(f"op_{t}")
        m.Add(sum(x[i, j, t] for i in range(n_stacks) for j in range(n_stacks)) == op[t])
        for i in range(n_stacks):
            o_s[i, t] = m.NewBoolVar(f"oS_{i}_{t}")                  # Eq (7)
            m.Add(sum(x[i, j, t] for j in range(n_stacks)) == o_s[i, t])
            o_e[i, t] = m.NewBoolVar(f"oE_{i}_{t}")                  # Eq (8)
            m.Add(sum(x[j, i, t] for j in range(n_stacks)) == o_e[i, t])
    for i in range(n_stacks):                                        # Eq (22)
        for t in range(H + 1):
            height[i, t] = m.NewIntVar(0, k_max, f"h_{i}_{t}")
            m.Add(height[i, t] == sum(s[i, k, c, t] for k in range(k_max) for c in range(n)))

    h_s, h_e, g_s, g_e, r_s, r_e = {}, {}, {}, {}, {}, {}
    for t in range(H):
        h_s[t] = m.NewIntVar(0, k_max, f"hS_{t}")                    # Eq (9)
        h_e[t] = m.NewIntVar(0, k_max, f"hE_{t}")                    # Eq (10)
        zs, ze = [], []
        for i in range(n_stacks):
            zi = m.NewIntVar(0, k_max, f"zS_{i}_{t}")
            m.Add(zi == height[i, t]).OnlyEnforceIf(o_s[i, t])
            m.Add(zi == 0).OnlyEnforceIf(o_s[i, t].Not())
            zs.append(zi)
            zj = m.NewIntVar(0, k_max, f"zE_{i}_{t}")
            m.Add(zj == height[i, t]).OnlyEnforceIf(o_e[i, t])
            m.Add(zj == 0).OnlyEnforceIf(o_e[i, t].Not())
            ze.append(zj)
        m.Add(h_s[t] == sum(zs))
        m.Add(h_e[t] == sum(ze))
        m.Add(h_s[t] >= 1).OnlyEnforceIf(op[t])                      # Eq (11), notes #6
        m.Add(h_e[t] <= k_max - 1)                                   # Eq (12)
        for k in range(k_max):
            g_s[k, t] = m.NewBoolVar(f"gS_{k}_{t}")
            g_e[k, t] = m.NewBoolVar(f"gE_{k}_{t}")
        m.Add(sum((k + 1) * g_s[k, t] for k in range(k_max)) == h_s[t])  # Eq (13)
        m.Add(sum(g_s[k, t] for k in range(k_max)) == op[t])             # Eq (14)
        m.Add(sum(k * g_e[k, t] for k in range(k_max)) == h_e[t])        # Eq (15)
        m.Add(sum(g_e[k, t] for k in range(k_max)) == op[t])             # Eq (16)
        for i in range(n_stacks):
            for k in range(k_max):
                r_s[i, k, t] = _and2(m, o_s[i, t], g_s[k, t], f"rS_{i}_{k}_{t}")  # Eq (17)
                r_e[i, k, t] = _and2(m, o_e[i, t], g_e[k, t], f"rE_{i}_{k}_{t}")  # Eq (18)

    # Eq (19): initial state — containers in yard, vessel empty
    init = {(i, k): c for i, stack in enumerate(inst.yard) for k, c in enumerate(stack)}
    for i in range(n_stacks):
        for k in range(k_max):
            for c in range(n):
                m.Add(s[i, k, c, 0] == (1 if init.get((i, k)) == c + 1 else 0))

    # Eq (20)-(21): moved container and state transition
    for t in range(H):
        for c in range(n):
            mf = {}
            for i in range(n_stacks):
                for k in range(k_max):
                    mf[i, k] = _and2(m, r_s[i, k, t], s[i, k, c, t], f"mf_{i}_{k}_{c}_{t}")
            mc = m.NewBoolVar(f"m_{c}_{t}")
            m.Add(sum(mf.values()) == mc)                            # Eq (20)
            for i in range(n_stacks):
                for k in range(k_max):
                    mt = _and2(m, r_e[i, k, t], mc, f"mt_{i}_{k}_{c}_{t}")
                    m.Add(s[i, k, c, t + 1] == s[i, k, c, t] - mf[i, k] + mt)  # Eq (21)

    # Eq (23): vessel monotonicity at every time slice (index 0 = furthest)
    for t in range(H + 1):
        for va in range(s_v):
            for vb in range(va + 1, s_v):
                m.Add(height[s_y + vb, t] <= height[s_y + va, t])

    # Eq (24) + objective Eq (1); S_T constant so |s - S_T| is a literal
    target = {(s_y + vs, k): c for vs, stack in enumerate(inst.stowage)
              for k, c in enumerate(stack)}
    mis_literals = []
    for i in range(n_stacks):
        for k in range(k_max):
            for c in range(n):
                if target.get((i, k)) == c + 1:
                    mis_literals.append(s[i, k, c, H].Not())
                else:
                    mis_literals.append(s[i, k, c, H])
    m.Minimize(w1 * sum(mis_literals)
               + w2 * sum(x[i, j, t] for i in range(n_stacks)
                          for j in range(n_stacks) for t in range(H)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    t0 = time.perf_counter()
    status = solver.Solve(m)
    wall = time.perf_counter() - t0
    names = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE",
             cp_model.INFEASIBLE: "INFEASIBLE"}
    sname = names.get(status, "UNKNOWN")
    if sname not in ("OPTIMAL", "FEASIBLE"):
        return MipResult(sname, -1, -1, False, wall)
    total = sum(int(solver.BooleanValue(x[i, j, t])) for i in range(n_stacks)
                for j in range(n_stacks) for t in range(H))
    reloc = sum(int(solver.BooleanValue(x[i, j, t])) for i in range(s_y)
                for j in range(s_y) for t in range(H))
    mis = sum(int(solver.BooleanValue(lit)) for lit in mis_literals)
    return MipResult(sname, total, reloc, mis == 0, wall)
