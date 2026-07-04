# Design: Faithful Reproduction of Wang et al. (2025) — CRPSP Hybrid Learning Algorithms

**Date:** 2026-07-04
**Source paper:** Peixiang Wang et al., "Learning-based hybrid algorithms for container relocation problem with storage plan", *Transportation Research Part E* 197 (2025) 104048. DOI: 10.1016/j.tre.2025.104048. Full text: `Paper/1-s2.0-S1366554525000894-main (1).pdf`.

## Goal

Implement, exactly as proposed in the paper, the full CRPSP solution framework: MIP model, A* algorithm with cycle-based lower bound, PPO with self-attention actor-critic and lower-bound-shaped reward, ensemble methods (Majority Voting, Stacking/XGBoost), and the forward-looking greedy heuristic baseline — plus a standard-CRP mode (Caserta-compatible) as an extension for future benchmarking.

**Hard constraint:** This project delivers code + tests only. No full training runs, no heavy compute on this machine. Smoke tests must be tiny (seconds-to-a-few-minutes on CPU). Full training/experiments run later on the user's server.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Scope | Faithful reproduction + standard-CRP mode |
| MIP solver | OR-Tools CP-SAT (free) instead of Gurobi; same model Eq (1)–(26) |
| Compute | Device-agnostic PyTorch; CPU for dev/tests; server training later |
| RL framework | Pure PyTorch, custom PPO implementing the paper's Algorithm 1 exactly (no SB3/RL4CO) |
| Ambiguities | Resolve with reasonable choices, all logged in `REPRODUCTION_NOTES.md` |
| Python env | Dedicated venv, Python 3.11/3.12 (3.14 lacks stable torch wheels) |

## Problem definition (paper Section 2)

- Yard bay: `S_y` stacks × `T_y` max tiers holding `N` containers. Vessel bay: `S_v` stacks; vessel stack 1 is furthest from shoreline.
- Stowage plan assigns each container a fixed vessel slot; containers labeled 1..N by stowage-plan position (column-major from shore-side stack, bottom-up, per Fig. 4).
- Operations: **relocation** (top of yard stack → top of another non-full yard stack; agent-decided) and **transfer** (top of yard stack → its designated vessel slot when that slot is the current top of its vessel stack; executed automatically whenever feasible).
- Loading constraint (Eq 23, Assumption 4): after every operation, vessel heights satisfy `h_i ≤ h_j` for all vessel stacks `i > j` (nearer shore never taller). Applies at all times including the final state; the generated stowage plan must have a height profile non-increasing toward shore.
- Objective: minimize number of relocations (equivalently total operations, since transfers = N is constant).
- Unrestricted variant: any top container may be relocated.

## Components

### 1. `crpsp/instance.py`
- `Instance` dataclass: yard matrix, stowage plan, S_y, S_v, T_y, N, mode (`crpsp` | `crp`).
- Generator per paper Section 4.2: given stack counts, create N containers, build stowage plan satisfying the shoreline height constraint, label containers by plan position, shuffle into yard stacks randomly. Seeded RNG.
- Caserta `.dat` parser for standard-CRP mode (priorities instead of stowage plan).

### 2. `crpsp/env.py`
- `CRPSPEnv`: gymnasium-style (`reset(instance)`, `step(action)`), no gym dependency required.
- State: matrix Θ ∈ R^(S_y × T_y); entry = container index (normalized by N) or 0.
- Action: discrete index over ordered pairs (s, d), s ≠ d → S_y·(S_y−1) actions. Provides a boolean `action_mask` (source non-empty, destination not full).
- `step`: apply relocation → run transfer closure (repeat all feasible transfers, respecting Eq 23) → return next state, reward, done, info.
- Reward (Eq 36): `r = −1 + h(s_t) − h(s_{t+1})` with `h` = the A* lower bound; on completion add terminal bonus `R_done` (paper: "large positive reward", value unspecified — configurable, default documented in notes). Simple-reward variant `r = −1` selectable for the Section 5.5 ablation.
- CRP mode: no vessel/stowage constraints; retrieval by priority; restricted/unrestricted flag.

### 3. `crpsp/lower_bound.py`
- Directed graph over containers remaining in yard (Eq 27–30): yard-blocking edges (above-in-same-stack), stowage must-precede edges (beneath-in-same-vessel-stack), shoreline precedence edges (tier-k of a farther stack precedes tier-k of a nearer stack, as required by Eq 23 — the paper's prose on edge direction conflicts with Eq 23; we follow Eq 23 and log this in notes).
- `h(n) = N_L + ½ · Σ δ(S_ij + S_ji, 2)` (remaining containers + 2-cycles count).

### 4. `crpsp/astar.py`
- A* per Section 3: nodes = yard states after transfer closure; successors = all feasible relocations; `g` = operations so far, `f = g + h`. Open list (heap) + closed set (hashed canonical state). Returns optimal operation sequence and relocation count. Optional node/time limits.

### 5. `crpsp/mip_cpsat.py`
- CP-SAT translation of Eq (1)–(26): binary variables per Table 1, products of binaries via `AddMultiplicationEquality` or reified constraints; objective weights ω1:ω2 = 10:1; horizon |T| bounded by a heuristic solution length. Intended for N ≤ 10 (Table 2 scale).

### 6. `crpsp/heuristic.py`
- Forward-looking greedy (Section 5.7): repeatedly select the container with the fewest blockers that can be loaded; move blockers to destinations that least impact loading (fewest new blocking pairs, tie-break lowest stack).

### 7. `crpsp/models.py`
- `SelfAttentionActor` and `SelfAttentionCritic` (separate networks, Fig. 9): row-wise self-attention on Θ (stacks as tokens; Q=SW_Q, K=SW_K, V=SW_V; scaled dot-product, Eq 38–39, single head) → flatten → MLP (hidden 128) → action logits (actor, masked) / scalar value (critic). No-attention variants (MLP-only) for the Section 5.4 ablation.

### 8. `crpsp/ppo.py`
- `PPOTrainer` implementing Algorithm 1 verbatim: per iteration generate M fresh instances → rollouts (max T=50 steps) → GAE (γ=0.4, λ=0.9) → value targets `r + γV(s')` → shuffle, minibatches B=10 → clipped objective (ε=0.15), critic MSE; Adam lr 5e-4 both nets (all values from Table 10 in `configs/default.yaml`).
- Convergence metric for Tables 5–7 reproduction: first iteration where a moving-average solve-rate/gap criterion stabilizes (paper leaves this undefined; exact criterion documented in notes). Checkpointing + metrics logging (CSV/JSON) for server runs.

### 9. `crpsp/ensemble.py`
- Voting (Eq 40): mode of per-model argmax actions; ties broken by summed probabilities.
- Stacking (Eq 41–46): features = stacked k×n action-probability matrix from k models; labels = A*-optimal actions; aggregator = XGBoost classifier.

### 10. `crpsp/evaluate.py` + `experiments/`
- Gap = (total_ops_alg − total_ops_opt) / total_ops_opt (consistent with Table 4 values; documented). Timing utilities.
- One runnable script per paper table (`experiments/table2_mip.py` … `table9_voting_vs_heuristic.py`) with `--smoke` flags for tiny local verification; full-scale parameters match the paper (200 instances etc.) for server use.

## Verification strategy (definition of "accurate")

1. **Cross-check optimality:** A* relocation counts must equal CP-SAT optimal on every generated instance with N ≤ 8 (both must also satisfy all constraints).
2. **Env invariants (unit tests):** container conservation; Eq 23 never violated; transfer closure complete and correct; action mask correctness.
3. **LB admissibility:** `h(n)` ≤ true optimal remaining ops on sampled small instances (checked against A*).
4. **PPO machinery tests:** GAE computation against hand-computed values; clipped-loss math; masked sampling never picks invalid actions; one tiny smoke training run (N=5, few iterations, seconds) asserting reward trend improves — explicitly NOT a full training run.
5. **Ablation wiring:** designed vs simple reward and attention vs no-attention selectable purely by config, so server runs can reproduce Tables 5–7 without code changes.

## Out of scope

- Full training (40k iterations), 200-instance evaluations, hyperparameter sweeps — server only.
- Any modification/improvement of the paper's method (that is the *next* research phase).

## Ambiguity log (to maintain in `REPRODUCTION_NOTES.md`)

Known items to resolve and document during implementation: terminal bonus magnitude; post-attention layer shapes (Fig. 9 is schematic); state normalization; shoreline-edge direction in the LB graph (prose vs Eq 23); convergence-iteration criterion; gap denominator; container labeling convention; Fig. 4's apparent height-profile inconsistency with Eq 23.
