# Reproduction Notes — Wang et al. (2025) CRPSP

Every decision made where the paper is ambiguous or silent. Format:
**[component] question** → decision → justification.

1. **[env] Terminal bonus magnitude** — paper says "a large positive reward" (Section 4.2), value unspecified → default `terminal_bonus: 10.0`, configurable; applied only in `designed` mode (the simple baseline is defined as exactly −1/step in Section 5.5).
2. **[instance] Container labeling** — labels 1..N assigned per stowage plan, iterating stacks shore→far, bottom→top (Fig. 4 reading). Any fixed convention is equivalent up to relabeling.
3. **[instance] Fig. 4 height profile** — the figure appears to violate Eq (23); we follow Eq (23): generated stowage height profiles are non-increasing toward shore (vessel stack index 0 = furthest from shore = tallest).
4. **[lower_bound] Yard edges use direct adjacency** — "situated atop" read as directly-on-top. With transitive edges the 2-cycle count can exceed the true optimum (two cycles sharing one mover), breaking admissibility; direct adjacency keeps movers unique → admissible. Verified empirically: `test_lb_is_admissible_on_random_instances` and the CP-SAT/A* cross-check.
5. **[lower_bound] Shoreline edge direction** — the paper's prose says edges go shore→far, but Eq (23) implies tier-k of a FARTHER stack must be placed before tier-k of a NEARER stack. We follow Eq (23) (edges far→near).
6. **[mip] Eq (11) relaxed to operation steps** — `h_S ≥ 1` is enforced only when an operation occurs at step t, allowing idle steps so the horizon may exceed the exact operation count. The paper is silent on horizon selection; experiments use `heuristic_total_ops + 2`.
7. **[models] Post-attention layers** — Fig. 9 is schematic; we use attention → flatten → Linear(·,128) → ReLU → Linear(128, out). Hidden dim 128 per Table 10. Single-head attention exactly as Eq (38)–(39); no positional encoding (none is described).
8. **[ppo] No entropy bonus, no gradient clipping** — neither is mentioned in the paper; omitted.
9. **[ppo] Value targets** — `y_t = r_t + γ·V(s_{t+1})` per Algorithm 1 line 17 (TD(0), not GAE returns), bootstrapping 0 at terminal states. Values are recomputed with the current critic after collection (Algorithm 1 lines 13–18), and one shuffled mini-batch pass per iteration (`ppo_epochs: 1`; Algorithm 1 shows a single pass).
10. **[ppo] Convergence iteration (Tables 5–7)** — undefined in the paper → defined as: end index of the first 20-iteration window in which every iteration has solve_rate = 100% and mean relocations are stable (non-degrading within the window). Implemented in `crpsp.ppo.convergence_iteration`; capped at 1000 iterations as in Section 5.5.
11. **[evaluate] Gap** = (total_ops_alg − total_ops_opt) / total_ops_opt, consistent with Table 4 values (5.00% ≈ 1/20, 9.09% = 2/22). Total operations = transfers + relocations; transfers are constant (=N), so ranking by gap equals ranking by relocations.
12. **[heuristic] "Minimally impacts the loading process"** — destination chosen by lexicographic minimum of (# containers already in the destination that must precede the moved container, resulting stack height).
13. **[env] Invalid actions** — the paper doesn't discuss masking; we mask at sampling time (source non-empty, destination not full, s≠d) and the env raises `ValueError` on invalid input.
14. **[env] State encoding** — Θ entries are container ids normalized by N (`id/n`), 0 = empty. The paper specifies the matrix layout (rows = stacks) but not the numeric scaling.
15. **[ensemble] Stacking label encoding** — XGBoost requires contiguous class labels; sparse action ids are re-indexed for training and mapped back at prediction. Prediction is restricted to currently valid (masked) actions.
16. **[ensemble] Voting tie-break** — Eq (40) leaves ties undefined → broken by the highest summed probability across models.
17. **[crp] Caserta format (extension, not in the paper)** — rows are read bottom→top; yard height limit follows the CV convention H_max = H + 2; container ids double as retrieval priorities. Restricted variant masks relocation sources to the stack holding the current target.
18. **[mip] Solver** — OR-Tools CP-SAT instead of Gurobi (license-free). Same model and objective (ω1:ω2 = 10:1, Eq 1); optimal objective values coincide (verified vs A*), wall-clock times are not comparable to Table 2's Gurobi times.
19. **[general] Instance parameters** — Section 5 states "stacks ranging from 4 to 11, container numbers from 10 to 30"; per-table parameter grids are taken verbatim from Tables 2–9 and encoded in `experiments/`.
