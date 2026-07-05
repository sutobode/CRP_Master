# Reproduction Notes — Wang et al. (2025) CRPSP

Every decision made where the paper is ambiguous or silent. Format:
**[component] question** → decision → justification.

1. **[env] Terminal bonus magnitude** — paper says "a large positive reward" (Section 4.2), value unspecified → default `terminal_bonus: 10.0`, configurable; applied only in `designed` mode (the simple baseline is defined as exactly −1/step in Section 5.5).
2. **[instance] Container labeling** — labels 1..N assigned per stowage plan, iterating stacks shore→far, bottom→top (Fig. 4 reading). Any fixed convention is equivalent up to relabeling.
3. **[instance] Fig. 4 height profile** — figure appears to violate Eq (23); we follow Eq (23): generated stowage height profiles are non-increasing toward shore.
4. **[lower_bound] Yard edges use direct adjacency** ("situated atop" read as directly-on-top). With transitive edges the 2-cycle count can exceed the true optimum (two cycles sharing one mover), breaking admissibility; direct adjacency keeps movers unique → admissible. Verified against CP-SAT/A* optima in tests.
5. **[lower_bound] Shoreline edge direction** — paper prose says edges go shore→far, but Eq (23) implies far tier-k must precede near tier-k. We follow Eq (23).
6. **[mip] Eq (11) relaxed to fire only on operation steps** (`h_S ≥ 1` only if an op occurs at t) so the horizon can exceed the exact op count; the paper is silent on how the horizon was chosen.
7. **[models] Post-attention layers** — Fig. 9 is schematic; we use flatten → Linear(·,128) → ReLU → Linear(128, out). Hidden dim 128 per Table 10.
8. **[ppo] No entropy bonus, no grad clipping** — neither is mentioned in the paper; omitted.
9. **[ppo] Value targets** — `y_t = r_t + γV(s_{t+1})` per Algorithm 1 line 17 (TD(0), not GAE returns), bootstrapping 0 at terminal states.
10. **[ppo] Convergence iteration (Tables 5–7)** — undefined in paper → we define: first iteration ending a 20-iteration window with 100% solve-rate and stable mean relocations; capped at 1000.
11. **[evaluate] Gap** = (total_ops_alg − total_ops_opt) / total_ops_opt, consistent with Table 4 values (5.00% ≈ 1/20, 9.09% = 2/22).
12. **[heuristic] "Minimally impacts the loading process"** — destination chosen minimizing (# containers already in dest that must precede the moved container, resulting height).
13. **[env] Invalid actions** — paper doesn't discuss masking; we mask at sampling (source non-empty, dest not full, s≠d) and the env raises on invalid input.
