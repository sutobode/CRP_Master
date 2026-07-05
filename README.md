# CRPSP — Faithful Reproduction of Wang et al. (2025)

Reproducible baseline for the **Container Relocation Problem with Storage Plan (CRPSP)**, implementing — equation by equation — the hybrid exact/RL framework of:

> Peixiang Wang, Qihang Xu, Yufei Li, Qunlong Chen, Jinghan Tao, Wei Qin, Heng Huang, Ying Zou (2025).
> **"Learning-based hybrid algorithms for container relocation problem with storage plan."**
> *Transportation Research Part E: Logistics and Transportation Review*, 197, 104048.
> DOI: [10.1016/j.tre.2025.104048](https://doi.org/10.1016/j.tre.2025.104048)

This repository is meant to be used as a **verified starting point for new research** (new reward shaping, new architectures, new problem variants) — not as a one-off script. Section ["Extending this baseline"](#extending-this-baseline-for-new-research) below is written specifically for that purpose.

---

## 1. What CRPSP is (30-second primer)

A yard bay holds `N` containers stacked in `S_y` yard stacks (max `T_y` tiers). Each container has a **predetermined slot** on the vessel (`S_v` vessel stacks, same `T_y` tier cap). A crane can only move the **top** container of a stack. Two operation types:

- **Relocation**: move the top container of one yard stack onto the top of another yard stack (agent decision).
- **Transfer**: move the top container of a yard stack directly onto its designated vessel slot, *if that slot is currently the top of its vessel stack* (happens automatically whenever feasible).

Extra loading constraint (Eq. 23): at every instant, vessel stacks nearer the shore must never be taller than vessel stacks farther from the shore.

**Objective**: minimize the number of relocations needed to move all containers from yard to vessel while respecting that constraint. This is NP-hard (Tanaka & Voß, 2019).

The paper attacks this with **four cooperating methods**, all implemented here:

1. An exact **MIP model** (small instances, ground truth).
2. An **A\* search** with a problem-specific admissible lower bound (medium instances, ground truth + training labels).
3. A **PPO agent** with a self-attention actor-critic, whose reward is *shaped* by the A\* lower bound so it converges faster than a plain `-1`/step reward.
4. **Ensembles** (Majority Voting, Stacking-with-XGBoost) that combine several trained PPO agents.

---

## 2. Faithfulness map — every paper component → code → proof

This is the core promise of this repo: **each algorithm is the one the paper proposes**, not a "spirit of the paper" reimplementation. The right-most column says *how you can verify it yourself*, not just where the code lives.

| Paper component | Equations / Section | Code | Verified by |
|---|---|---|---|
| Problem definition, instance generation | Section 2, 4.2, Fig. 4 | [`crpsp/instance.py`](crpsp/instance.py) | `tests/test_instance.py` (8 tests): labeling convention, Eq-23-consistent height profile, **yard AND vessel capacity enforced at T_y** |
| MIP model | Eq. (1)–(26) | [`crpsp/mip_cpsat.py`](crpsp/mip_cpsat.py) (OR-Tools CP-SAT instead of Gurobi) | `tests/test_mip.py`: 3 direct tests + **`@pytest.mark.slow` cross-check — MIP optimal objective == A\* optimal on random N=5 instances** |
| A\* admissible lower bound `h(n)` | Eq. (27)–(30) | [`crpsp/lower_bound.py`](crpsp/lower_bound.py) | `tests/test_lower_bound.py`: hand-computed cycle cases + admissibility check against 30 random instances |
| A\* search | Section 3, Fig. 5–7 | [`crpsp/astar.py`](crpsp/astar.py) | `tests/test_astar.py`: optimality on hand-built cases + **returned trajectory replays exactly through the real environment** (`test_trajectory_replays_to_solution`) |
| Environment: state Θ, action (s,d), auto-transfer closure, reward | Section 4.2, Eq. (36)–(37), Fig. 8 | [`crpsp/env.py`](crpsp/env.py) | `tests/test_env.py`: reward formula checked against a hand-computed example, Eq. (23) checked on every step of 10 random 50-step rollouts, action masking correctness |
| Self-attention actor/critic | Eq. (38)–(39), Fig. 9 | [`crpsp/models.py`](crpsp/models.py) | `tests/test_models.py`: attention weights sum to 1 (softmax correctness), masking, gradient flow |
| PPO training loop | Algorithm 1, Section 4.1–4.2, Table 10 hyperparameters | [`crpsp/ppo.py`](crpsp/ppo.py) | `tests/test_ppo.py`: **GAE advantages and TD targets checked against a hand-computed numeric example**, config-from-YAML matches Table 10 exactly, checkpoint save/load roundtrip |
| Reward-shaping ablation (designed vs. simple `-1`) | Section 5.5 | `CRPSPEnv(reward_mode=...)` in `env.py` | `experiments/table6_7_reward_ablation.py` |
| Attention ablation | Section 5.4 | `Actor/Critic(use_attention=...)` in `models.py` | `experiments/table5_attention_ablation.py` |
| Majority Voting | Eq. (40) | [`crpsp/ensemble.py`](crpsp/ensemble.py) `vote()` | `tests/test_ensemble.py`: matches single-model argmax, tie-break, never picks a masked action |
| Stacking (XGBoost aggregator) | Eq. (41)–(46) | [`crpsp/ensemble.py`](crpsp/ensemble.py) `build_stacking_dataset`/`train_stacking` | `tests/test_ensemble.py`: dataset built from real A\* trajectories, prediction respects the action mask |
| Forward-looking greedy heuristic (baseline) | Section 5.7 | [`crpsp/heuristic.py`](crpsp/heuristic.py) | `tests/test_heuristic.py`: always solves, never beats the A\* optimum |
| Tables 2–9 (experiments) | Section 5 | [`experiments/table2_mip.py`](experiments/table2_mip.py) … [`table9_voting_vs_heuristic.py`](experiments/table9_voting_vs_heuristic.py) | `tests/test_experiments_smoke.py` (wiring only — see §5) |

**Full test suite: 62 tests, all passing** (`.venv/Scripts/python.exe -m pytest`, ~40s).

### Standard-CRP mode (extension beyond the paper)

[`crpsp/instance.py::load_caserta`](crpsp/instance.py) + `CRPSPEnv(restricted=...)` add a second, priority-based mode compatible with the classic Caserta/Schwarze/Voß Block Relocation Problem benchmark format — not in the paper, added so the same solvers can later be benchmarked against the wider CRP literature. Covered by `tests/test_crp_mode.py`.

---

## 3. Repository layout

```
crpsp/
  instance.py       Instance dataclass, Section-4.2 generator, Caserta loader
  lower_bound.py     Eq (27)-(30): admissible lower bound h(n)
  transfer.py        shared transfer-closure logic (used by env, A*, heuristic — single
                      source of truth so all solvers agree on the paper's transition rule)
  env.py              Gym-style environment: state/action/reward (Eq 36), Eq-23 enforcement
  astar.py            A* search (Section 3) + optimal-trajectory extraction
  mip_cpsat.py        Eq (1)-(26) on OR-Tools CP-SAT
  heuristic.py        forward-looking greedy baseline (Section 5.7)
  models.py           self-attention Actor/Critic (Eq 38-39, Fig. 9)
  ppo.py              PPO trainer = Algorithm 1, verbatim (GAE, clipped objective, TD targets)
  ensemble.py         Majority Voting (Eq 40) + Stacking/XGBoost (Eq 41-46)
  evaluate.py         gap() definition, generic policy rollout helper
experiments/
  common.py           CLI plumbing shared by every table script (--smoke/--full, CSV output,
                       A* time/node safety limits)
  train_ppo.py         full PPO training entry point — refuses to run without --full
  table2_mip.py .. table9_voting_vs_heuristic.py   one script per paper table
configs/
  default.yaml         every hyperparameter from Table 10, plus env/model settings
tests/                 61 unit/integration tests + 1 slow cross-check
docs/superpowers/
  specs/, plans/        the original design spec and implementation plan for this repo
REPRODUCTION_NOTES.md  every place the paper was ambiguous, what we chose, and why
```

---

## 4. Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe ortools xgboost scikit-learn numpy pyyaml pytest
```

On the training server, install a CUDA build of torch instead of the CPU wheel (everything else is identical — `TrainConfig.device: "auto"` picks up CUDA automatically).

---

## 5. Verify (fast, laptop-safe — this is where "correct" is proven, not asserted)

```bash
.venv/Scripts/python.exe -m pytest              # 62 tests, ~40s
.venv/Scripts/python.exe -m pytest -m slow      # 1 test: MIP optimal == A* optimal (~5s)
```

What this actually checks, concretely — not just "code runs":

- **The A\* lower bound is provably admissible** on random instances (never overestimates true remaining ops).
- **The MIP and A\* agree on the optimal number of operations** on the same random instances — two completely independently-coded solvers of the same paper's model converging to the same answer is the strongest evidence the model itself was translated correctly.
- **A\*'s returned optimal trajectory literally replays through the live environment** and reproduces the same operation count — the solver and the environment are not just individually "correct in isolation," they agree with each other.
- **Eq. (23) (the shoreline height constraint) is checked on every single step** of randomized rollouts, not just at construction time.
- **GAE and TD-target arithmetic is checked against a hand-solved numeric example** — not just "does it run without crashing."
- **`tests/test_experiments_smoke.py`** additionally guarantees `experiments/train_ppo.py` *refuses to execute* without `--full`, and every table script finishes its smoke path in seconds with bounded CPU/RAM (see `experiments/common.py::astar_time_limit/astar_node_limit`) — this repo was built under a hard "no heavy compute on the dev machine" constraint; only `--full` runs (server-only) approach the paper's actual scale.

Two real bugs were found and fixed by this exact process during a post-implementation review — see items **#20–#21** in `REPRODUCTION_NOTES.md` for the full story (vessel-stack capacity wasn't enforced by the generator; A\* had no time/memory ceiling in smoke mode). This is the kind of thing the test suite above is designed to catch if it happens again.

---

## 6. Full-scale training & experiments (SERVER ONLY)

Every experiment script defaults to a tiny `--smoke` run; paper-scale execution requires an explicit `--full` flag. Do not run `--full` on a laptop.

```bash
# 1. Train the RL policy (repeat with --seed 0..9 for the 10-model ensemble in Table 8)
python experiments/train_ppo.py --full --seed 0 --out checkpoints/m0.pt

# 2. Exact/heuristic baselines (Tables 2, 3)
python experiments/table2_mip.py --full
python experiments/table3_astar.py --full

# 3. Trained-policy comparison against A* (Table 4)
python experiments/table4_ppo_vs_astar.py --full --ckpt checkpoints/m0.pt

# 4. Ablations (Tables 5, 6-7) — each spawns fresh PPOTrainer instances
python experiments/table5_attention_ablation.py --full
python experiments/table6_7_reward_ablation.py --full

# 5. Ensembles (Tables 8, 9) — need all 10 checkpoints from step 1
python experiments/table8_ensemble.py --full --ckpt-dir checkpoints/
python experiments/table9_voting_vs_heuristic.py --full --ckpt-dir checkpoints/
```

Results land in `results/*.csv`, checkpoints + per-iteration metrics in `checkpoints/`.

**Sanity targets from the paper** (Section 5) to check your server run against: PPO gap ≈ 7.33% vs. A\* optimal at (N=15,S_y=5,S_v=5,T_y=5); attention converges in ~50–200 iterations vs. ~1500–2200 without (Table 5); Majority Voting gap ≈ 3.33%, Stacking noticeably *worse* (~44%) — that last one being counter-intuitively bad is a real reported finding, not a bug if you reproduce it.

---

## 7. Known deviations from the literal paper text

Full list with justification: **[`REPRODUCTION_NOTES.md`](REPRODUCTION_NOTES.md)** (21 documented items). The ones that matter most if you're about to build on this code:

- **MIP solver**: OR-Tools CP-SAT, not Gurobi (license-free). Same model, same objective weights (ω1:ω2 = 10:1); optimal *objective values* coincide with A\* (verified), but wall-clock times are not comparable to the paper's Table 2 Gurobi numbers.
- **Two underspecified paper details you'll need to pick a value for if you tune them**: the terminal "large positive reward" magnitude (default `10.0`, `notes #1`) and the Tables 5–7 "convergence iteration" criterion (`notes #10`, implemented in `crpsp.ppo.convergence_iteration`).
- **Two real ambiguities in the paper's own text**, resolved in favor of the *equations* over the *prose* where they conflicted (Fig. 4's height profile vs. Eq. 23; the lower-bound graph's shoreline edge direction vs. Eq. 23) — `notes #3, #5`.

---

## 8. Extending this baseline for new research

This codebase was designed so a new research direction plugs in at exactly one seam, without touching the verified core.

| If your idea changes... | Touch only... | Leave untouched |
|---|---|---|
| **Reward shaping** (a different lower bound, a different shaping term) | `crpsp/env.py::step()` reward branch, or a new function alongside `lower_bound()` | `crpsp/astar.py`, `mip_cpsat.py` stay the ground-truth oracle you compare against |
| **State representation** (e.g. slot-level bay-row-tier instead of stack-only, or add container attributes like 20/40ft, reefer) | `CRPSPEnv.encode()` / a new `Instance` field + `Actor`/`Critic` input dim | `ppo.py`'s training loop is architecture-agnostic — it only calls `actor(obs, mask)`/`critic(obs)` |
| **Action masking for new constraints** (e.g. size-compatibility masks) | `CRPSPEnv._mask()` | Everything downstream already treats the mask as the single source of truth for validity |
| **New network architecture** (multi-head attention, decoder, pointer network) | `crpsp/models.py` — keep the `Actor.forward(obs, mask) -> logits` / `Critic.forward(obs) -> value` contract | `ppo.py`, `ensemble.py` only depend on that contract, not on internals |
| **A genuinely new problem variant** (e.g. mixed container sizes, multi-port) | Add a new `Instance.mode` value (the codebase already dispatches on `mode` for `"crpsp"` vs `"crp"` in `lower_bound.py`, `env.py`, `astar.py`, `transfer.py`) | The dispatch pattern is already there; follow it rather than branching ad hoc |
| **A new baseline to compare against** | A new module exposing the same shape as `HeuristicResult`/`AStarResult`/`MipResult` (`total_ops`, `relocations`, `optimal`/`solved`) | `crpsp/evaluate.py::gap()` and `rollout_policy()` already work with anything shaped that way |

Recommended first move given the codebase you already have: this repo's `Actor`/`Critic` operate at the **master/block level implicit in the paper's own state matrix** (one row per stack, not per individual slot). A concrete, well-scoped extension — already discussed as a promising gap versus the wider RL-for-terminal-logistics literature — is to refine `Instance`/`env.py` to a **slot-level action space** (bay-row-tier, not just stack index) with masking for mixed 20/40ft/reefer containers, keeping this repo's PPO/attention/A\*-lower-bound-reward machinery as the backbone and the paper's own PPO (this repo, unmodified) as your primary experimental baseline.

---

## 9. Citation

If you build on this reproduction, cite the original paper:

```bibtex
@article{wang2025crpsp,
  title   = {Learning-based hybrid algorithms for container relocation problem with storage plan},
  author  = {Wang, Peixiang and Xu, Qihang and Li, Yufei and Chen, Qunlong and Tao, Jinghan and Qin, Wei and Huang, Heng and Zou, Ying},
  journal = {Transportation Research Part E: Logistics and Transportation Review},
  volume  = {197},
  pages   = {104048},
  year    = {2025},
  doi     = {10.1016/j.tre.2025.104048}
}
```
