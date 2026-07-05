# `crpsp` — package reference

This is the core library: solvers, environment, models, and training code for
the CRPSP (Wang et al., 2025) and an extended standard-CRP mode. Everything
under `experiments/` and `tests/` is a *consumer* of this package — the
package itself has no CLI, no I/O side effects beyond `save()`/`load()`, and
no hardcoded instance sizes.

For **what each module implements from the paper and how that's verified**,
see the top-level [`../README.md`](../README.md#2-faithfulness-map--every-paper-component--code--proof).
For **every place the paper was ambiguous and what we chose**, see
[`../REPRODUCTION_NOTES.md`](../REPRODUCTION_NOTES.md). This document is a
plain API/usage reference for working *inside* the package.

## Dependency graph

```
instance.py  (no internal deps)
    |
    +--> lower_bound.py   (Instance)
    +--> transfer.py       (no Instance dep — pure list-of-lists logic)
    |
    +--> env.py           (instance, lower_bound, transfer)
    +--> astar.py          (instance, lower_bound, transfer)
    +--> mip_cpsat.py      (instance)
    +--> heuristic.py      (instance, lower_bound, transfer)
    |
models.py   (no internal deps — pure torch.nn)
    |
    +--> ppo.py           (env, instance, models)
    +--> ensemble.py       (astar, env, instance, models)
    |
evaluate.py (env, instance) — generic, works with any policy/solver
```

`env.py`, `astar.py`, and `heuristic.py` all delegate the actual transfer
rule to `transfer.py` — this is deliberate: there is exactly **one**
implementation of "when does a container move from yard to vessel/get
retrieved," shared by every solver, so they cannot silently disagree with
each other about the environment's transition dynamics.

## Quick start

```python
import random
from crpsp.instance import generate_instance
from crpsp.astar import solve_astar
from crpsp.heuristic import solve_heuristic
from crpsp.mip_cpsat import solve_mip
from crpsp.env import CRPSPEnv

rng = random.Random(0)
inst = generate_instance(n=15, s_y=5, s_v=5, t_y=5, rng=rng)

opt = solve_astar(inst)                       # exact, fast for small/medium N
opt.total_ops, opt.relocations, opt.optimal

h = solve_heuristic(inst)                     # fast, suboptimal baseline
h.total_ops >= opt.total_ops                  # always True

mip = solve_mip(inst, horizon=opt.total_ops + 2, time_limit_s=60)  # exact via CP-SAT
mip.total_ops == opt.total_ops                # should hold when both prove optimality

env = CRPSPEnv(reward_mode="designed")
obs, mask = env.reset(inst)
obs, mask, reward, done, truncated, info = env.step(action=0)  # action = index into env.action_pairs
```

```python
# One PPO training iteration (see experiments/train_ppo.py for the full loop)
import random
from crpsp.ppo import PPOTrainer, TrainConfig

cfg = TrainConfig(n=15, s_y=5, s_v=5, t_y=5, reward_mode="designed",
                  terminal_bonus=10.0, max_steps=50, hidden_dim=128,
                  use_attention=True, lr=5e-4, gamma=0.4, gae_lambda=0.9,
                  clip_eps=0.15, batch_size=10, instances_per_iter=10,
                  iterations=1, ppo_epochs=1, seed=0, device="cpu")
trainer = PPOTrainer(cfg)
metrics = trainer.train_iteration(random.Random(0))  # {"mean_reward", "solve_rate", ...}
trainer.save("checkpoint.pt")
```

Or load `TrainConfig` straight from `configs/default.yaml` (this is what
`experiments/train_ppo.py` does):

```python
import yaml, pathlib
from crpsp.ppo import TrainConfig
cfg = TrainConfig.from_yaml(yaml.safe_load(pathlib.Path("configs/default.yaml").read_text()))
```

## Module reference

### `instance.py` — problem data

- **`Instance`** — frozen dataclass. `yard`/`stowage`: tuples of tuples of container ids, bottom→top per stack. `stowage[0]` is the vessel stack *furthest from shore* (Eq. 23 ordering). `t_y` is the shared tier cap for **both** yard and vessel stacks. `mode` is `"crpsp"` (stowage-plan driven, the paper's problem) or `"crp"` (priority-driven, the standard-CRP extension). Properties `n`, `s_y`, `s_v` are derived, not stored.
- **`generate_instance(n, s_y, s_v, t_y, rng) -> Instance`** — Section 4.2 random generator. Raises `ValueError` if `n` doesn't fit the yard *or* the vessel at `t_y` tiers each (both are capacity-checked — see `REPRODUCTION_NOTES.md` #20 for why the vessel check matters). Deterministic given `rng`.
- **`slot_map(inst) -> dict[container_id, (vessel_stack, tier)]`** — every CRPSP solver needs this to know where a container is headed; computed once per instance/episode, not per step.
- **`load_caserta(path) -> Instance`** — standard-CRP loader (extension, not in the paper). Returns `mode="crp"` with `priorities = (1..n)`.

### `lower_bound.py` — Eq. (27)–(30)

- **`must_precede_pairs(inst) -> frozenset[(a, b)]`** — the set of "a before b" ordering constraints: same-vessel-stack ordering + Eq-23 shoreline ordering (CRPSP mode), or priority ordering (CRP mode). Compute **once per instance** (it's static for the whole episode) and pass into every `lower_bound()` call — this is why `env.py`/`astar.py`/`heuristic.py` all cache it in a local variable rather than recomputing it per state.
- **`lower_bound(yard, precede) -> int`** — `h(n)` = remaining containers + number of blocking 2-cycles among *directly adjacent* yard pairs. Admissible (never overestimates true remaining ops) — this is what makes A\* optimal and the RL reward shaping in `env.py` valid.

### `transfer.py` — the one transition rule, shared everywhere

- **`closure_crpsp(yard, vessel, slot_of) -> int`** — mutates `yard`/`vessel` in place, repeatedly moving any top-of-yard-stack container onto its designated vessel slot while Eq. (23) holds. Returns the number of transfers performed.
- **`closure_crp(yard, next_priority) -> (int, int)`** — mutates `yard` in place, repeatedly popping any top-of-stack container matching the current retrieval priority. Returns `(retrievals, new_next_priority)`.

If you ever need to change *when* a container may leave the yard (e.g. a new constraint), this is the only place to change it — `env.py`, `astar.py`, and `heuristic.py` all call into these two functions rather than each having their own copy.

### `env.py` — the RL environment

- **`CRPSPEnv(reward_mode="designed", terminal_bonus=10.0, max_steps=50, restricted=False)`**
  - `reset(instance) -> (obs, mask)` — `obs` is the state matrix Θ (`np.float32`, shape `(s_y, t_y)`, container id normalized by `n`, `0` = empty); `mask` is a boolean vector over `env.action_pairs` (all `(source, dest)` stack pairs, `source != dest`), `True` where the move is currently legal.
  - `step(action: int) -> (obs, mask, reward, done, truncated, info)` — `action` indexes `env.action_pairs`. Reward: Eq. (36) in `"designed"` mode (`-1 + h(s_t) - h(s_{t+1})`, plus `terminal_bonus` on completion), flat `-1` in `"simple"` mode (the Section 5.5 ablation). `info` has `n_relocations`, `n_transfers`, `total_ops`.
  - `restricted=True` (CRP mode only) additionally masks sources down to whichever stack currently holds the next retrieval target (restricted-CRP variant).
  - After `reset`/`step`, `env.yard`, `env.vessel`, `env.done`, `env.n_relocations`, `env.n_transfers`, `env.t` are the live episode state if you need to inspect it directly (e.g. for a custom reward or logging).
  - `CRPSPEnv.encode(yard, s_y, t_y, n) -> np.ndarray` is a `@staticmethod` — reuse it if you build states outside a live episode (e.g. in `ensemble.py`, which encodes A\*-trajectory states directly).

### `astar.py` — exact search

- **`AStarResult`**: `total_ops`, `relocations`, `optimal` (`False` if the time/node budget ran out — **check this before trusting `total_ops`**, which is `-1` when not optimal), `nodes_expanded`, `trajectory`.
- **`solve_astar(inst, time_limit_s=None, node_limit=None) -> AStarResult`** — `trajectory` is a list of `(yard_before, vessel_or_priority_state_before, (source, dest))` along the optimal path, one entry per relocation; replaying these actions through a fresh `CRPSPEnv` reproduces the same operation count exactly (see `tests/test_astar.py::test_trajectory_replays_to_solution`). This is what `ensemble.py::build_stacking_dataset` uses to get ground-truth `(state, action)` training pairs.
- Dedups search states by `(sorted(yard), vessel_or_priority_state)` — yard stacks are physically interchangeable, so this canonicalization prunes the search space without changing the optimal objective value (verified by the trajectory-replay test).

### `mip_cpsat.py` — exact via CP-SAT

- **`solve_mip(inst, horizon=None, time_limit_s=120.0, w1=10, w2=1) -> MipResult`** — `status` is `"OPTIMAL"`/`"FEASIBLE"`/`"INFEASIBLE"`/`"UNKNOWN"`; `target_reached` tells you whether the returned plan actually matches the stowage plan (mirrors the paper's two-part objective, Eq. 1). You must supply a `horizon` large enough to reach the target — `experiments/table2_mip.py` uses `solve_heuristic(inst).total_ops + 2` as a cheap, always-sufficient upper bound.

### `heuristic.py` — fast baseline

- **`solve_heuristic(inst, max_ops=None) -> HeuristicResult`** — Section 5.7's forward-looking greedy rule. Always terminates (bounded by `max_ops`, default `10*n`); `solved=False` if it hits that cap without finishing (shouldn't happen in practice — see the test suite).

### `models.py` — networks (pure PyTorch, no environment/training logic)

- **`Actor(s_y, t_y, hidden_dim=128, use_attention=True).forward(obs, mask) -> logits`** — `obs`/`mask` are batched (`(B, s_y, t_y)` / `(B, s_y*(s_y-1))`). Masked-out actions get logit `-1e9` (i.e. ~0 probability after softmax, but still autograd-safe, unlike `-inf`).
- **`Critic(s_y, t_y, hidden_dim=128, use_attention=True).forward(obs) -> value`** — shape `(B,)`.
- **`use_attention=False`** swaps the `RowSelfAttention` block for a flatten-only trunk — this is the Section 5.4 ablation switch; both variants expose the exact same `forward` signature, so nothing downstream needs to know which one it's using.
- If you're changing the architecture (bigger attention, a decoder, etc.), **keep the `forward(obs, mask) -> logits` / `forward(obs) -> value` contract** — `ppo.py` and `ensemble.py` only call through that interface.

### `ppo.py` — training

- **`compute_gae(rewards, values, next_values, dones, gamma, lam) -> (advantages, td_targets)`** — pure tensor math, no environment dependency; easy to unit-test in isolation (see `tests/test_ppo.py::test_gae_hand_computed`).
- **`TrainConfig`** — one field per entry in `configs/default.yaml` (flattened `env` + `model` + `train` sections). `TrainConfig.from_yaml(cfg_dict)` is the standard constructor; the plain dataclass constructor is mainly for tests that need a tiny config.
- **`PPOTrainer(cfg)`**
  - `train_iteration(rng) -> dict` — **one full outer loop of Algorithm 1**: `cfg.instances_per_iter` fresh episodes, GAE, one shuffled mini-batch pass, clipped actor update + MSE critic update. Returns `mean_reward`, `solve_rate`, `mean_relocations`, `actor_loss`, `critic_loss`.
  - `save(path)` / `load(path)` — checkpoints both networks plus the config (so `experiments/table4_ppo_vs_astar.py` can reconstruct a compatible `PPOTrainer` from just the checkpoint file).
- **`convergence_iteration(history, window=20) -> int | None`** — the paper doesn't define "converged"; this is the operational definition used for Tables 5–7 (see `REPRODUCTION_NOTES.md` #10). `history` is a list of the dicts returned by `train_iteration`.

### `ensemble.py` — Voting & Stacking

- **`vote(actors, obs, mask, device) -> int`** — Eq. (40): majority of per-model argmax actions, ties broken by summed probability.
- **`build_stacking_dataset(actors, instances, device, time_limit_s=None) -> (X, y)`** — runs `solve_astar` on each instance (pass `time_limit_s` to bound this at paper scale — see `experiments/table8_ensemble.py` for how the smoke/full split threads this through); skips instances A\* doesn't prove optimal within the budget. `X` rows are the concatenation of every model's action-probability vector at each optimal-trajectory state; `y` is the optimal action index.
- **`train_stacking(X, y, n_actions) -> StackingPolicy`** — trains an `xgboost.XGBClassifier` (labels are re-indexed internally since XGBoost needs contiguous classes; `StackingPolicy.predict` maps back and additionally restricts to currently-valid actions via the mask).

### `evaluate.py` — generic, solver-agnostic

- **`gap(alg_ops, opt_ops) -> float`** = `(alg_ops - opt_ops) / opt_ops`.
- **`rollout_policy(select_action, inst, env_kwargs) -> dict`** — drives any `select_action(obs, mask) -> int` callable (a trained actor, `vote(...)`, a `StackingPolicy`, even a random policy) through a fresh `CRPSPEnv(**env_kwargs)` and reports `total_ops`/`relocations`/`solved`/`seconds`. This is the one function every `experiments/table*.py` script calls to turn "a policy" into "a row in a results CSV" — write new policies to this same `(obs, mask) -> int` shape and you get the same evaluation harness for free.

## The `mode` dispatch pattern

Four modules (`lower_bound.py`, `transfer.py`, `env.py`, `astar.py`) branch on
`Instance.mode` (`"crpsp"` vs `"crp"`). If you add a third problem variant,
follow the same pattern rather than adding new booleans/flags:

1. Give `Instance` whatever extra field the new mode needs (like `priorities` for `"crp"`).
2. Add a `closure_<mode>()` function to `transfer.py`.
3. Add the branch to `must_precede_pairs()` in `lower_bound.py`.
4. Dispatch on `inst.mode` in `env.py::_transfer_closure` and `astar.py::run_closure`.

Everything else (`models.py`, `ppo.py`, `ensemble.py`, `evaluate.py`) is
already mode-agnostic — they only see `obs`/`mask`/`Instance.s_y`/`t_y`, never
`mode` directly.
