# CRPSP Reproduction (Wang et al. 2025, TR-E) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithfully implement the full CRPSP framework of Wang et al. (2025): instance generator, cycle lower bound, environment, A*, CP-SAT MIP, greedy heuristic, PPO with self-attention actor-critic, ensembles (Voting/Stacking), evaluation scripts — verified by tests only.

**Architecture:** Pure-Python package `crpsp/` with one module per paper component. PPO is custom PyTorch implementing the paper's Algorithm 1 verbatim (no RL framework). All solvers share the `Instance` dataclass and the lower-bound module. Experiments are thin CLI scripts mapping 1:1 to paper Tables 2–9.

**Tech Stack:** Python 3.11/3.12 venv, PyTorch (CPU wheels, device-agnostic code), OR-Tools CP-SAT, XGBoost, NumPy, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-07-04-crpsp-reproduction-design.md`
**Paper:** `Paper/1-s2.0-S1366554525000894-main (1).pdf`

## Global Constraints

- **NO training runs, NO heavy compute on this machine.** Only unit tests and smoke tests (each < ~60 s CPU). Experiment scripts default to `--smoke`; full scale requires explicit `--full` (meant for the user's server). Never invoke `--full` yourself.
- Hyperparameters exactly per paper Table 10: actor/critic lr `5e-4`, γ `0.4`, batch `10`, hidden dim `128`, GAE λ `0.9`, clip ε `0.15`, max steps per instance `50`, instances/iteration `10`.
- Reward Eq (36): `r = −1 + h(s_t) − h(s_{t+1})`; simple variant `r = −1` selectable (Section 5.5 ablation).
- Objective weights MIP: ω1:ω2 = 10:1.
- Every ambiguity resolution goes into `REPRODUCTION_NOTES.md` in the same commit.
- Vessel stack index 0 = furthest from shoreline; Eq (23): heights non-increasing with index, enforced at every time step.
- Work happens inside `c:/Users/X1/Project/CRP_Master`; a fresh git repo is created there (Task 1) — do not commit to the accidental home-directory repo.
- All code, comments, and docs in English.

---

### Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `configs/default.yaml`, `REPRODUCTION_NOTES.md`, `crpsp/__init__.py`, `tests/test_sanity.py`

**Interfaces:**
- Produces: importable package `crpsp`, config file consumed by all later tasks via `yaml.safe_load`.

- [ ] **Step 1: Init isolated git repo and venv**

```bash
cd /c/Users/X1/Project/CRP_Master
git init -b master
py -0p   # list available interpreters; pick 3.12 or 3.11
py -3.12 -m venv .venv || py -3.11 -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install ortools xgboost numpy pyyaml pytest
```
Expected: venv created; `import torch, ortools, xgboost` all succeed. If no 3.11/3.12 is available, install via `winget install Python.Python.3.12` first.

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
results/
checkpoints/
baseline_actor_critic/
mdpi_page.html
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "crpsp"
version = "0.1.0"
description = "Faithful reproduction of Wang et al. (2025) CRPSP hybrid learning algorithms"
requires-python = ">=3.11"
dependencies = ["torch", "ortools", "xgboost", "numpy", "pyyaml"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: excluded by default; run with -m slow"]
addopts = "-m 'not slow'"
```

- [ ] **Step 4: Write `configs/default.yaml`** (Table 10 + Section 5 instance params)

```yaml
env:
  n: 15
  s_y: 5
  s_v: 5
  t_y: 5
  reward_mode: designed      # designed | simple   (Section 5.5)
  terminal_bonus: 10.0       # paper: "large positive reward", value unspecified — see notes
  max_steps: 50              # Table 10: max attempt times T
model:
  hidden_dim: 128            # Table 10
  use_attention: true        # false => Section 5.4 no-attention ablation
train:
  lr: 5.0e-4                 # Table 10, actor and critic
  gamma: 0.4                 # Table 10
  gae_lambda: 0.9            # Table 10
  clip_eps: 0.15             # Table 10
  batch_size: 10             # Table 10
  instances_per_iter: 10     # Table 10: M
  iterations: 1000
  ppo_epochs: 1              # Algorithm 1 shows a single pass — see notes
  seed: 42
  device: auto               # auto => cuda if available else cpu
```

- [ ] **Step 5: Write `REPRODUCTION_NOTES.md` skeleton**

```markdown
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
10. **[ppo] Convergence iteration (Tables 5–7)** — undefined in paper → we define: first iteration where rolling solve-rate over a 20-iteration window is 100% and mean relocations stop improving (tolerance 0); capped at 1000.
11. **[evaluate] Gap** = (total_ops_alg − total_ops_opt) / total_ops_opt, consistent with Table 4 values (5.00% ≈ 1/20, 9.09% = 2/22).
12. **[heuristic] "Minimally impacts the loading process"** — destination chosen minimizing (# containers already in dest that must precede the moved container, resulting height).
13. **[env] Invalid actions** — paper doesn't discuss masking; we mask at sampling (source non-empty, dest not full, s≠d) and the env raises on invalid input.
```

- [ ] **Step 6: Write `crpsp/__init__.py`** (empty) **and failing sanity test `tests/test_sanity.py`**

```python
import yaml, pathlib

def test_config_loads_table10_values():
    cfg = yaml.safe_load(pathlib.Path("configs/default.yaml").read_text())
    assert cfg["train"]["lr"] == 5.0e-4
    assert cfg["train"]["gamma"] == 0.4
    assert cfg["train"]["batch_size"] == 10
    assert cfg["model"]["hidden_dim"] == 128
    assert cfg["train"]["gae_lambda"] == 0.9
    assert cfg["train"]["clip_eps"] == 0.15
    assert cfg["env"]["max_steps"] == 50
    assert cfg["train"]["instances_per_iter"] == 10
```

- [ ] **Step 7: Run test**

Run: `.venv/Scripts/python -m pytest tests/test_sanity.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "chore: scaffold crpsp package, config (Table 10), reproduction notes"
```

---

### Task 2: Instance dataclass + generator (paper Section 4.2)

**Files:**
- Create: `crpsp/instance.py`
- Test: `tests/test_instance.py`

**Interfaces:**
- Produces:
  - `Instance(yard, stowage, t_y, mode="crpsp", priorities=())` frozen dataclass; `yard: tuple[tuple[int,...],...]` bottom→top ids per yard stack; `stowage` likewise per vessel stack, index 0 = furthest from shore; properties `n`, `s_y`, `s_v`.
  - `generate_instance(n: int, s_y: int, s_v: int, t_y: int, rng: random.Random) -> Instance`
  - `slot_map(inst) -> dict[int, tuple[int,int]]` mapping container id → (vessel stack, tier).

- [ ] **Step 1: Write failing tests `tests/test_instance.py`**

```python
import random
import pytest
from crpsp.instance import Instance, generate_instance, slot_map

def test_generate_basic_properties():
    rng = random.Random(0)
    inst = generate_instance(n=15, s_y=5, s_v=5, t_y=5, rng=rng)
    assert inst.n == 15 and inst.s_y == 5 and inst.s_v == 5
    ids = sorted(c for st in inst.yard for c in st)
    assert ids == list(range(1, 16))                    # every container exactly once in yard
    assert all(len(st) <= inst.t_y for st in inst.yard) # yard height limit
    assert sorted(c for st in inst.stowage for c in st) == list(range(1, 16))

def test_stowage_heights_nonincreasing_toward_shore():
    rng = random.Random(1)
    for _ in range(20):
        inst = generate_instance(12, 4, 4, 5, rng)
        hs = [len(st) for st in inst.stowage]           # index 0 = furthest from shore
        assert all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))  # Eq 23 final profile

def test_labeling_shore_first_bottom_up():
    rng = random.Random(2)
    inst = generate_instance(6, 3, 3, 5, rng)
    labels = []
    for s in range(inst.s_v - 1, -1, -1):               # shore -> far
        labels.extend(inst.stowage[s])                  # bottom -> top
    assert labels == list(range(1, 7))

def test_slot_map_roundtrip():
    rng = random.Random(3)
    inst = generate_instance(10, 4, 4, 5, rng)
    sm = slot_map(inst)
    for vs, stack in enumerate(inst.stowage):
        for vt, c in enumerate(stack):
            assert sm[c] == (vs, vt)

def test_capacity_error():
    with pytest.raises(ValueError):
        generate_instance(n=30, s_y=2, t_y=5, s_v=3, rng=random.Random(0))

def test_seeded_determinism():
    a = generate_instance(15, 5, 5, 5, random.Random(7))
    b = generate_instance(15, 5, 5, 5, random.Random(7))
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_instance.py -v`
Expected: FAIL — `ModuleNotFoundError: crpsp.instance`.

- [ ] **Step 3: Implement `crpsp/instance.py`**

```python
"""Instance definition and random generator (paper Section 2 and 4.2)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    yard: tuple[tuple[int, ...], ...]     # bottom->top container ids per yard stack
    stowage: tuple[tuple[int, ...], ...]  # bottom->top ids per vessel stack; 0 = furthest from shore
    t_y: int                              # yard max height
    mode: str = "crpsp"                   # "crpsp" | "crp"
    priorities: tuple[int, ...] = ()      # crp mode only: ids in retrieval order

    @property
    def n(self) -> int:
        return sum(len(s) for s in self.yard)

    @property
    def s_y(self) -> int:
        return len(self.yard)

    @property
    def s_v(self) -> int:
        return len(self.stowage)


def generate_instance(n: int, s_y: int, s_v: int, t_y: int, rng: random.Random) -> Instance:
    """Random CRPSP instance per Section 4.2.

    Stowage height profile is non-increasing toward shore so Eq (23) is
    satisfiable; labels 1..N assigned shore->far, bottom->top (notes #2, #3).
    """
    if n > s_y * t_y:
        raise ValueError(f"yard capacity {s_y * t_y} cannot hold {n} containers")
    counts = [0] * s_v
    for _ in range(n):
        counts[rng.randrange(s_v)] += 1
    counts.sort(reverse=True)             # index 0 (furthest) tallest
    stowage: list[list[int]] = [[0] * h for h in counts]
    label = 1
    for s in range(s_v - 1, -1, -1):      # shore side first
        for k in range(counts[s]):        # bottom up
            stowage[s][k] = label
            label += 1
    ids = list(range(1, n + 1))
    rng.shuffle(ids)
    yard: list[list[int]] = [[] for _ in range(s_y)]
    for c in ids:
        open_stacks = [i for i in range(s_y) if len(yard[i]) < t_y]
        yard[rng.choice(open_stacks)].append(c)
    return Instance(
        yard=tuple(tuple(s) for s in yard),
        stowage=tuple(tuple(s) for s in stowage),
        t_y=t_y,
    )


def slot_map(inst: Instance) -> dict[int, tuple[int, int]]:
    """Container id -> (vessel stack index, tier index)."""
    return {c: (vs, vt) for vs, stack in enumerate(inst.stowage) for vt, c in enumerate(stack)}
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_instance.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crpsp/instance.py tests/test_instance.py
git commit -m "feat: CRPSP instance dataclass and Section-4.2 generator"
```

---

### Task 3: Lower bound h(n) (paper Section 3.2, Eq 27–30)

**Files:**
- Create: `crpsp/lower_bound.py`
- Test: `tests/test_lower_bound.py`

**Interfaces:**
- Consumes: `Instance`, `slot_map` from Task 2.
- Produces:
  - `must_precede_pairs(inst: Instance) -> frozenset[tuple[int,int]]` — pair (a,b): a must be placed on the vessel before b (same-stack below→above; Eq-23 shoreline far-tier-k→near-tier-k; CRP mode: priority order).
  - `lower_bound(yard: Sequence[Sequence[int]], precede: frozenset) -> int` — `h(n) = N_L + #2-cycles` (Eq 29; ½ factor absorbed by counting unordered pairs once).

- [ ] **Step 1: Write failing tests `tests/test_lower_bound.py`**

```python
import random
from crpsp.instance import Instance, generate_instance
from crpsp.lower_bound import must_precede_pairs, lower_bound

def _inst(yard, stowage, t_y=4):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)

def test_same_stack_precedence():
    inst = _inst([[1, 2]], [[1, 2]])       # vessel stack: 1 below 2
    p = must_precede_pairs(inst)
    assert (1, 2) in p and (2, 1) not in p

def test_shoreline_precedence_eq23():
    # stack 0 = far holds container 2; stack 1 = shore holds container 1.
    inst = _inst([[1], [2]], [[2], [1]])
    p = must_precede_pairs(inst)
    assert (2, 1) in p                      # far tier-1 before near tier-1 (Eq 23)
    assert (1, 2) not in p

def test_lb_counts_transfers_only_when_no_blocking():
    inst = _inst([[2, 1]], [[1, 2]])        # yard: 1 on top of 2; no cycle
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) == 2   # N_L=2, zero cycles

def test_lb_detects_forced_relocation():
    inst = _inst([[1, 2]], [[1, 2]])        # 2 sits on 1, but 1 must go first
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) == 3   # 2 transfers + 1 forced relocation

def test_lb_direct_adjacency_only():
    # yard stack bottom->top: 1,3,2 ; stowage single stack (1,2,3).
    # 2 directly atop 3: precede(3,2)? no wait — (2 above 3) and 3 before 2? priority: 1<2<3 in stack (1,2,3): precede has (1,2),(1,3),(2,3).
    # pairs: (3 atop-of? ) yard adjacency: (3 on 1) -> precede(1,3) => cycle; (2 on 3) -> precede(3,2)? no, (2,3) in precede means 2 before 3; adjacency 2-on-3 with (3,2) required => not present => no cycle.
    inst = _inst([[1, 3, 2]], [[1, 2, 3]])
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) == 3 + 1   # N_L=3 + one cycle (3 over 1)

def test_lb_admissible_vs_bruteforce_small():
    # exhaustively verified case: optimal ops = 4 (relocate 2, then transfers 1,2... )
    # yard: [[1,2],[ ]], stowage [[1,2]] -> optimal: move 2 to stack 1 (1 reloc), transfer 1, transfer 2 = 3 ops
    inst = _inst([[1, 2], []], [[1, 2]])
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_lower_bound.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `crpsp/lower_bound.py`**

```python
"""A* lower bound h(n) via 2-cycle counting (paper Eq 27-30, Tanaka & Voss 2019)."""
from __future__ import annotations

from typing import Sequence

from .instance import Instance


def must_precede_pairs(inst: Instance) -> frozenset[tuple[int, int]]:
    """(a, b) in result <=> a must be placed on the vessel before b.

    CRPSP: same-vessel-stack below->above, plus Eq-23 shoreline precedence
    (tier-k of a farther stack before tier-k of any nearer stack; notes #5).
    CRP mode: retrieval priority order.
    """
    pairs: set[tuple[int, int]] = set()
    if inst.mode == "crp":
        order = inst.priorities
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                pairs.add((order[i], order[j]))
        return frozenset(pairs)
    for stack in inst.stowage:
        for lo in range(len(stack)):
            for hi in range(lo + 1, len(stack)):
                pairs.add((stack[lo], stack[hi]))
    for far in range(inst.s_v):
        for near in range(far + 1, inst.s_v):
            depth = min(len(inst.stowage[far]), len(inst.stowage[near]))
            for k in range(depth):
                pairs.add((inst.stowage[far][k], inst.stowage[near][k]))
    return frozenset(pairs)


def lower_bound(yard: Sequence[Sequence[int]], precede: frozenset[tuple[int, int]]) -> int:
    """h(n) = remaining containers + number of 2-cycles (Eq 29).

    Yard edges use DIRECT adjacency (notes #4): container `above` directly on
    `below` forms a cycle iff `below` must be placed before `above` — forcing
    one relocation of `above`. Movers are distinct, so the count is admissible.
    """
    n_l = sum(len(s) for s in yard)
    cycles = 0
    for stack in yard:
        for k in range(1, len(stack)):
            above, below = stack[k], stack[k - 1]
            if (below, above) in precede:
                cycles += 1
    return n_l + cycles
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_lower_bound.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crpsp/lower_bound.py tests/test_lower_bound.py
git commit -m "feat: cycle-based admissible lower bound (Eq 27-30)"
```

---

### Task 4: Environment (paper Section 4.2)

**Files:**
- Create: `crpsp/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Instance`, `slot_map`, `must_precede_pairs`, `lower_bound`.
- Produces class `CRPSPEnv`:
  - `__init__(reward_mode="designed", terminal_bonus=10.0, max_steps=50)`
  - `reset(instance) -> (obs: np.ndarray float32 (s_y,t_y), mask: np.ndarray bool (s_y*(s_y-1),))`
  - `step(action: int) -> (obs, mask, reward: float, done: bool, truncated: bool, info: dict)` — `info` has `n_relocations`, `n_transfers`, `total_ops`.
  - `action_pairs: list[tuple[int,int]]` ordered `[(s,d) for s in range(s_y) for d in range(s_y) if d != s]`.
  - Static `encode(yard, s_y, t_y, n) -> np.ndarray` (id/n normalization, 0 = empty).

- [ ] **Step 1: Write failing tests `tests/test_env.py`**

```python
import numpy as np
import pytest
from crpsp.instance import Instance
from crpsp.env import CRPSPEnv

def _inst(yard, stowage, t_y=3):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)

def test_reset_runs_transfer_closure():
    # yard top of stack0 is 1 -> immediately transferable; then 2 -> episode done at reset
    inst = _inst([[2, 1], []], [[1, 2]])
    env = CRPSPEnv()
    obs, mask = env.reset(inst)
    assert env.done and env.n_transfers == 2 and env.n_relocations == 0

def test_step_relocation_then_auto_transfer():
    inst = _inst([[1, 2], []], [[1, 2]])      # 2 blocks 1
    env = CRPSPEnv(reward_mode="simple")
    obs, mask = env.reset(inst)
    assert not env.done
    a = env.action_pairs.index((0, 1))        # move 2 from stack0 to stack1
    obs, mask, r, done, trunc, info = env.step(a)
    assert done and info["n_relocations"] == 1 and info["n_transfers"] == 2
    assert info["total_ops"] == 3
    assert r == -1.0                          # simple reward: exactly -1 per step

def test_designed_reward_eq36():
    inst = _inst([[1, 2], []], [[1, 2]])
    env = CRPSPEnv(reward_mode="designed", terminal_bonus=10.0)
    env.reset(inst)
    # h(s_t): N_L=2 + 1 cycle = 3 ; after step: yard empty -> h=0
    a = env.action_pairs.index((0, 1))
    _, _, r, done, _, _ = env.step(a)
    assert done
    assert r == pytest.approx(-1 + 3 - 0 + 10.0)

def test_eq23_blocks_transfer():
    # container 1 designated to SHORE stack; far stack container 2 not yet placed
    # stowage: stack0(far)=[2], stack1(shore)=[1]
    inst = _inst([[1], [2], []], [[2], [1]])
    env = CRPSPEnv()
    env.reset(inst)
    # only container 2 may transfer first; 1 must wait even though on top
    assert env.vessel == [[2], [1]] and env.done  # closure: 2 then 1 both transfer

def test_eq23_never_violated_during_random_play():
    import random
    from crpsp.instance import generate_instance
    rng = random.Random(0)
    env = CRPSPEnv()
    for _ in range(10):
        inst = generate_instance(10, 4, 4, 5, rng)
        obs, mask = env.reset(inst)
        steps = 0
        while not env.done and steps < 50:
            valid = np.flatnonzero(mask)
            obs, mask, r, done, trunc, _ = env.step(int(rng.choice(list(valid))))
            hs = [len(s) for s in env.vessel]
            assert all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))
            # container conservation
            total = sum(len(s) for s in env.yard) + sum(len(s) for s in env.vessel)
            assert total == inst.n
            steps += 1

def test_mask_correctness():
    inst = _inst([[1, 2], [], [3]], [[1, 2, 3]], t_y=2)
    env = CRPSPEnv()
    obs, mask = env.reset(inst)
    for idx, (s, d) in enumerate(env.action_pairs):
        expected = len(env.yard[s]) > 0 and len(env.yard[d]) < inst.t_y
        assert mask[idx] == expected

def test_invalid_action_raises():
    inst = _inst([[1, 2], []], [[1, 2]])
    env = CRPSPEnv()
    env.reset(inst)
    bad = env.action_pairs.index((1, 0))      # stack1 empty
    with pytest.raises(ValueError):
        env.step(bad)

def test_truncation_at_max_steps():
    inst = _inst([[1, 2], [], []], [[1, 2]], t_y=3)
    env = CRPSPEnv(max_steps=2)
    obs, mask = env.reset(inst)
    # shuffle 2 back and forth between stacks 1 and 2 (never solving)
    a1 = env.action_pairs.index((0, 1))
    env.step(a1)  # this actually solves it; rebuild: use stowage [[2,1]] wait --
```

  Replace the last test with a genuinely non-solving loop:

```python
def test_truncation_at_max_steps():
    # 1 must precede 2 (same vessel stack), 2 buried under 1 => moving 1 back and forth never solves
    inst = _inst([[2, 1], [], []], [[2, 1]], t_y=3)  # vessel: 2 below 1... then top-of-yard 1 can't go until 2
    env = CRPSPEnv(max_steps=2)
    obs, mask = env.reset(inst)
    assert not env.done
    a = env.action_pairs.index((1, 2))
    # move 1: stack0->stack1, then stack1->stack2 : still unsolved -> truncated
    env.step(env.action_pairs.index((0, 1)))
    obs, mask, r, done, trunc, _ = env.step(env.action_pairs.index((1, 2)))
    assert trunc and not done
```

  Note: with yard `[[2,1]]` and stowage `[[2,1]]`, at reset nothing transfers (top 1 is not 2), moving 1 away lets 2 transfer, then 1 transfers — solved in one step. To keep an unsolvable-in-2-steps case, use stowage `[[1, 2]]` and yard `[[1, 2]]` with only full/blocked destinations... **Executor:** implement `test_truncation_at_max_steps` by giving `max_steps=1` and an instance needing ≥2 relocations, e.g. yard `[[1, 3, 2], []]`, stowage `[[1, 2, 3]]` (needs relocating 2 and 3): one step cannot finish → truncated.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_env.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `crpsp/env.py`**

```python
"""CRPSP environment: relocation actions, automatic transfer closure (Section 4.2)."""
from __future__ import annotations

import numpy as np

from .instance import Instance, slot_map
from .lower_bound import lower_bound, must_precede_pairs


class CRPSPEnv:
    def __init__(self, reward_mode: str = "designed", terminal_bonus: float = 10.0,
                 max_steps: int = 50):
        assert reward_mode in ("designed", "simple")
        self.reward_mode = reward_mode
        self.terminal_bonus = terminal_bonus
        self.max_steps = max_steps

    # ------------------------------------------------------------------ setup
    def reset(self, instance: Instance):
        self.inst = instance
        self.yard: list[list[int]] = [list(s) for s in instance.yard]
        self.vessel: list[list[int]] = [[] for _ in range(instance.s_v)]
        self.slot_of = slot_map(instance)
        self.precede = must_precede_pairs(instance)
        self.action_pairs = [(s, d) for s in range(instance.s_y)
                             for d in range(instance.s_y) if d != s]
        self.n_relocations = 0
        self.n_transfers = 0
        self.t = 0
        self._transfer_closure()
        self.done = self._all_placed()
        return self._obs(), self._mask()

    # ------------------------------------------------------------------ core
    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        s, d = self.action_pairs[action]
        if not self.yard[s] or len(self.yard[d]) >= self.inst.t_y:
            raise ValueError(f"invalid action {(s, d)}")
        h_before = lower_bound(self.yard, self.precede)
        self.yard[d].append(self.yard[s].pop())
        self.n_relocations += 1
        self._transfer_closure()
        self.t += 1
        self.done = self._all_placed()
        truncated = (not self.done) and self.t >= self.max_steps
        if self.reward_mode == "designed":
            h_after = lower_bound(self.yard, self.precede)
            reward = -1.0 + h_before - h_after          # Eq (36)
            if self.done:
                reward += self.terminal_bonus           # notes #1
        else:
            reward = -1.0                               # Section 5.5 baseline
        info = {"n_relocations": self.n_relocations, "n_transfers": self.n_transfers,
                "total_ops": self.n_relocations + self.n_transfers}
        return self._obs(), self._mask(), reward, self.done, truncated, info

    # ------------------------------------------------------------- internals
    def _all_placed(self) -> bool:
        return all(not s for s in self.yard)

    def _height_ok(self, vs: int) -> bool:
        """Eq (23): after adding to vessel stack vs, nearer-shore stacks stay <= farther ones."""
        h_new = len(self.vessel[vs]) + 1
        return all(len(self.vessel[j]) >= h_new for j in range(vs))

    def _transfer_closure(self) -> None:
        moved = True
        while moved:
            moved = False
            for stack in self.yard:
                if not stack:
                    continue
                c = stack[-1]
                vs, vt = self.slot_of[c]
                if len(self.vessel[vs]) == vt and self._height_ok(vs):
                    stack.pop()
                    self.vessel[vs].append(c)
                    self.n_transfers += 1
                    moved = True

    def _obs(self) -> np.ndarray:
        return self.encode(self.yard, self.inst.s_y, self.inst.t_y, self.inst.n)

    def _mask(self) -> np.ndarray:
        m = np.zeros(len(self.action_pairs), dtype=bool)
        for i, (s, d) in enumerate(self.action_pairs):
            m[i] = bool(self.yard[s]) and len(self.yard[d]) < self.inst.t_y
        return m

    @staticmethod
    def encode(yard, s_y: int, t_y: int, n: int) -> np.ndarray:
        obs = np.zeros((s_y, t_y), dtype=np.float32)
        for i, stack in enumerate(yard):
            for k, c in enumerate(stack):
                obs[i, k] = c / n
        return obs
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_env.py -v`
Expected: all PASS (after fixing the truncation test per the executor note).

- [ ] **Step 5: Commit**

```bash
git add crpsp/env.py tests/test_env.py
git commit -m "feat: CRPSP environment with transfer closure, Eq-36 reward, action masking"
```

---

### Task 5: A* solver (paper Section 3)

**Files:**
- Create: `crpsp/astar.py`
- Test: `tests/test_astar.py`

**Interfaces:**
- Consumes: `Instance`, `slot_map`, `must_precede_pairs`, `lower_bound`.
- Produces:
  - `solve_astar(inst, time_limit_s=None, node_limit=None) -> AStarResult`
  - `AStarResult` dataclass: `total_ops: int`, `relocations: int`, `optimal: bool`, `nodes_expanded: int`, `trajectory: list[tuple[tuple[tuple[int,...],...], tuple[int,...], tuple[int,int]]]` — (yard state, vessel heights, relocation (s,d)) pairs along the optimal path, exactly the (state, optimal action) pairs Stacking needs (Eq 41).

- [ ] **Step 1: Write failing tests `tests/test_astar.py`**

```python
import random
import pytest
from crpsp.instance import Instance, generate_instance
from crpsp.astar import solve_astar
from crpsp.lower_bound import must_precede_pairs, lower_bound

def _inst(yard, stowage, t_y=3):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)

def test_zero_relocation_instance():
    res = solve_astar(_inst([[2, 1], []], [[1, 2]]))
    assert res.relocations == 0 and res.total_ops == 2 and res.optimal

def test_one_forced_relocation():
    res = solve_astar(_inst([[1, 2], []], [[1, 2]]))
    assert res.relocations == 1 and res.total_ops == 3
    assert len(res.trajectory) == 1
    (yard, vheights, action) = res.trajectory[0]
    assert action[0] == 0                      # must move off stack 0

def test_lb_is_admissible_on_random_instances():
    rng = random.Random(0)
    for _ in range(30):
        inst = generate_instance(8, 3, 3, 4, rng)
        res = solve_astar(inst)
        p = must_precede_pairs(inst)
        assert lower_bound(inst.yard, p) <= res.total_ops
        assert res.optimal

def test_deterministic_and_solves_table3_smallest():
    rng = random.Random(1)
    inst = generate_instance(12, 4, 4, 5, rng)
    res = solve_astar(inst, time_limit_s=30)
    assert res.optimal and res.total_ops >= 12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_astar.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `crpsp/astar.py`**

```python
"""A* for the CRPSP (paper Section 3). g = ops so far, h = Eq-29 lower bound."""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

from .instance import Instance, slot_map
from .lower_bound import lower_bound, must_precede_pairs

Yard = tuple[tuple[int, ...], ...]
VHeights = tuple[int, ...]


@dataclass
class AStarResult:
    total_ops: int
    relocations: int
    optimal: bool
    nodes_expanded: int
    trajectory: list[tuple[Yard, VHeights, tuple[int, int]]] = field(default_factory=list)


def _closure(yard: list[list[int]], vessel: list[list[int]], slot_of, s_v: int) -> int:
    """Execute all feasible transfers; return how many were done (Eq 23 respected)."""
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


def solve_astar(inst: Instance, time_limit_s: float | None = None,
                node_limit: int | None = None) -> AStarResult:
    slot_of = slot_map(inst)
    precede = must_precede_pairs(inst)
    s_y, s_v, t_y = inst.s_y, inst.s_v, inst.t_y

    def rebuild_vessel(vh: VHeights) -> list[list[int]]:
        return [list(inst.stowage[i][:vh[i]]) for i in range(s_v)]

    # root: apply closure first (paper: nodes never hold transferable containers)
    yard0 = [list(s) for s in inst.yard]
    vessel0 = [[] for _ in range(s_v)]
    g0 = _closure(yard0, vessel0, slot_of, s_v)
    root_yard = tuple(tuple(s) for s in yard0)
    root_vh = tuple(len(v) for v in vessel0)

    counter = itertools.count()
    start = time.perf_counter()
    h0 = lower_bound(root_yard, precede)
    heap = [(g0 + h0, next(counter), g0, root_yard, root_vh)]
    best_g: dict = {(tuple(sorted(root_yard)), root_vh): g0}
    parent: dict = {(root_yard, root_vh): None}   # child -> (parent_state, action)
    expanded = 0

    while heap:
        if time_limit_s is not None and time.perf_counter() - start > time_limit_s:
            return AStarResult(-1, -1, False, expanded)
        if node_limit is not None and expanded > node_limit:
            return AStarResult(-1, -1, False, expanded)
        f, _, g, yard, vh = heapq.heappop(heap)
        if all(not s for s in yard):
            traj = _reconstruct(parent, (yard, vh))
            return AStarResult(g, g - inst.n, True, expanded, traj)
        key = (tuple(sorted(yard)), vh)
        if g > best_g.get(key, float("inf")):
            continue
        expanded += 1
        for s in range(s_y):
            if not yard[s]:
                continue
            for d in range(s_y):
                if d == s or len(yard[d]) >= t_y:
                    continue
                ny = [list(st) for st in yard]
                ny[d].append(ny[s].pop())
                nv = rebuild_vessel(vh)
                moved = _closure(ny, nv, slot_of, s_v)
                child_yard = tuple(tuple(st) for st in ny)
                child_vh = tuple(len(v) for v in nv)
                ng = g + 1 + moved
                ckey = (tuple(sorted(child_yard)), child_vh)
                if ng < best_g.get(ckey, float("inf")):
                    best_g[ckey] = ng
                    parent[(child_yard, child_vh)] = ((yard, vh), (s, d))
                    nh = lower_bound(child_yard, precede)
                    heapq.heappush(heap, (ng + nh, next(counter), ng, child_yard, child_vh))
    return AStarResult(-1, -1, False, expanded)


def _reconstruct(parent, goal):
    traj = []
    cur = goal
    while parent.get(cur) is not None:
        prev, action = parent[cur]
        traj.append((prev[0], prev[1], action))
        cur = prev
    traj.reverse()
    return traj
```

- [ ] **Step 4: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_astar.py -v`
Expected: all PASS in a few seconds.

- [ ] **Step 5: Commit**

```bash
git add crpsp/astar.py tests/test_astar.py
git commit -m "feat: A* solver with closure nodes and optimal-trajectory extraction"
```

---

### Task 6: MIP via CP-SAT (paper Section 2.2, Eq 1–26)

**Files:**
- Create: `crpsp/mip_cpsat.py`
- Test: `tests/test_mip.py`

**Interfaces:**
- Consumes: `Instance`, `slot_map`.
- Produces `solve_mip(inst, horizon=None, time_limit_s=120.0, w1=10, w2=1) -> MipResult` with fields `status: str` (`OPTIMAL|FEASIBLE|INFEASIBLE|UNKNOWN`), `total_ops`, `relocations`, `target_reached: bool`, `wall_time_s`.

- [ ] **Step 1: Write failing tests `tests/test_mip.py`**

```python
import random
import pytest
from crpsp.instance import Instance, generate_instance
from crpsp.mip_cpsat import solve_mip
from crpsp.astar import solve_astar

def _inst(yard, stowage, t_y=3):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)

def test_trivial_no_relocation():
    inst = _inst([[2, 1], []], [[1, 2]])
    res = solve_mip(inst, horizon=4)
    assert res.status == "OPTIMAL" and res.target_reached
    assert res.total_ops == 2 and res.relocations == 0

def test_one_relocation():
    inst = _inst([[1, 2], []], [[1, 2]])
    res = solve_mip(inst, horizon=5)
    assert res.status == "OPTIMAL" and res.target_reached
    assert res.total_ops == 3 and res.relocations == 1

def test_eq23_enforced():
    # far stack must be loaded first; MIP must still reach target
    inst = _inst([[1], [2], []], [[2], [1]])
    res = solve_mip(inst, horizon=4)
    assert res.status == "OPTIMAL" and res.target_reached and res.relocations == 0

@pytest.mark.slow
def test_cross_check_vs_astar_n5():
    rng = random.Random(0)
    for _ in range(3):
        inst = generate_instance(5, 3, 3, 5, rng)
        a = solve_astar(inst)
        ub = a.total_ops
        m = solve_mip(inst, horizon=ub + 2, time_limit_s=120)
        assert m.status == "OPTIMAL" and m.target_reached
        assert m.total_ops == a.total_ops
```

- [ ] **Step 2: Run fast tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_mip.py -v`
Expected: FAIL — module missing (slow test excluded by default).

- [ ] **Step 3: Implement `crpsp/mip_cpsat.py`** — literal translation of Eq (1)–(26); 0-based tiers (paper is 1-based). Deviations: Eq (11) enforced only on operation steps (notes #6); redundant gravity constraint added as a solver aid.

```python
"""CP-SAT translation of the CRPSP MIP, Eq (1)-(26) of Wang et al. (2025)."""
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
    z = model.NewBoolVar(name)
    model.AddBoolAnd([a, b]).OnlyEnforceIf(z)
    model.AddBoolOr([a.Not(), b.Not()]).OnlyEnforceIf(z.Not())
    return z


def solve_mip(inst: Instance, horizon: int | None = None,
              time_limit_s: float = 120.0, w1: int = 10, w2: int = 1) -> MipResult:
    n, s_y, s_v, t_y = inst.n, inst.s_y, inst.s_v, inst.t_y
    n_stacks = s_y + s_v                       # 0..s_y-1 yard, s_y..s_y+s_v-1 vessel
    k_max = max(t_y, max((len(st) for st in inst.stowage), default=1))
    H = horizon if horizon is not None else 2 * n
    m = cp_model.CpModel()

    # s[i,k,c,t]: container c (0-based id c+1) at stack i tier k, before op t; t in 0..H
    s = {(i, k, c, t): m.NewBoolVar(f"s_{i}_{k}_{c}_{t}")
         for i in range(n_stacks) for k in range(k_max)
         for c in range(n) for t in range(H + 1)}
    x = {(i, j, t): m.NewBoolVar(f"x_{i}_{j}_{t}")
         for i in range(n_stacks) for j in range(n_stacks) for t in range(H)}

    for t in range(H):
        for i in range(n_stacks):
            m.Add(x[i, i, t] == 0)                                   # Eq (2)
            if i >= s_y:                                             # Eq (3): no moves out of vessel
                for j in range(n_stacks):
                    m.Add(x[i, j, t] == 0)

    # yard tier cap: yard stacks never exceed t_y
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
    # gravity (redundant, implicit in the paper; solver aid)
    for i in range(n_stacks):
        for k in range(1, k_max):
            for t in range(H + 1):
                m.Add(sum(s[i, k, c, t] for c in range(n))
                      <= sum(s[i, k - 1, c, t] for c in range(n)))

    op, o_s, o_e, height = {}, {}, {}, {}
    for t in range(H):
        m.Add(sum(x[i, j, t] for i in range(n_stacks) for j in range(n_stacks)) <= 1)  # Eq (6)
        op[t] = m.NewBoolVar(f"op_{t}")
        m.Add(sum(x[i, j, t] for i in range(n_stacks) for j in range(n_stacks)) == op[t])
        for i in range(n_stacks):
            o_s[i, t] = m.NewBoolVar(f"oS_{i}_{t}")                  # Eq (7)
            m.Add(sum(x[i, j, t] for j in range(n_stacks)) == o_s[i, t])
            o_e[i, t] = m.NewBoolVar(f"oE_{i}_{t}")                  # Eq (8)
            m.Add(sum(x[j, i, t] for j in range(n_stacks)) == o_e[i, t])
    for i in range(n_stacks):
        for t in range(H + 1):
            height[i, t] = m.NewIntVar(0, k_max, f"h_{i}_{t}")
            m.Add(height[i, t] == sum(s[i, k, c, t] for k in range(k_max) for c in range(n)))

    h_s, h_e, g_s, g_e, r_s, r_e = {}, {}, {}, {}, {}, {}
    for t in range(H):
        h_s[t] = m.NewIntVar(0, k_max, f"hS_{t}")                    # Eq (9)
        h_e[t] = m.NewIntVar(0, k_max, f"hE_{t}")                    # Eq (10)
        zs = []
        ze = []
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
        m.Add(h_s[t] >= 1).OnlyEnforceIf(op[t])                      # Eq (11), relaxed (notes #6)
        m.Add(h_e[t] <= k_max - 1)                                   # Eq (12)
        for k in range(k_max):
            g_s[k, t] = m.NewBoolVar(f"gS_{k}_{t}")
            g_e[k, t] = m.NewBoolVar(f"gE_{k}_{t}")
        m.Add(sum((k + 1) * g_s[k, t] for k in range(k_max)) == h_s[t])   # Eq (13) 0-based
        m.Add(sum(g_s[k, t] for k in range(k_max)) == op[t])              # Eq (14) tied to op
        m.Add(sum(k * g_e[k, t] for k in range(k_max)) == h_e[t])         # Eq (15) 0-based
        m.Add(sum(g_e[k, t] for k in range(k_max)) == op[t])              # Eq (16) tied to op
        for i in range(n_stacks):
            for k in range(k_max):
                r_s[i, k, t] = _and2(m, o_s[i, t], g_s[k, t], f"rS_{i}_{k}_{t}")  # Eq (17)
                r_e[i, k, t] = _and2(m, o_e[i, t], g_e[k, t], f"rE_{i}_{k}_{t}")  # Eq (18)

    # Eq (19): initial state (yard only, vessel empty)
    init = {(i, k): c for i, stack in enumerate(inst.yard) for k, c in enumerate(stack)}
    for i in range(n_stacks):
        for k in range(k_max):
            for c in range(n):
                want = 1 if init.get((i, k)) == c + 1 else 0
                m.Add(s[i, k, c, 0] == want)

    # Eq (20)-(21): moved container and state transition
    for t in range(H):
        mc = {}
        for c in range(n):
            mf_list = []
            for i in range(n_stacks):
                for k in range(k_max):
                    mf = _and2(m, r_s[i, k, t], s[i, k, c, t], f"mf_{i}_{k}_{c}_{t}")
                    mf_list.append((i, k, mf))
            mc[c] = m.NewBoolVar(f"m_{c}_{t}")
            m.Add(sum(v for _, _, v in mf_list) == mc[c])            # Eq (20)
            for i, k, mf in mf_list:
                mt = _and2(m, r_e[i, k, t], mc[c], f"mt_{i}_{k}_{c}_{t}")
                m.Add(s[i, k, c, t + 1] == s[i, k, c, t] - mf + mt)  # Eq (21)
        # cells not covered above still need Eq 21's mt term; add for all (i,k,c):
        # (handled: loop above covers every (i,k,c) since mf_list spans all i,k)

    # Eq (23): vessel monotonicity at every time slice
    for t in range(H + 1):
        for va in range(s_v):
            for vb in range(va + 1, s_v):
                m.Add(height[s_y + vb, t] <= height[s_y + va, t])

    # Eq (24) + objective Eq (1): |s_final - S_T| linearized by constants
    target = {(s_y + vs, k): c for vs, stack in enumerate(inst.stowage)
              for k, c in enumerate(stack)}
    mis_terms = []
    for i in range(n_stacks):
        for k in range(k_max):
            for c in range(n):
                st = 1 if target.get((i, k)) == c + 1 else 0
                if st == 1:
                    lit = s[i, k, c, H].Not()
                else:
                    lit = s[i, k, c, H]
                mis_terms.append(lit)
    m.Minimize(w1 * sum(mis_terms)
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
    total = sum(int(solver.Value(x[i, j, t])) for i in range(n_stacks)
                for j in range(n_stacks) for t in range(H))
    reloc = sum(int(solver.Value(x[i, j, t])) for i in range(s_y)
                for j in range(s_y) for t in range(H))
    mis = sum(int(solver.Value(v) if hasattr(v, "Index") else solver.BooleanValue(v))
              for v in mis_terms)
    return MipResult(sname, total, reloc, mis == 0, wall)
```

  Note for executor: `mis_terms` holds literals (possibly negated); evaluate with `solver.BooleanValue(lit)` for all of them (simplify the `mis` line accordingly).

- [ ] **Step 4: Run fast tests, then the slow cross-check once**

Run: `.venv/Scripts/python -m pytest tests/test_mip.py -v`
Expected: 3 fast tests PASS.
Run: `.venv/Scripts/python -m pytest tests/test_mip.py -m slow -v`
Expected: cross-check vs A* PASS (may take ~1–5 min total; acceptable one-time verification, not part of default suite).

- [ ] **Step 5: Commit**

```bash
git add crpsp/mip_cpsat.py tests/test_mip.py
git commit -m "feat: CP-SAT translation of MIP Eq(1)-(26) with A* cross-check"
```

---

### Task 7: Forward-looking greedy heuristic (paper Section 5.7)

**Files:**
- Create: `crpsp/heuristic.py`
- Test: `tests/test_heuristic.py`

**Interfaces:**
- Consumes: `Instance`, `slot_map`, `must_precede_pairs`.
- Produces `solve_heuristic(inst, max_ops=None) -> HeuristicResult` with `total_ops`, `relocations`, `solved: bool`.

- [ ] **Step 1: Write failing tests `tests/test_heuristic.py`**

```python
import random
from crpsp.instance import Instance, generate_instance
from crpsp.heuristic import solve_heuristic
from crpsp.astar import solve_astar

def test_solves_trivial():
    inst = Instance(((2, 1), ()), ((1, 2),), 3)
    r = solve_heuristic(inst)
    assert r.solved and r.relocations == 0 and r.total_ops == 2

def test_never_beats_optimal_and_always_solves():
    rng = random.Random(0)
    for _ in range(20):
        inst = generate_instance(10, 4, 4, 5, rng)
        h = solve_heuristic(inst)
        a = solve_astar(inst)
        assert h.solved
        assert h.total_ops >= a.total_ops
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_heuristic.py -v` — FAIL, module missing.

- [ ] **Step 3: Implement `crpsp/heuristic.py`**

```python
"""Forward-looking greedy heuristic (paper Section 5.7; notes #12)."""
from __future__ import annotations

from dataclasses import dataclass

from .instance import Instance, slot_map
from .lower_bound import must_precede_pairs


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
        moved = True
        while moved:
            moved = False
            for st in yard:
                if not st:
                    continue
                c = st[-1]
                vs, vt = slot_of[c]
                if len(vessel[vs]) == vt and height_ok(vs):
                    vessel[vs].append(st.pop())
                    n_tra += 1
                    moved = True

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
        _, _, si = min(cands)              # fewest blockers on top
        b = yard[si][-1]                   # relocate topmost blocker
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
```

- [ ] **Step 4: Run tests** — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crpsp/heuristic.py tests/test_heuristic.py
git commit -m "feat: forward-looking greedy heuristic baseline (Section 5.7)"
```

---

### Task 8: Actor & Critic networks (paper Section 4.2, Eq 38–39, Fig. 9)

**Files:**
- Create: `crpsp/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces:
  - `RowSelfAttention(n_tiers, d)` — `forward(x: (B,m,n)) -> (B,m,d)` implementing Eq 38–39.
  - `Actor(s_y, t_y, hidden_dim=128, use_attention=True)` — `forward(obs: (B,s_y,t_y), mask: (B,A)) -> logits (B,A)` with invalid actions at −1e9; `A = s_y*(s_y-1)`.
  - `Critic(s_y, t_y, hidden_dim=128, use_attention=True)` — `forward(obs) -> (B,)`.

- [ ] **Step 1: Write failing tests `tests/test_models.py`**

```python
import torch
from crpsp.models import RowSelfAttention, Actor, Critic

def test_attention_shapes_eq38_39():
    att = RowSelfAttention(n_tiers=5, d=128)
    x = torch.randn(7, 6, 5)                      # B=7, m=6 stacks, n=5 tiers
    out = att(x)
    assert out.shape == (7, 6, 128)

def test_attention_rows_are_convex_combinations():
    att = RowSelfAttention(3, 4)
    x = torch.randn(1, 2, 3)
    q = att.wq(x); k = att.wk(x)
    w = torch.softmax(q @ k.transpose(-2, -1) / (4 ** 0.5), dim=-1)
    assert torch.allclose(w.sum(-1), torch.ones(1, 2), atol=1e-6)

def test_actor_masks_invalid_actions():
    actor = Actor(s_y=4, t_y=5)
    obs = torch.randn(2, 4, 5)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, 0] = False
    logits = actor(obs, mask)
    assert logits.shape == (2, 12)
    assert (logits[:, 0] < -1e8).all()
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs[:, 0], torch.zeros(2), atol=1e-6)

def test_critic_scalar_and_grads_flow():
    critic = Critic(s_y=4, t_y=5)
    obs = torch.randn(3, 4, 5, requires_grad=True)
    v = critic(obs)
    assert v.shape == (3,)
    v.sum().backward()
    assert obs.grad is not None

def test_no_attention_variant():
    actor = Actor(s_y=4, t_y=5, use_attention=False)
    logits = actor(torch.randn(2, 4, 5), torch.ones(2, 12, dtype=torch.bool))
    assert logits.shape == (2, 12)
```

- [ ] **Step 2: Run tests to verify they fail** — FAIL, module missing.

- [ ] **Step 3: Implement `crpsp/models.py`**

```python
"""Actor-critic networks with row-wise self-attention (Eq 38-39, Fig. 9, notes #7)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RowSelfAttention(nn.Module):
    """Single-head scaled dot-product attention over yard-stack rows (Eq 38-39)."""

    def __init__(self, n_tiers: int, d: int):
        super().__init__()
        self.d = d
        self.wq = nn.Linear(n_tiers, d, bias=False)   # Q = S W_Q
        self.wk = nn.Linear(n_tiers, d, bias=False)   # K = S W_K
        self.wv = nn.Linear(n_tiers, d, bias=False)   # V = S W_V

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, m, n)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d), dim=-1)
        return att @ v                                        # (B, m, d)


class _Trunk(nn.Module):
    def __init__(self, s_y: int, t_y: int, hidden_dim: int, use_attention: bool):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att = RowSelfAttention(t_y, hidden_dim)
            in_dim = s_y * hidden_dim
        else:
            self.att = None
            in_dim = s_y * t_y
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:     # obs: (B, s_y, t_y)
        h = self.att(obs) if self.use_attention else obs
        return self.fc(h.flatten(1))


class Actor(nn.Module):
    def __init__(self, s_y: int, t_y: int, hidden_dim: int = 128, use_attention: bool = True):
        super().__init__()
        self.trunk = _Trunk(s_y, t_y, hidden_dim, use_attention)
        self.head = nn.Linear(hidden_dim, s_y * (s_y - 1))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.trunk(obs))
        return logits.masked_fill(~mask, -1e9)


class Critic(nn.Module):
    def __init__(self, s_y: int, t_y: int, hidden_dim: int = 128, use_attention: bool = True):
        super().__init__()
        self.trunk = _Trunk(s_y, t_y, hidden_dim, use_attention)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs)).squeeze(-1)
```

- [ ] **Step 4: Run tests** — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add crpsp/models.py tests/test_models.py
git commit -m "feat: self-attention actor-critic networks (Eq 38-39)"
```

---

### Task 9: PPO trainer (paper Section 4.1–4.2, Algorithm 1)

**Files:**
- Create: `crpsp/ppo.py`
- Test: `tests/test_ppo.py`

**Interfaces:**
- Consumes: `CRPSPEnv`, `Actor`, `Critic`, `generate_instance`.
- Produces:
  - `compute_gae(rewards, values, next_values, dones, gamma, lam) -> (advantages, td_targets)` — lists/1-D tensors per episode; `td_targets[t] = r_t + γ·next_values[t]·(1−done_t)` (Algorithm 1 line 17).
  - `TrainConfig` dataclass mirroring `configs/default.yaml` keys (env+model+train flattened).
  - `PPOTrainer(cfg)` with `.train_iteration(rng) -> dict` (one Algorithm-1 outer loop: collect M episodes, single-pass minibatch updates; returns metrics: `mean_reward`, `solve_rate`, `mean_relocations`, `actor_loss`, `critic_loss`) and `.save(path)` / `.load(path)`.
  - `convergence_iteration(history: list[dict], window=20) -> int | None` (notes #10).

- [ ] **Step 1: Write failing tests `tests/test_ppo.py`**

```python
import random
import torch
from crpsp.ppo import compute_gae, TrainConfig, PPOTrainer, convergence_iteration

def test_gae_hand_computed():
    # gamma=0.5, lam=0.5 ; two steps, terminal at t=1
    rewards      = torch.tensor([1.0, 1.0])
    values       = torch.tensor([0.5, 0.5])
    next_values  = torch.tensor([0.5, 0.9])   # bootstrap ignored at terminal
    dones        = torch.tensor([0.0, 1.0])
    adv, targets = compute_gae(rewards, values, next_values, dones, 0.5, 0.5)
    # d0 = 1 + 0.5*0.5 - 0.5 = 0.75 ; d1 = 1 + 0 - 0.5 = 0.5
    # A1 = 0.5 ; A0 = 0.75 + (0.25)*0.5 = 0.875
    assert torch.allclose(adv, torch.tensor([0.875, 0.5]), atol=1e-6)
    # y0 = 1 + 0.5*0.5 = 1.25 ; y1 = 1 + 0 (terminal) = 1.0
    assert torch.allclose(targets, torch.tensor([1.25, 1.0]), atol=1e-6)

def _tiny_cfg():
    return TrainConfig(n=5, s_y=3, s_v=3, t_y=3, reward_mode="designed",
                       terminal_bonus=10.0, max_steps=50, hidden_dim=32,
                       use_attention=True, lr=5e-4, gamma=0.4, gae_lambda=0.9,
                       clip_eps=0.15, batch_size=10, instances_per_iter=2,
                       iterations=3, ppo_epochs=1, seed=0, device="cpu")

def test_smoke_train_three_iterations():
    """Machinery check only — NOT a training run (Global Constraints)."""
    cfg = _tiny_cfg()
    tr = PPOTrainer(cfg)
    rng = random.Random(0)
    hist = [tr.train_iteration(rng) for _ in range(3)]
    for m in hist:
        assert torch.isfinite(torch.tensor(m["actor_loss"]))
        assert torch.isfinite(torch.tensor(m["critic_loss"]))
        assert 0.0 <= m["solve_rate"] <= 1.0

def test_save_load_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    tr = PPOTrainer(cfg)
    p = tmp_path / "ck.pt"
    tr.save(p)
    tr2 = PPOTrainer(cfg)
    tr2.load(p)
    for a, b in zip(tr.actor.parameters(), tr2.actor.parameters()):
        assert torch.equal(a, b)

def test_convergence_iteration_definition():
    hist = [{"solve_rate": 0.0, "mean_relocations": 9}] * 5 \
         + [{"solve_rate": 1.0, "mean_relocations": 3}] * 25
    assert convergence_iteration(hist, window=20) == 25   # first idx where window all-solved & stable
    assert convergence_iteration(hist[:10], window=20) is None
```

- [ ] **Step 2: Run tests to verify they fail** — FAIL, module missing.

- [ ] **Step 3: Implement `crpsp/ppo.py`**

```python
"""PPO for CRPSP — faithful implementation of Algorithm 1 (notes #8, #9, #10)."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .env import CRPSPEnv
from .instance import generate_instance
from .models import Actor, Critic


def compute_gae(rewards, values, next_values, dones, gamma: float, lam: float):
    """A_t = sum (gamma*lam)^{k-t} delta_k ; targets y_t = r_t + gamma*V(s_{t+1})(1-done)."""
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nv = next_values[t] * (1.0 - dones[t])
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    targets = rewards + gamma * next_values * (1.0 - dones)
    return adv, targets


@dataclass
class TrainConfig:
    n: int; s_y: int; s_v: int; t_y: int
    reward_mode: str; terminal_bonus: float; max_steps: int
    hidden_dim: int; use_attention: bool
    lr: float; gamma: float; gae_lambda: float; clip_eps: float
    batch_size: int; instances_per_iter: int; iterations: int
    ppo_epochs: int; seed: int; device: str = "auto"

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TrainConfig":
        return cls(**cfg["env"], **cfg["model"], **cfg["train"])

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class PPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = cfg.resolved_device()
        torch.manual_seed(cfg.seed)
        self.actor = Actor(cfg.s_y, cfg.t_y, cfg.hidden_dim, cfg.use_attention).to(self.device)
        self.critic = Critic(cfg.s_y, cfg.t_y, cfg.hidden_dim, cfg.use_attention).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.env = CRPSPEnv(cfg.reward_mode, cfg.terminal_bonus, cfg.max_steps)

    # ------------------------------------------------------------ collection
    def _rollout(self, rng: random.Random) -> tuple[list[dict], dict]:
        inst = generate_instance(self.cfg.n, self.cfg.s_y, self.cfg.s_v, self.cfg.t_y, rng)
        obs, mask = self.env.reset(inst)
        ep = []
        solved = self.env.done
        while not self.env.done and self.env.t < self.cfg.max_steps:
            to = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            tm = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.actor(to, tm)
            dist = Categorical(logits=logits)
            a = dist.sample()
            nobs, nmask, r, done, trunc, info = self.env.step(int(a.item()))
            ep.append({"obs": obs, "mask": mask, "action": int(a.item()),
                       "logp": float(dist.log_prob(a).item()), "reward": float(r),
                       "next_obs": nobs, "done": float(done)})
            obs, mask = nobs, nmask
            solved = done
            if done or trunc:
                break
        stats = {"solved": bool(solved), "relocations": self.env.n_relocations,
                 "reward": sum(e["reward"] for e in ep)}
        return ep, stats

    # ------------------------------------------------------------- one iter
    def train_iteration(self, rng: random.Random) -> dict:
        episodes, stats = [], []
        for _ in range(self.cfg.instances_per_iter):       # Algorithm 1 lines 3-12
            ep, st = self._rollout(rng)
            if ep:
                episodes.append(ep)
            stats.append(st)
        if not episodes:
            return {"mean_reward": float(np.mean([s["reward"] for s in stats])),
                    "solve_rate": float(np.mean([s["solved"] for s in stats])),
                    "mean_relocations": float(np.mean([s["relocations"] for s in stats])),
                    "actor_loss": 0.0, "critic_loss": 0.0}

        obs = torch.as_tensor(np.array([e["obs"] for ep in episodes for e in ep]),
                              device=self.device)
        masks = torch.as_tensor(np.array([e["mask"] for ep in episodes for e in ep]),
                                device=self.device)
        acts = torch.tensor([e["action"] for ep in episodes for e in ep], device=self.device)
        logp_old = torch.tensor([e["logp"] for ep in episodes for e in ep], device=self.device)

        # Algorithm 1 lines 13-18: recompute values with the CURRENT critic
        with torch.no_grad():
            all_v = self.critic(obs)
            next_obs = torch.as_tensor(np.array([e["next_obs"] for ep in episodes for e in ep]),
                                       device=self.device)
            all_nv = self.critic(next_obs)
        adv_list, tgt_list = [], []
        i = 0
        for ep in episodes:
            L = len(ep)
            r = torch.tensor([e["reward"] for e in ep])
            d = torch.tensor([e["done"] for e in ep])
            a, y = compute_gae(r, all_v[i:i + L].cpu(), all_nv[i:i + L].cpu(), d,
                               self.cfg.gamma, self.cfg.gae_lambda)
            adv_list.append(a); tgt_list.append(y)
            i += L
        adv = torch.cat(adv_list).to(self.device)
        tgt = torch.cat(tgt_list).to(self.device)

        # Algorithm 1 lines 19-25: shuffle, minibatch, clipped update (single pass)
        n_tr = len(acts)
        actor_losses, critic_losses = [], []
        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n_tr)
            for s0 in range(0, n_tr, self.cfg.batch_size):
                idx = perm[s0:s0 + self.cfg.batch_size]
                logits = self.actor(obs[idx], masks[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(acts[idx])
                ratio = torch.exp(logp - logp_old[idx])
                a_b = adv[idx]
                l_clip = -torch.min(
                    ratio * a_b,
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * a_b,
                ).mean()                                    # Eq (35)
                self.opt_actor.zero_grad(); l_clip.backward(); self.opt_actor.step()
                v = self.critic(obs[idx])
                l_v = ((tgt[idx] - v) ** 2).mean()          # Algorithm 1 line 23
                self.opt_critic.zero_grad(); l_v.backward(); self.opt_critic.step()
                actor_losses.append(float(l_clip.item()))
                critic_losses.append(float(l_v.item()))

        return {"mean_reward": float(np.mean([s["reward"] for s in stats])),
                "solve_rate": float(np.mean([s["solved"] for s in stats])),
                "mean_relocations": float(np.mean([s["relocations"] for s in stats])),
                "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses))}

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "cfg": self.cfg.__dict__}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])


def convergence_iteration(history: list[dict], window: int = 20) -> int | None:
    """First iteration index ending a window of 100% solve-rate with non-improving
    mean relocations (notes #10). None if never converged."""
    if len(history) < window:
        return None
    for end in range(window, len(history) + 1):
        w = history[end - window:end]
        if all(m["solve_rate"] >= 1.0 for m in w):
            first = w[0]["mean_relocations"]
            if all(m["mean_relocations"] >= first - 1e-9 for m in w[1:]) or \
               max(m["mean_relocations"] for m in w) - min(m["mean_relocations"] for m in w) < 1e-9:
                return end
    return None
```

  Note for executor: `convergence_iteration` must make `test_convergence_iteration_definition` pass — the intended semantics: sliding window of `window` consecutive iterations, all with `solve_rate == 1.0` and stable (non-decreasing-quality) `mean_relocations`; return the END index of the first such window. Adjust the stability predicate to the test, keep notes #10 in sync with the final predicate.

- [ ] **Step 4: Run tests** — Expected: all PASS; the smoke test finishes in seconds.

- [ ] **Step 5: Commit**

```bash
git add crpsp/ppo.py tests/test_ppo.py
git commit -m "feat: PPO trainer implementing Algorithm 1 (GAE, clipped objective, TD targets)"
```

---

### Task 10: Ensembles — Majority Voting & Stacking (paper Section 4.3)

**Files:**
- Create: `crpsp/ensemble.py`
- Test: `tests/test_ensemble.py`

**Interfaces:**
- Consumes: `Actor`, `CRPSPEnv.encode`, `solve_astar` trajectories.
- Produces:
  - `vote(actors: list[Actor], obs: np.ndarray, mask: np.ndarray, device) -> int` — Eq (40); ties broken by summed probability.
  - `build_stacking_dataset(actors, instances, device) -> (X: np.ndarray, y: np.ndarray)` — features = concatenated per-model action-probability vectors at each A*-optimal state; labels = optimal action index (Eq 41–45).
  - `StackingPolicy(model: xgboost.XGBClassifier, n_actions)` with `.predict(features, mask) -> int` choosing the best VALID action by class probability (Eq 46).
  - `train_stacking(X, y, n_actions) -> StackingPolicy`.

- [ ] **Step 1: Write failing tests `tests/test_ensemble.py`**

```python
import numpy as np
import random
import torch
from crpsp.models import Actor
from crpsp.ensemble import vote, build_stacking_dataset, train_stacking
from crpsp.instance import generate_instance

def _actors(k, s_y=3, t_y=3, seed0=0):
    out = []
    for i in range(k):
        torch.manual_seed(seed0 + i)
        out.append(Actor(s_y, t_y, hidden_dim=16))
    return out

def test_vote_majority_and_tie_break():
    s_y, t_y = 3, 3
    actors = _actors(3, s_y, t_y)
    obs = np.random.RandomState(0).rand(s_y, t_y).astype(np.float32)
    mask = np.ones(s_y * (s_y - 1), dtype=bool)
    a = vote(actors, obs, mask, torch.device("cpu"))
    assert 0 <= a < s_y * (s_y - 1)
    # single-model vote equals its argmax
    with torch.no_grad():
        logits = actors[0](torch.as_tensor(obs).unsqueeze(0),
                           torch.as_tensor(mask).unsqueeze(0))
    assert vote(actors[:1], obs, mask, torch.device("cpu")) == int(logits.argmax())

def test_vote_never_selects_masked_action():
    s_y, t_y = 3, 3
    actors = _actors(5, s_y, t_y)
    obs = np.zeros((s_y, t_y), dtype=np.float32)
    mask = np.zeros(s_y * (s_y - 1), dtype=bool)
    mask[3] = True
    assert vote(actors, obs, mask, torch.device("cpu")) == 3

def test_stacking_dataset_and_training():
    rng = random.Random(0)
    instances = [generate_instance(6, 3, 3, 4, rng) for _ in range(5)]
    actors = _actors(3, 3, 4)
    X, y = build_stacking_dataset(actors, instances, torch.device("cpu"))
    n_actions = 3 * 2
    if len(y) == 0:                    # all instances solvable with 0 relocations — regenerate
        assert False, "need at least one instance requiring relocation; adjust seed"
    assert X.shape[1] == len(actors) * n_actions
    pol = train_stacking(X, y, n_actions)
    mask = np.ones(n_actions, dtype=bool)
    a = pol.predict(X[0], mask)
    assert 0 <= a < n_actions
```

- [ ] **Step 2: Run tests to verify they fail** — FAIL, module missing.

- [ ] **Step 3: Implement `crpsp/ensemble.py`**

```python
"""Ensemble methods: Majority Voting (Eq 40) and Stacking with XGBoost (Eq 41-46)."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import xgboost as xgb

from .astar import solve_astar
from .env import CRPSPEnv
from .instance import Instance
from .models import Actor


def _probs(actor: Actor, obs: np.ndarray, mask: np.ndarray, device) -> np.ndarray:
    with torch.no_grad():
        logits = actor(torch.as_tensor(obs, device=device).unsqueeze(0),
                       torch.as_tensor(mask, device=device).unsqueeze(0))
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


def vote(actors: list[Actor], obs: np.ndarray, mask: np.ndarray, device) -> int:
    """Eq (40): most frequent argmax; ties broken by summed probability."""
    all_p = [_probs(a, obs, mask, device) for a in actors]
    choices = [int(p.argmax()) for p in all_p]
    counts = Counter(choices)
    top = max(counts.values())
    tied = [c for c, k in counts.items() if k == top]
    if len(tied) == 1:
        return tied[0]
    sums = {c: sum(p[c] for p in all_p) for c in tied}
    return max(sums, key=sums.get)


def build_stacking_dataset(actors: list[Actor], instances: list[Instance], device):
    """(state, optimal action) pairs from A* trajectories (Eq 41-44);
    features = stacked per-model probability vectors (Eq 45)."""
    X, y = [], []
    for inst in instances:
        res = solve_astar(inst)
        if not res.optimal:
            continue
        env = CRPSPEnv()
        for yard, vh, action in res.trajectory:
            obs = CRPSPEnv.encode(yard, inst.s_y, inst.t_y, inst.n)
            mask = np.zeros(inst.s_y * (inst.s_y - 1), dtype=bool)
            pairs = [(s, d) for s in range(inst.s_y) for d in range(inst.s_y) if d != s]
            for i, (s, d) in enumerate(pairs):
                mask[i] = len(yard[s]) > 0 and len(yard[d]) < inst.t_y
            feats = np.concatenate([_probs(a, obs, mask, device) for a in actors])
            X.append(feats)
            y.append(pairs.index(action))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class StackingPolicy:
    def __init__(self, model: xgb.XGBClassifier, n_actions: int, classes: np.ndarray):
        self.model = model
        self.n_actions = n_actions
        self.classes = classes                     # maps model output cols -> action ids

    def predict(self, features: np.ndarray, mask: np.ndarray) -> int:
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        full = np.zeros(self.n_actions)
        for col, action in enumerate(self.classes):
            full[int(action)] = proba[col]
        full[~mask] = -1.0
        return int(full.argmax())


def train_stacking(X: np.ndarray, y: np.ndarray, n_actions: int) -> StackingPolicy:
    model = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss")
    model.fit(X, y)
    return StackingPolicy(model, n_actions, model.classes_)
```

- [ ] **Step 4: Run tests** — Expected: all PASS. If the dataset test's assertion about needing relocations trips, bump the instance count or seed until at least one A* trajectory is non-empty (deterministic once chosen).

- [ ] **Step 5: Commit**

```bash
git add crpsp/ensemble.py tests/test_ensemble.py
git commit -m "feat: majority-voting and XGBoost-stacking ensembles (Section 4.3)"
```

---

### Task 11: Evaluation + experiment scripts (paper Section 5, Tables 2–9)

**Files:**
- Create: `crpsp/evaluate.py`, `experiments/common.py`, `experiments/table2_mip.py`, `experiments/table3_astar.py`, `experiments/table4_ppo_vs_astar.py`, `experiments/table5_attention_ablation.py`, `experiments/table6_7_reward_ablation.py`, `experiments/table8_ensemble.py`, `experiments/table9_voting_vs_heuristic.py`, `experiments/train_ppo.py`
- Test: `tests/test_evaluate.py`, `tests/test_experiments_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `gap(alg_ops: int, opt_ops: int) -> float` (notes #11).
  - `rollout_policy(select_action: Callable[[obs, mask], int], inst, env_kwargs) -> dict` with `total_ops`, `relocations`, `solved`, `seconds`.
  - `experiments/common.py`: `parse_args()` (flags `--smoke` default True, `--full`, `--out results/...csv`, `--seed`), `write_csv(path, rows)`.
  - `experiments/train_ppo.py`: CLI to run full training from `configs/default.yaml` — **server use**; refuses to run without `--full`.

- [ ] **Step 1: Write failing tests `tests/test_evaluate.py`**

```python
import random
import pytest
from crpsp.instance import generate_instance
from crpsp.evaluate import gap, rollout_policy
from crpsp.astar import solve_astar

def test_gap_matches_table4_examples():
    assert gap(21, 20) == pytest.approx(0.05)     # 5.00%
    assert gap(22, 20) == pytest.approx(0.10)
    assert gap(24, 22) == pytest.approx(2 / 22)   # 9.09%

def test_rollout_random_policy_reports_ops():
    rng = random.Random(0)
    inst = generate_instance(8, 3, 3, 4, rng)
    import numpy as np
    def rand_policy(obs, mask):
        return int(rng.choice(list(np.flatnonzero(mask))))
    out = rollout_policy(rand_policy, inst, {"max_steps": 50})
    assert "total_ops" in out and "solved" in out and out["seconds"] >= 0
    if out["solved"]:
        assert out["total_ops"] >= solve_astar(inst).total_ops
```

- [ ] **Step 2: Implement `crpsp/evaluate.py`**

```python
"""Evaluation utilities: optimality gap (notes #11) and policy rollouts."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .env import CRPSPEnv
from .instance import Instance


def gap(alg_ops: int, opt_ops: int) -> float:
    return (alg_ops - opt_ops) / opt_ops


def rollout_policy(select_action: Callable, inst: Instance, env_kwargs: dict) -> dict:
    env = CRPSPEnv(**env_kwargs)
    t0 = time.perf_counter()
    obs, mask = env.reset(inst)
    while not env.done and env.t < env.max_steps:
        a = select_action(obs, mask)
        obs, mask, r, done, trunc, info = env.step(a)
        if done or trunc:
            break
    secs = time.perf_counter() - t0
    return {"total_ops": env.n_relocations + env.n_transfers,
            "relocations": env.n_relocations, "solved": env.done, "seconds": secs}
```

- [ ] **Step 3: Implement `experiments/common.py`**

```python
"""Shared CLI plumbing. --smoke (default) = tiny sizes for wiring checks;
--full = paper-scale runs, intended for the training server ONLY."""
from __future__ import annotations

import argparse
import csv
import pathlib


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--full", action="store_true",
                   help="paper-scale run (server only); default is a tiny smoke run")
    p.add_argument("--out", default=None, help="CSV output path")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.smoke = not args.full
    return args


def write_csv(path: str | pathlib.Path, rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
```

- [ ] **Step 4: Implement the seven table scripts.** Pattern (complete example for Table 3; others follow the same shape with their own parameter grids):

`experiments/table3_astar.py`:

```python
"""Table 3: A* solve times. Paper grid: (N,S_y,S_v,T_y) x 200 instances.
Smoke: first two configs x 3 instances."""
import random
import sys
import time
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from crpsp.astar import solve_astar
from crpsp.instance import generate_instance
from experiments.common import parse_args, write_csv

FULL_GRID = [(12, 4, 4, 5), (12, 4, 5, 5), (15, 4, 4, 5), (15, 4, 5, 5),
             (15, 5, 5, 5), (15, 5, 6, 5), (20, 5, 5, 5), (20, 5, 6, 5)]
FULL_COUNT = 200

def main():
    args = parse_args(__doc__)
    grid = FULL_GRID if args.full else FULL_GRID[:2]
    count = FULL_COUNT if args.full else 3
    rng = random.Random(args.seed)
    rows = []
    for (n, s_y, s_v, t_y) in grid:
        times, ok = [], 0
        for _ in range(count):
            inst = generate_instance(n, s_y, s_v, t_y, rng)
            t0 = time.perf_counter()
            res = solve_astar(inst, time_limit_s=1000)
            times.append(time.perf_counter() - t0)
            ok += int(res.optimal)
        rows.append({"N": n, "S_y": s_y, "S_v": s_v, "T_y": t_y,
                     "instances": count, "solved": ok,
                     "mean_T_s": sum(times) / len(times)})
        print(rows[-1])
    write_csv(args.out or "results/table3_astar.csv", rows)

if __name__ == "__main__":
    main()
```

  The other scripts (each with `FULL_*` constants straight from the paper and a tiny smoke subset):
  - `table2_mip.py`: grid `[(5,2,2,4),(5,3,3,5),(6,3,3,5),(7,3,3,5),(9,3,3,5),(10,3,3,5)]`, one instance each, `solve_mip` with `horizon = solve_heuristic(inst).total_ops + 2`, `time_limit_s=10800` full / `60` smoke, records status/gap/time. Smoke = first two configs.
  - `table4_ppo_vs_astar.py`: loads a checkpoint (`--ckpt` arg required), evaluates greedy-argmax policy vs `solve_astar` on 200 (full) / 3 (smoke) instances of (15,5,5,5); reports per-instance and mean gap/time.
  - `table5_attention_ablation.py`: trains 5 models with attention and 5 without (full) — **server only**; smoke asserts both model variants build and run 1 iteration.
  - `table6_7_reward_ablation.py`: full grid LR {1e-4, 5e-4} × γ {0.4, 0.5, 0.7} × N {5, 10, 15} × ε {0.01, 0.05, 0.15} × 6 runs, cap 1000 iterations, records `convergence_iteration` for simple vs designed rewards; smoke = 1 cell × 2 iterations.
  - `table8_ensemble.py`: 10 checkpoints (`--ckpt-dir`), voting + stacking on 200 (full) / 3 (smoke) instances of (15,5,5,5).
  - `table9_voting_vs_heuristic.py`: N ∈ {13..17}, 50 (full) / 3 (smoke) instances, voting ensemble vs `solve_heuristic`.
  - `train_ppo.py`: `--full` required to start real training (1000 iterations by default from config, `--iterations` override, `--n/--s-y/--s-v/--t-y` overrides, `--reward {designed,simple}`, `--no-attention`, `--seed`, `--out checkpoints/...pt`); saves checkpoint + metrics CSV per iteration; prints a warning and exits if `--full` missing.

- [ ] **Step 5: Write `tests/test_experiments_smoke.py`**

```python
"""Wiring checks only: every script must import and its smoke path must run fast."""
import subprocess
import sys

PY = sys.executable

def _run(script, extra=()):
    r = subprocess.run([PY, f"experiments/{script}", *extra],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr

def test_table3_smoke():
    _run("table3_astar.py")

def test_table2_smoke():
    _run("table2_mip.py")

def test_train_ppo_refuses_without_full():
    r = subprocess.run([PY, "experiments/train_ppo.py"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode != 0          # refuses: training is server-only
```

- [ ] **Step 6: Run tests**

Run: `.venv/Scripts/python -m pytest tests/test_evaluate.py tests/test_experiments_smoke.py -v`
Expected: PASS (table2 smoke ≤ ~2 min worst case with 60 s CP-SAT limit).

- [ ] **Step 7: Commit**

```bash
git add crpsp/evaluate.py experiments/ tests/test_evaluate.py tests/test_experiments_smoke.py
git commit -m "feat: evaluation utils and Table 2-9 experiment scripts (smoke/full split)"
```

---

### Task 12: Standard-CRP mode (Caserta-compatible extension)

**Files:**
- Modify: `crpsp/instance.py` (add `load_caserta`, extend generator docstring), `crpsp/env.py` (CRP transfer rule + restricted mask)
- Test: `tests/test_crp_mode.py`, fixture `tests/fixtures/caserta_sample.dat`

**Interfaces:**
- Produces:
  - `load_caserta(path) -> Instance` — parses the Caserta format (line 1: `<n_stacks> <n_containers>`; line i+1: `<count> <p1> ... <pcount>` bottom→top priorities), returns `Instance(mode="crp", priorities=(1,...,N), stowage=())` with `t_y = max observed height + 2` (the CV convention `H_max = H + 2`).
  - `CRPSPEnv` in CRP mode: transfer rule = top container equals the lowest-priority remaining id; `restricted=True` env flag limits sources to the stack holding the current retrieval target.

- [ ] **Step 1: Write fixture `tests/fixtures/caserta_sample.dat`**

```
3 6
2 5 2
2 4 1
2 6 3
```

- [ ] **Step 2: Write failing tests `tests/test_crp_mode.py`**

```python
import numpy as np
from crpsp.instance import load_caserta
from crpsp.env import CRPSPEnv
from crpsp.astar import solve_astar

def test_parse_caserta():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    assert inst.mode == "crp"
    assert inst.s_y == 3 and inst.n == 6
    assert inst.yard == ((5, 2), (4, 1), (6, 3))
    assert inst.t_y == 4                      # H=2 -> H+2
    assert inst.priorities == (1, 2, 3, 4, 5, 6)

def test_crp_transfer_rule():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    env = CRPSPEnv()
    obs, mask = env.reset(inst)
    # priority 1 is under 4 in stack 1 -> no transfer at reset
    assert env.n_transfers == 0 and not env.done
    # relocate 4 (stack1 top) elsewhere -> 1 then 2 auto-retrieve? 2 is under 5: only 1 retrieves
    a = env.action_pairs.index((1, 0))
    obs, mask, r, done, trunc, info = env.step(a)
    assert info["n_transfers"] == 1

def test_astar_solves_crp_instance():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    res = solve_astar(inst)
    assert res.optimal and res.total_ops >= 6

def test_restricted_mask():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    env = CRPSPEnv(restricted=True)
    obs, mask = env.reset(inst)
    # target=1 sits in stack index 1 -> only sources from stack 1 allowed
    for i, (s, d) in enumerate(env.action_pairs):
        if mask[i]:
            assert s == 1
```

- [ ] **Step 3: Run tests to verify they fail** — FAIL (`load_caserta` missing).

- [ ] **Step 4: Implement.** In `instance.py` add:

```python
def load_caserta(path) -> Instance:
    """Caserta/Voss BRP format: '<n_stacks> <n>' then per stack '<cnt> <ids bottom->top>'.
    Yard height limit follows the CV convention H_max = H + 2."""
    import pathlib
    lines = [ln.split() for ln in pathlib.Path(path).read_text().split("\n") if ln.strip()]
    n_stacks, n = int(lines[0][0]), int(lines[0][1])
    yard = []
    for row in lines[1:1 + n_stacks]:
        cnt = int(row[0])
        yard.append(tuple(int(v) for v in row[1:1 + cnt]))
    h = max((len(s) for s in yard), default=0)
    return Instance(yard=tuple(yard), stowage=(), t_y=h + 2, mode="crp",
                    priorities=tuple(range(1, n + 1)))
```

  In `env.py`: `__init__` gains `restricted: bool = False`. In `reset`, when `inst.mode == "crp"`: `self.slot_of = None`; track `self._next_priority = 1`. `_transfer_closure` for CRP: while any stack top equals `self._next_priority`, pop it (counts as transfer/retrieval) and increment. `_all_placed` = `self._next_priority > inst.n`. `_mask` when `restricted`: only allow sources equal to the stack currently containing `self._next_priority`. A* (`astar.py`) needs the same branching: replace `slot_map/closure` calls with a mode dispatch — extract the closure into a shared helper `crpsp/transfer.py` OR give `_closure` a mode parameter; keep ONE implementation used by env, astar and heuristic (DRY). The `must_precede_pairs` CRP branch already exists (Task 3).

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: everything PASS (CRPSP tests unaffected).

- [ ] **Step 6: Commit**

```bash
git add crpsp/ tests/test_crp_mode.py tests/fixtures/
git commit -m "feat: standard-CRP mode with Caserta parser and restricted variant"
```

---

### Task 13: Final review pass, notes, README

**Files:**
- Modify: `REPRODUCTION_NOTES.md` (fill any decision made during implementation that isn't recorded)
- Create: `README.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Reconcile `REPRODUCTION_NOTES.md`** — re-read every module; every deviation/choice must appear in the notes (grep for "notes #" references and confirm each exists; add missing ones, e.g. the final convergence predicate).

- [ ] **Step 2: Write `README.md`**

```markdown
# CRPSP Reproduction — Wang et al. (2025), Transportation Research Part E

Faithful implementation of "Learning-based hybrid algorithms for container
relocation problem with storage plan" (DOI 10.1016/j.tre.2025.104048):
MIP (CP-SAT), A* with cycle lower bound, PPO + self-attention actor-critic,
Majority Voting / Stacking ensembles, greedy heuristic — plus a standard-CRP
(Caserta) mode.

## Setup
    py -3.12 -m venv .venv
    .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/Scripts/python -m pip install ortools xgboost numpy pyyaml pytest

## Verify
    .venv/Scripts/python -m pytest            # fast suite
    .venv/Scripts/python -m pytest -m slow    # one-time MIP/A* cross-check

## Training (SERVER ONLY — heavy)
    python experiments/train_ppo.py --full --out checkpoints/m0.pt --seed 0
    # 10 seeds for the ensemble, then:
    python experiments/table4_ppo_vs_astar.py --full --ckpt checkpoints/m0.pt
    python experiments/table8_ensemble.py --full --ckpt-dir checkpoints/

Every table of the paper maps to one script in `experiments/`.
All ambiguity resolutions vs. the paper: `REPRODUCTION_NOTES.md`.
```

- [ ] **Step 3: Full suite + commit**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all PASS.

```bash
git add README.md REPRODUCTION_NOTES.md
git commit -m "docs: README and reconciled reproduction notes"
```
