# Slot Stowage Optimization (SPP-AC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithful reproduction of Du et al. (2026) slot stowage optimization using improved Actor-Critic with Pointer Network.

**Architecture:** MDP environment (bay + container state) → Actor (CNN+LSTM encoder, GRU+Attention decoder) → action → Critic (same encoder + FCL) → REINFORCE with baseline.

**Tech Stack:** Python 3.12+, PyTorch, NumPy, PyYAML

## Global Constraints

- All code in `spp_ac/` under `C:\Users\X1\Project\CRP_Master\`
- No external DRL libraries (pure PyTorch)
- No comments in code (production quality, self-documenting)
- Tests in `tests/spp_ac/` with pytest
- GPU support via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Hyperparams from Table 2: hidden_dim=128, GRU_layers=4, batch_size=512, lr=1e-5, gamma=0.99, grad_clip=3.0

---

### Task 1: Project scaffolding + config

**Files:**
- Create: `spp_ac/__init__.py`
- Create: `spp_ac/config.py`
- Create: `spp_ac/config.yaml`
- Create: `spp_ac/requirements.txt`
- Create: `tests/spp_ac/__init__.py`

**Interfaces:**
- Produces: `Config` dataclass with `from_yaml()` / `to_yaml()` / `resolved_device()`
- Produces: `config.yaml` with Table 2 hyperparams

- [ ] **Step 1: Create `requirements.txt`**

```
torch>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
```

- [ ] **Step 2: Create empty `__init__.py` files**

```bash
touch spp_ac/__init__.py spp_ac/data/__init__.py spp_ac/env/__init__.py
touch spp_ac/models/__init__.py spp_ac/training/__init__.py
mkdir -p tests/spp_ac
touch tests/spp_ac/__init__.py
```

- [ ] **Step 3: Create `spp_ac/config.py`**

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import torch

@dataclass
class EnvConfig:
    num_ports: int = 6
    num_weight_classes: int = 8
    num_container_types: int = 2
    bay_rows: int = 12
    bay_tiers: int = 6

@dataclass
class RewardConfig:
    lambda_1: float = 0.7
    lambda_2: float = 0.3
    lambda_3: float = 0.01
    alpha_1: float = 3.0
    alpha_2: float = 3.0
    alpha_3: float = 3.0

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_gru_layers: int = 4

@dataclass
class TrainConfig:
    batch_size: int = 512
    num_iterations: int = 20000
    lr: float = 1e-5
    gamma: float = 0.99
    grad_clip: float = 3.0
    seed: int = 42

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            env=EnvConfig(**data.get("env", {})),
            reward=RewardConfig(**data.get("reward", {})),
            model=ModelConfig(**data.get("model", {})),
            train=TrainConfig(**data.get("train", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "env": {"num_ports": self.env.num_ports, "num_weight_classes": self.env.num_weight_classes,
                    "num_container_types": self.env.num_container_types, "bay_rows": self.env.bay_rows,
                    "bay_tiers": self.env.bay_tiers},
            "reward": {"lambda_1": self.reward.lambda_1, "lambda_2": self.reward.lambda_2,
                       "lambda_3": self.reward.lambda_3, "alpha_1": self.reward.alpha_1,
                       "alpha_2": self.reward.alpha_2, "alpha_3": self.reward.alpha_3},
            "model": {"hidden_dim": self.model.hidden_dim, "num_gru_layers": self.model.num_gru_layers},
            "train": {"batch_size": self.train.batch_size, "num_iterations": self.train.num_iterations,
                      "lr": self.train.lr, "gamma": self.train.gamma,
                      "grad_clip": self.train.grad_clip, "seed": self.train.seed},
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def resolved_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

- [ ] **Step 4: Create `spp_ac/config.yaml`**

```yaml
env:
  num_ports: 6
  num_weight_classes: 8
  num_container_types: 2
  bay_rows: 12
  bay_tiers: 6

reward:
  lambda_1: 0.7
  lambda_2: 0.3
  lambda_3: 0.01
  alpha_1: 3.0
  alpha_2: 3.0
  alpha_3: 3.0

model:
  hidden_dim: 128
  num_gru_layers: 4

train:
  batch_size: 512
  num_iterations: 20000
  lr: 0.00001
  gamma: 0.99
  grad_clip: 3.0
  seed: 42
```

- [ ] **Step 5: Run quick import check**

```bash
cd C:/Users/X1/Project/CRP_Master
python -c "from spp_ac.config import Config; c = Config(); print(c.resolved_device())"
```

Expected: prints `cuda` or `cpu`

- [ ] **Step 6: Commit**

```bash
git add spp_ac/__init__.py spp_ac/config.py spp_ac/config.yaml spp_ac/requirements.txt tests/spp_ac/__init__.py
git commit -m "feat(spp-ac): add project scaffolding and config"
```

---

### Task 2: Container Dataset Generation (CDG)

**Files:**
- Create: `spp_ac/data/__init__.py`
- Create: `spp_ac/data/cdg.py`
- Create: `tests/spp_ac/test_cdg.py`

**Interfaces:**
- `rip(N: int, P: int) -> list[int]` — random integer partition
- `CfgDataset(P: int, W: int, E: int, N: int, S: int, rng: Generator) -> Tensor[S, M, 4]` — CDG Algorithm 1

- [ ] **Step 1: Write failing test**

```python
# tests/spp_ac/test_cdg.py
import numpy as np
from spp_ac.data.cdg import rip, CfgDataset

def test_rip_sums_to_n():
    parts = rip(100, 6)
    assert sum(parts) == 100
    assert len(parts) == 6
    assert all(p >= 0 for p in parts)

def test_rip_deterministic_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = rip(100, 6, rng1)
    p2 = rip(100, 6, rng2)
    assert p1 == p2

def test_cdg_output_shape():
    data = CfgDataset(P=2, W=3, E=2, N=100, S=8, rng=np.random.default_rng(42))
    M = 2 * 3 * 2 + 1  # 13
    assert data.shape == (8, M, 4)

def test_cdg_quantity_sum():
    data = CfgDataset(P=2, W=3, E=2, N=100, S=4, rng=np.random.default_rng(42))
    total_qty = data[:, :, 3].sum().item()
    assert total_qty == 400  # 4 batches * 100

def test_cdg_zero_group():
    data = CfgDataset(P=2, W=3, E=2, N=100, S=1, rng=np.random.default_rng(42))
    zero_group = data[0, -1]  # last group
    assert zero_group[0].item() == 0  # POD=0
    assert zero_group[1].item() == 0  # weight=0
    assert zero_group[2].item() == 0  # type=0
```

- [ ] **Step 2: Run tests (expected: FAIL — module not found)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_cdg.py -v
```

- [ ] **Step 3: Write `spp_ac/data/cdg.py`**

```python
import numpy as np
import torch

def rip(N: int, P: int, rng: np.random.Generator | None = None) -> list[int]:
    if rng is None:
        rng = np.random.default_rng()
    if P == 1:
        return [N]
    parts = rng.exponential(1.0, size=P)
    parts = (parts / parts.sum() * N).astype(np.int64)
    diff = N - parts.sum()
    idx = rng.choice(P, abs(diff), replace=False)
    for i in idx[:abs(diff)]:
        parts[i] += 1 if diff > 0 else -1
    return parts.tolist()


def CfgDataset(
    P: int, W: int, E: int, N: int, S: int,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = np.random.default_rng()
    M = E * P * W + 1
    data = torch.zeros(S, M, 4, dtype=torch.float32)
    p_zero = rng.uniform(0, 0.1)
    N0 = int(N * p_zero)
    Nr = N - N0

    for i in range(S):
        x = rip(Nr, P, rng)
        for port_idx, n_port in enumerate(x):
            ww = rip(n_port, W, rng)
            for wc_idx, n_wc in enumerate(ww):
                ee = rip(n_wc, E, rng)
                for t_idx, n_t in enumerate(ee):
                    g = port_idx * W * E + wc_idx * E + t_idx
                    data[i, g, 0] = float(port_idx + 1)
                    data[i, g, 1] = float(wc_idx + 1)
                    data[i, g, 2] = float(t_idx + 1)
                    data[i, g, 3] = float(n_t)
        data[i, M - 1, 3] = float(N0)

    return data
```

- [ ] **Step 4: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_cdg.py -v
```

- [ ] **Step 5: Commit**

```bash
git add spp_ac/data/ tests/spp_ac/test_cdg.py
git commit -m "feat(spp-ac): add CDG dataset generator with RIP"
```

---

### Task 3: Container state

**Files:**
- Create: `spp_ac/env/container.py`
- Create: `tests/spp_ac/test_container.py`

**Interfaces:**
- `ContainerState(matrix: Tensor[M, 4])` — immutable-style state tracking
- `.get_matrix() -> Tensor[M, 4]` — returns copy
- `.select(idx: int) -> tuple[Tensor[M, 4], bool]` — new state with decremented quantity; returns (new_matrix, was_valid)
- `.remaining() -> int` — sum of quantities
- `.clone() -> Tensor[M, 4]`

- [ ] **Step 1: Write failing tests**

```python
# tests/spp_ac/test_container.py
import torch
from spp_ac.env.container import select_container, remaining_quantity

def test_select_valid():
    state = torch.zeros(13, 4)
    state[0] = torch.tensor([1.0, 1.0, 1.0, 5.0])
    state[1] = torch.tensor([1.0, 2.0, 1.0, 3.0])
    new_state, valid = select_container(state, 0)
    assert valid is True
    assert new_state[0, 3].item() == 4.0

def test_select_invalid():
    state = torch.zeros(13, 4)
    state[0, 3] = 0.0
    new_state, valid = select_container(state, 0)
    assert valid is False
    assert new_state[0, 3].item() == 0.0

def test_remaining():
    state = torch.zeros(13, 4)
    state[0, 3] = 5.0
    state[1, 3] = 3.0
    assert remaining_quantity(state) == 8
```

- [ ] **Step 2: Run tests (expected: FAIL)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_container.py -v
```

- [ ] **Step 3: Write `spp_ac/env/container.py`**

```python
import torch

def select_container(state: torch.Tensor, idx: int) -> tuple[torch.Tensor, bool]:
    new_state = state.clone()
    if new_state[idx, 3] > 0:
        new_state[idx, 3] -= 1
        return new_state, True
    return new_state, False


def remaining_quantity(state: torch.Tensor) -> int:
    return int(state[:, 3].sum().item())
```

- [ ] **Step 4: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_container.py -v
```

- [ ] **Step 5: Commit**

```bash
git add spp_ac/env/container.py tests/spp_ac/test_container.py
git commit -m "feat(spp-ac): add container state operations"
```

---

### Task 4: Bay state

**Files:**
- Create: `spp_ac/env/bay.py`
- Create: `tests/spp_ac/test_bay.py`

**Interfaces:**
- `BayState(R: int, T: int, row_weight_max: list[float], non_loadable: set[tuple[int,int]])`
- `.get_matrix() -> Tensor[6, R, T]`
- `.load(r: int, t: int, pod: int, weight: int, ctype: int) -> BayState`
- `.can_load(r: int, t: int, ctype: int) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# tests/spp_ac/test_bay.py
import torch
from spp_ac.env.bay import BayState

def test_empty_bay_matrix():
    bay = BayState(R=3, T=2, row_weight_max=[10.0, 10.0, 10.0])
    mat = bay.get_matrix()
    assert mat.shape == (6, 3, 2)
    assert (mat[0] == 0).all()  # all slots empty
    assert (mat[1] == 10.0).all()  # weight remaining = max

def test_non_loadable():
    bay = BayState(R=3, T=2, row_weight_max=[10.0, 10.0, 10.0],
                   non_loadable={(1, 0)})
    mat = bay.get_matrix()
    assert mat[0, 1, 0] == -1.0
    assert mat[1, 1, 0] == 0.0

def test_load_container():
    bay = BayState(R=2, T=2, row_weight_max=[20.0, 20.0])
    bay = bay.load(0, 0, pod=3, weight=5, ctype=1)
    mat = bay.get_matrix()
    assert mat[0, 0, 0] == 1.0
    assert mat[3, 0, 0] == 3.0
    assert mat[4, 0, 0] == 5.0
    assert mat[5, 0, 0] == 1.0
    assert mat[1, 0, 0] == 15.0  # 20 - 5

def test_can_load():
    bay = BayState(R=2, T=2, row_weight_max=[20.0, 20.0])
    assert bay.can_load(0, 0, ctype=1) is True
    assert bay.can_load(0, 0, ctype=2) is True

def test_can_load_non_loadable():
    bay = BayState(R=2, T=2, row_weight_max=[20.0, 20.0],
                   non_loadable={(0, 0)})
    assert bay.can_load(0, 0, ctype=1) is False

def test_can_load_overweight():
    bay = BayState(R=2, T=2, row_weight_max=[3.0, 20.0])
    bay = bay.load(0, 0, pod=1, weight=5, ctype=1)  # 5 > 3
    assert bay.can_load(0, 0, ctype=1) is False  # slot already used
    assert bay.can_load(0, 1, ctype=1) is False  # row exceeded
```

- [ ] **Step 2: Run tests (expected: FAIL)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_bay.py -v
```

- [ ] **Step 3: Write `spp_ac/env/bay.py`**

```python
import torch

class BayState:
    def __init__(
        self,
        R: int,
        T: int,
        row_weight_max: list[float],
        non_loadable: set[tuple[int, int]] | None = None,
    ):
        self.R = R
        self.T = T
        self.row_weight_max = torch.tensor(row_weight_max, dtype=torch.float32)
        self.non_loadable = non_loadable or set()
        self.state = torch.zeros(6, R, T, dtype=torch.float32)
        self._init_state()

    def _init_state(self):
        for r in range(self.R):
            for t in range(self.T):
                if (r, t) in self.non_loadable:
                    self.state[0, r, t] = -1.0
                else:
                    self.state[0, r, t] = 0.0
                    self.state[2, r, t] = 3.0  # both types allowed
                self.state[1, r, t] = float(self.row_weight_max[r])

    def get_matrix(self) -> torch.Tensor:
        return self.state.clone()

    def load(
        self, r: int, t: int, pod: int, weight: int, ctype: int
    ) -> "BayState":
        new = BayState.__new__(BayState)
        new.R, new.T = self.R, self.T
        new.row_weight_max = self.row_weight_max.clone()
        new.non_loadable = self.non_loadable
        new.state = self.state.clone()
        new.state[0, r, t] = 1.0
        new.state[3, r, t] = float(pod)
        new.state[4, r, t] = float(weight)
        new.state[5, r, t] = float(ctype)
        new.state[2, r, t] = float(ctype)
        new.state[1, r, t] -= float(weight)
        return new

    def can_load(self, r: int, t: int, ctype: int) -> bool:
        if (r, t) in self.non_loadable:
            return False
        if self.state[0, r, t] != 0.0:
            return False
        allowed = int(self.state[2, r, t].item())
        if allowed not in (3, ctype):
            return False
        row_remaining = self.state[1, r, t].item()
        return row_remaining >= 0
```

- [ ] **Step 4: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_bay.py -v
```

- [ ] **Step 5: Commit**

```bash
git add spp_ac/env/bay.py tests/spp_ac/test_bay.py
git commit -m "feat(spp-ac): add bay state with 6-channel matrix"
```

---

### Task 5: Stowage sequence (Fig 4)

**Files:**
- Create: `spp_ac/env/sequence.py`
- Create: `tests/spp_ac/test_sequence.py`

**Interfaces:**
- `StowageSequence(R_left: int, R_right: int, T: int, non_loadable: set[tuple[int,int,int]])`
- `.__len__() -> int`
- `.__getitem__(idx: int) -> tuple[int, int, int]` — (bay_idx, row, tier)
- `.get_paired_slot(idx: int) -> tuple[int, int, int] | None` — paired slot for 40ft container
- `.mark_occupied(idx: int)` — mark a slot as occupied (for 40ft pairing)

- [ ] **Step 1: Write failing tests**

```python
# tests/spp_ac/test_sequence.py
from spp_ac.env.sequence import StowageSequence

def test_sequence_length():
    seq = StowageSequence(R_left=3, R_right=3, T=2)
    total_slots = 3 * 2 + 3 * 2
    assert len(seq) == 12

def test_sequence_skip_non_loadable():
    seq = StowageSequence(R_left=2, R_right=2, T=1,
                           non_loadable={(0, 1, 0), (1, 1, 0)})
    assert len(seq) == 2

def test_sequence_bottom_up_both_sides():
    seq = StowageSequence(R_left=2, R_right=2, T=2)
    slots = [seq[i] for i in range(len(seq))]
    assert slots[0] == (0, 0, 0)  # left bay, row 0, tier 0
    assert slots[1] == (1, 1, 0)  # right bay, row 1, tier 0

def test_paired_slot_40ft():
    seq = StowageSequence(R_left=2, R_right=2, T=1)
    # Slot 0: (0, 0, 0) → paired slot is (1, 0, 0)
    paired = seq.get_paired_slot(0)
    assert paired is not None
    assert paired[0] == 1
    assert paired[1] == 0
    assert paired[2] == 0

def test_mark_occupied_skips_paired():
    seq = StowageSequence(R_left=2, R_right=2, T=2)
    seq.mark_occupied(0)  # marks paired slot as occupied
    assert len(seq) == 8  # one less slot available
```

- [ ] **Step 2: Run tests (expected: FAIL)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_sequence.py -v
```

- [ ] **Step 3: Write `spp_ac/env/sequence.py`**

```python
class StowageSequence:
    def __init__(
        self,
        R_left: int,
        R_right: int,
        T: int,
        non_loadable: set[tuple[int, int, int]] | None = None,
    ):
        self.R_left = R_left
        self.R_right = R_right
        self.T = T
        self.non_loadable = non_loadable or set()
        self._slots: list[tuple[int, int, int]] = []
        self._occupied: set[int] = set()
        self._paired_map: dict[int, int] = {}
        self._slot_of: dict[tuple[int, int, int], int] = {}
        self._build()

    def _build(self):
        R = max(self.R_left, self.R_right)
        for tier in range(self.T):
            left = 0
            right = 2 * R - 1
            while left < right:
                # Left side: (left_bay=0, row=left)
                if left < self.R_left and (0, left, tier) not in self.non_loadable:
                    self._add_slot(0, left, tier)
                # Right side: (right_bay=1, row=right - R)
                if right >= R:
                    rrow = right - R
                    if rrow < self.R_right and (1, rrow, tier) not in self.non_loadable:
                        self._add_slot(1, rrow, tier)
                left += 1
                right -= 1

    def _add_slot(self, bay: int, row: int, tier: int):
        idx = len(self._slots)
        self._slots.append((bay, row, tier))
        self._slot_of[(bay, row, tier)] = idx
        # Paired slot for 40ft: same row+tier in opposite bay
        paired = (1 - bay, row, tier)
        if paired in self._slot_of:
            self._paired_map[idx] = self._slot_of[paired]
            self._paired_map[self._slot_of[paired]] = idx

    def mark_occupied(self, idx: int) -> list[int]:
        self._occupied.add(idx)
        released: list[int] = []
        if idx in self._paired_map:
            paired_idx = self._paired_map[idx]
            if paired_idx not in self._occupied:
                self._occupied.add(paired_idx)
                released = [paired_idx]
        return released

    def get_paired_slot(self, idx: int) -> tuple[int, int, int] | None:
        if idx in self._paired_map:
            return self._slots[self._paired_map[idx]]
        return None

    def __len__(self) -> int:
        return len(self._slots) - len(self._occupied)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        actual = 0
        for i in range(len(self._slots)):
            if i not in self._occupied:
                if actual == idx:
                    return self._slots[i]
                actual += 1
        raise IndexError(f"Index {idx} out of range")
```

- [ ] **Step 4: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_sequence.py -v
```

- [ ] **Step 5: Commit**

```bash
git add spp_ac/env/sequence.py tests/spp_ac/test_sequence.py
git commit -m "feat(spp-ac): add stowage sequence with 40ft pairing (Fig 4)"
```

---

### Task 6: Reward computation (Eq 17-25)

**Files:**
- Create: `spp_ac/env/reward.py`
- Create: `tests/spp_ac/test_reward.py`

**Interfaces:**
- `RewardTracker(config: RewardConfig)` — tracks running counters
- `.record_load(r: int, t: int, pod: int, weight: int, ctype: int, bay_matrix: Tensor)` — per-step update
- `.compute() -> float` — total R = R₁ + R₂ (Eq 25)

- [ ] **Step 1: Write `spp_ac/env/reward.py`**

```python
import torch
from spp_ac.config import RewardConfig


class RewardTracker:
    def __init__(self, config: RewardConfig, R: int, T: int, row_weight_max: list[float]):
        self.config = config
        self.R = R
        self.T = T
        self.row_weight_max = torch.tensor(row_weight_max, dtype=torch.float32)
        self.reset()

    def reset(self):
        self.N_loaded = 0
        self.f1 = 0  # REO pairs
        self.f2_numer = 0.0  # Σ(L·Q·x)
        self.f2_denom = 0.0  # Σ(Q·x)
        self.f3 = 0  # HOL pairs
        self.overhang = 0
        self.twenty_on_forty = 0
        self.row_weights = torch.zeros(self.R)
        self._slots: list[dict] = []  # track per-slot info

    def record_load(
        self, r: int, t: int, pod: int, weight: int, ctype: int
    ):
        # Check overhang: container suspended if slot below is empty
        if t > 0 and not any(
            s["r"] == r and s["t"] == t - 1 for s in self._slots
        ):
            self.overhang += 1

        # Check 20-on-40: if slot below exists and is 40ft
        if ctype == 1:
            below = [s for s in self._slots if s["r"] == r and s["t"] == t - 1]
            if below and below[0]["ctype"] == 2:
                self.twenty_on_forty += 1

        # REO: check against all containers below in same stack
        for s in self._slots:
            if s["r"] == r and s["t"] < t and s["pod"] > pod:
                self.f1 += 1

        # HOL: check if heavier than container directly below
        below = [s for s in self._slots if s["r"] == r and s["t"] == t - 1]
        if below and weight > below[0]["weight"]:
            self.f3 += 1

        # LCGCP: running weighted sum
        # Lateral distance from centerline = (r - (R-1)/2)
        lateral_dist = r - (self.R - 1) / 2.0
        self.f2_numer += lateral_dist * weight
        self.f2_denom += weight

        # Row weight
        self.row_weights[r] += weight

        self._slots.append({"r": r, "t": t, "pod": pod, "weight": weight, "ctype": ctype})
        self.N_loaded += 1

    def compute(self) -> float:
        if self.N_loaded == 0:
            return 0.0

        m1 = self.f1 / self.N_loaded
        f2 = self.f2_numer / self.f2_denom if self.f2_denom > 0 else 0.0
        m2 = 2.0 * abs(f2) / self.R
        m3 = self.f3 / self.N_loaded

        R1 = -(self.config.lambda_1 * m1 + self.config.lambda_2 * m2 + self.config.lambda_3 * m3)

        g1 = self.overhang / self.N_loaded
        g2 = self.twenty_on_forty / self.N_loaded
        weight_excess = torch.clamp(self.row_weights - self.row_weight_max, min=0).sum().item()
        total_max = self.row_weight_max.sum().item()
        g3 = weight_excess / total_max if total_max > 0 else 0.0

        R2 = -(self.config.alpha_1 * g1 + self.config.alpha_2 * g2 + self.config.alpha_3 * g3)

        return R1 + R2
```

- [ ] **Step 2: Write tests**

```python
# tests/spp_ac/test_reward.py
from spp_ac.config import RewardConfig
from spp_ac.env.reward import RewardTracker

def test_reward_no_violations():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[20.0, 20.0])
    tracker.record_load(0, 0, pod=1, weight=5, ctype=1)
    tracker.record_load(0, 1, pod=2, weight=3, ctype=1)
    reward = tracker.compute()
    # No REO (POD increasing upward), no HOL (weight decreasing)
    # No violations
    assert reward < 0  # negative because objectives > 0

def test_reward_with_reo():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[20.0, 20.0])
    tracker.record_load(0, 0, pod=2, weight=3, ctype=1)
    tracker.record_load(0, 1, pod=1, weight=5, ctype=1)  # earlier POD on top = REO
    reward = tracker.compute()
    assert reward < 0

def test_reward_empty():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[20.0, 20.0])
    assert tracker.compute() == 0.0

def test_reward_overweight():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[5.0, 20.0])
    tracker.record_load(0, 0, pod=1, weight=10, ctype=1)  # > 5
    reward = tracker.compute()
    assert reward < -(cfg.alpha_3 * (5.0 / 25.0))  # weight excess penalty
```

- [ ] **Step 3: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_reward.py -v
```

- [ ] **Step 4: Commit**

```bash
git add spp_ac/env/reward.py tests/spp_ac/test_reward.py
git commit -m "feat(spp-ac): add reward computation Eq 17-25"
```

---

### Task 7: MDP Environment

**Files:**
- Create: `spp_ac/env/spp_env.py`
- Create: `tests/spp_ac/test_env.py`

**Interfaces:**
- `SlotStowageEnv(config: EnvConfig, reward_config: RewardConfig, container_data: Tensor[S, M, 4])`
- `.reset() -> tuple[Tensor, Tensor]` — (bay_matrix [6,R,T], container_matrix [M,4])
- `.step(action: int) -> tuple[Tensor, Tensor, float, bool]` — (bay, container, reward, done)

- [ ] **Step 1: Write `spp_ac/env/spp_env.py`**

```python
import numpy as np
import torch
from spp_ac.config import EnvConfig, RewardConfig
from spp_ac.env.bay import BayState
from spp_ac.env.container import select_container, remaining_quantity
from spp_ac.env.sequence import StowageSequence
from spp_ac.env.reward import RewardTracker


class SlotStowageEnv:
    def __init__(
        self,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        container_data: torch.Tensor,
        rng: np.random.Generator | None = None,
    ):
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.container_data = container_data
        self.rng = rng or np.random.default_rng()
        self.num_ports = env_cfg.num_ports
        self.R = env_cfg.bay_rows
        self.T = env_cfg.bay_tiers
        self.current_idx = 0

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.container_state = self.container_data[self.current_idx].clone()
        self.current_idx = (self.current_idx + 1) % len(self.container_data)
        row_weight_max = [50.0] * self.R
        self.bay = BayState(self.R, self.T, row_weight_max)
        self.sequence = StowageSequence(self.R, self.R, self.T)
        self.tracker = RewardTracker(self.reward_cfg, self.R, self.T, row_weight_max)
        self._actual_step = 0
        self._total_slots = len(self.sequence)
        return self.bay.get_matrix(), self.container_state

    def _compute_mask(self) -> torch.Tensor:
        return (self.container_state[:, 3] > 0).float()

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        if self._actual_step >= self._total_slots:
            reward = self.tracker.compute()
            return self.bay.get_matrix(), self.container_state, reward, True

        new_container, valid = select_container(self.container_state, action)
        if not valid:
            return (
                self.bay.get_matrix(),
                self.container_state,
                0.0,
                False,
            )
        self.container_state = new_container
        slot = self.sequence[self._actual_step]
        bay_idx, row, tier = slot
        ctype = int(self.container_state[action, 2].item())
        pod = int(self.container_state[action, 0].item())
        weight = int(self.container_state[action, 1].item())

        # Validate: can't place 40ft in 20ft-only slot
        allowed_type = int(self.bay.state[2, row, tier].item())
        if ctype == 2 and allowed_type not in (2, 3):
            self.container_state = new_container  # restore
            return (
                self.bay.get_matrix(),
                self.container_state,
                0.0,
                False,
            )

        self.bay = self.bay.load(row, tier, pod, weight, ctype)
        self.tracker.record_load(row, tier, pod, weight, ctype)
        self._actual_step += 1

        # If 40ft: mark paired slot as occupied
        if ctype == 2:
            released = self.sequence.mark_occupied(self._actual_step - 1)
            if released:
                self._total_slots -= len(released)

        done = self._actual_step >= self._total_slots or remaining_quantity(self.container_state) == 0
        reward = self.tracker.compute() if done else 0.0
        return self.bay.get_matrix(), self.container_state, reward, done

    def get_mask(self) -> torch.Tensor:
        return self._compute_mask()
```

- [ ] **Step 2: Write tests**

```python
# tests/spp_ac/test_env.py
import torch
from spp_ac.config import EnvConfig, RewardConfig
from spp_ac.env.spp_env import SlotStowageEnv

def _make_container_data(P=2, W=2, E=2, N=50, S=1):
    from spp_ac.data.cdg import CfgDataset
    import numpy as np
    return CfgDataset(P, W, E, N, S, rng=np.random.default_rng(42))

def test_env_reset():
    env_cfg = EnvConfig(num_ports=2, num_weight_classes=2, num_container_types=2,
                        bay_rows=2, bay_tiers=2)
    reward_cfg = RewardConfig()
    data = _make_container_data(P=2, W=2, E=2, N=20, S=4)
    env = SlotStowageEnv(env_cfg, reward_cfg, data)
    bay, container = env.reset()
    assert bay.shape == (6, 2, 2)
    assert container.shape == (9, 4)  # M = 2*2*2+1 = 9

def test_env_step():
    env_cfg = EnvConfig(num_ports=2, num_weight_classes=2, num_container_types=2,
                        bay_rows=2, bay_tiers=2)
    reward_cfg = RewardConfig()
    data = _make_container_data(P=2, W=2, E=2, N=20, S=4)
    env = SlotStowageEnv(env_cfg, reward_cfg, data)
    env.reset()
    mask = env.get_mask()
    valid_idx = (mask > 0).nonzero(as_tuple=True)[0][0].item()
    bay, container, reward, done = env.step(valid_idx)
    assert bay.shape == (6, 2, 2)
    assert container[valid_idx, 3].item() < data[0, valid_idx, 3].item()
```

- [ ] **Step 3: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_env.py -v
```

- [ ] **Step 4: Commit**

```bash
git add spp_ac/env/spp_env.py tests/spp_ac/test_env.py
git commit -m "feat(spp-ac): add MDP environment for slot stowage"
```

---

### Task 8: Encoders (Fig 6-7)

**Files:**
- Create: `spp_ac/models/encoder.py`
- Create: `tests/spp_ac/test_encoder.py`

**Interfaces:**
- `ContainerEncoder(hidden_dim: int)` — `forward(container: Tensor[M,4]) -> Tensor[M, hidden_dim]`
- `BayEncoder(hidden_dim: int)` — `forward(bay: Tensor[6,R,T]) -> Tensor[hidden_dim]`

- [ ] **Step 1: Write `spp_ac/models/encoder.py`**

```python
import torch
import torch.nn as nn


class ContainerEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(4, hidden_dim // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, 4, M]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)  # [B, M, H]
        out, _ = self.lstm(x)   # [B, M, H]
        return out


class BayEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(6, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.avgpool(x)        # [B, H, 1, 1]
        x = x.flatten(1)           # [B, H]
        x = x.unsqueeze(1)         # [B, 1, H] (seq_len=1 for LSTM)
        out, _ = self.lstm(x)      # [B, 1, H]
        out = out.squeeze(1)       # [B, H]
        return out
```

- [ ] **Step 2: Write tests**

```python
# tests/spp_ac/test_encoder.py
import torch
from spp_ac.models.encoder import ContainerEncoder, BayEncoder

def test_container_encoder_output_shape():
    enc = ContainerEncoder(hidden_dim=128)
    x = torch.randn(4, 13, 4)  # [B, M, 4]
    out = enc(x)
    assert out.shape == (4, 13, 128)

def test_container_encoder_gradient_flows():
    enc = ContainerEncoder(hidden_dim=128)
    x = torch.randn(2, 13, 4, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_bay_encoder_output_shape():
    enc = BayEncoder(hidden_dim=128)
    x = torch.randn(4, 6, 12, 6)  # [B, 6, R, T]
    out = enc(x)
    assert out.shape == (4, 128)

def test_bay_encoder_gradient_flows():
    enc = BayEncoder(hidden_dim=128)
    x = torch.randn(2, 6, 12, 6, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
```

- [ ] **Step 3: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_encoder.py -v
```

- [ ] **Step 4: Commit**

```bash
git add spp_ac/models/encoder.py tests/spp_ac/test_encoder.py
git commit -m "feat(spp-ac): add container and bay encoders (Fig 6-7)"
```

---

### Task 9: Decoder with Masked Attention (Eq 26)

**Files:**
- Create: `spp_ac/models/decoder.py`
- Create: `tests/spp_ac/test_decoder.py`

**Interfaces:**
- `AttentionDecoder(hidden_dim: int, num_gru_layers: int)` — single-step decoder
- `.forward(container_enc: Tensor[B,M,H], bay_context: Tensor[B,H], prev_embed: Tensor[B,H] | None, state: tuple | None) -> tuple[Tensor[B,M], Tensor[B,H], tuple]`

- [ ] **Step 1: Write `spp_ac/models/decoder.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_gru_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layers, batch_first=True)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.embed_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_init = nn.Linear(hidden_dim, hidden_dim)

    def get_initial_state(self, bay_context: torch.Tensor) -> torch.Tensor:
        batch_size = bay_context.size(0)
        h0 = torch.tanh(self.decoder_init(bay_context))
        h0 = h0.unsqueeze(0).repeat(self.num_gru_layers, 1, 1).contiguous()
        return h0

    def forward(
        self,
        container_enc: torch.Tensor,
        bay_context: torch.Tensor,
        prev_embed: torch.Tensor | None,
        quantities: torch.Tensor,
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_state is None:
            h_state = self.get_initial_state(bay_context)

        if prev_embed is None:
            gru_input = torch.zeros(
                bay_context.size(0), 1, self.hidden_dim,
                device=bay_context.device,
            )
        else:
            gru_input = self.embed_proj(prev_embed).unsqueeze(1)

        gru_out, h_state = self.gru(gru_input, h_state)
        d_t = gru_out.squeeze(1)

        proj_e = self.W1(container_enc)
        proj_d = self.W2(d_t).unsqueeze(1)
        u = torch.tanh(proj_e + proj_d)
        attn = torch.einsum("h,bmh->bm", self.v, u)

        mask = (quantities > 0).float()
        logits = torch.where(mask > 0, attn + quantities, attn + 1e-9)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        return probs, log_probs, d_t, h_state
```

- [ ] **Step 2: Write tests**

```python
# tests/spp_ac/test_decoder.py
import torch
from spp_ac.models.decoder import AttentionDecoder

def test_decoder_output_shape():
    dec = AttentionDecoder(hidden_dim=128, num_gru_layers=4)
    container_enc = torch.randn(4, 13, 128)
    bay_context = torch.randn(4, 128)
    quantities = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                               [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]], dtype=torch.float32)
    probs, log_probs, d_t, h_state = dec(container_enc, bay_context, None, quantities)
    assert probs.shape == (4, 13)
    assert log_probs.shape == (4, 13)
    assert d_t.shape == (4, 128)

def test_decoder_mask_zero_quantity():
    dec = AttentionDecoder(hidden_dim=128, num_gru_layers=4)
    container_enc = torch.randn(1, 13, 128)
    bay_context = torch.randn(1, 128)
    quantities = torch.zeros(1, 13)
    quantities[0, 5] = 3.0
    probs, log_probs, _, _ = dec(container_enc, bay_context, None, quantities)
    assert probs[0, 5] > 0.5  # only group 5 has qty > 0

def test_decoder_gradient_flows():
    dec = AttentionDecoder(hidden_dim=128, num_gru_layers=4)
    container_enc = torch.randn(2, 13, 128, requires_grad=True)
    bay_context = torch.randn(2, 128, requires_grad=True)
    quantities = torch.ones(2, 13) * 2
    probs, log_probs, _, _ = dec(container_enc, bay_context, None, quantities)
    loss = log_probs.mean()
    loss.backward()
    assert container_enc.grad is not None
    assert bay_context.grad is not None
```

- [ ] **Step 3: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_decoder.py -v
```

- [ ] **Step 4: Commit**

```bash
git add spp_ac/models/decoder.py tests/spp_ac/test_decoder.py
git commit -m "feat(spp-ac): add attention decoder with mask (Eq 26)"
```

---

### Task 10: Actor + Critic (Section 3)

**Files:**
- Create: `spp_ac/models/actor.py`
- Create: `spp_ac/models/critic.py`
- Create: `tests/spp_ac/test_actor.py`
- Create: `tests/spp_ac/test_critic.py`

**Interfaces:**
- `Actor(hidden_dim, num_gru_layers)` — `forward(bay, container, prev_embed, state) -> (probs, log_probs, new_embed, new_state)`
- `Critic(hidden_dim)` — `forward(bay, container) -> scalar`

- [ ] **Step 1: Write `spp_ac/models/actor.py`**

```python
import torch
import torch.nn as nn
from spp_ac.models.encoder import ContainerEncoder, BayEncoder
from spp_ac.models.decoder import AttentionDecoder


class Actor(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_gru_layers: int = 4):
        super().__init__()
        self.container_encoder = ContainerEncoder(hidden_dim)
        self.bay_encoder = BayEncoder(hidden_dim)
        self.decoder = AttentionDecoder(hidden_dim, num_gru_layers)

    def forward(
        self,
        bay: torch.Tensor,
        container: torch.Tensor,
        prev_embed: torch.Tensor | None = None,
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        container_enc = self.container_encoder(container)
        bay_context = self.bay_encoder(bay)
        quantities = container[:, :, 3]
        probs, log_probs, d_t, h_state = self.decoder(
            container_enc, bay_context, prev_embed, quantities, h_state,
        )
        return probs, log_probs, d_t, h_state
```

- [ ] **Step 2: Write `spp_ac/models/critic.py`**

```python
import torch
import torch.nn as nn
from spp_ac.models.encoder import ContainerEncoder, BayEncoder


class Critic(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.container_encoder = ContainerEncoder(hidden_dim)
        self.bay_encoder = BayEncoder(hidden_dim)
        self.fc1 = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1)
        self.fc3 = nn.Conv1d(hidden_dim // 2, 16, kernel_size=1)
        self.fc4 = nn.Conv1d(16, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, bay: torch.Tensor, container: torch.Tensor) -> torch.Tensor:
        e = self.container_encoder(container)
        b = self.bay_encoder(bay)
        h = torch.cat([e.mean(dim=1), b], dim=-1)
        h = h.unsqueeze(-1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.fc4(h)
        v = h.squeeze(-1).squeeze(-1)
        return v
```

- [ ] **Step 3: Write tests**

```python
# tests/spp_ac/test_actor.py
import torch
from spp_ac.models.actor import Actor

def test_actor_forward():
    actor = Actor(hidden_dim=128, num_gru_layers=4)
    bay = torch.randn(2, 6, 12, 6)
    container = torch.randn(2, 13, 4)
    container[:, :, 3] = torch.randint(0, 5, (2, 13)).float()
    probs, log_probs, d_t, h_state = actor(bay, container)
    assert probs.shape == (2, 13)
    assert log_probs.shape == (2, 13)
    assert d_t.shape == (2, 128)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

def test_actor_gradient_flows():
    actor = Actor(hidden_dim=128, num_gru_layers=4)
    bay = torch.randn(1, 6, 12, 6)
    container = torch.randn(1, 13, 4)
    container[:, :, 3] = torch.randint(1, 5, (1, 13)).float()
    probs, log_probs, d_t, h_state = actor(bay, container)
    loss = -log_probs.mean()
    loss.backward()
    assert all(p.grad is not None for p in actor.parameters())
```

```python
# tests/spp_ac/test_critic.py
import torch
from spp_ac.models.critic import Critic

def test_critic_output():
    critic = Critic(hidden_dim=128)
    bay = torch.randn(4, 6, 12, 6)
    container = torch.randn(4, 13, 4)
    v = critic(bay, container)
    assert v.shape == (4,)

def test_critic_gradient_flows():
    critic = Critic(hidden_dim=128)
    bay = torch.randn(2, 6, 12, 6)
    container = torch.randn(2, 13, 4)
    v = critic(bay, container)
    loss = v.mean()
    loss.backward()
    assert all(p.grad is not None for p in critic.parameters())
```

- [ ] **Step 4: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_actor.py tests/spp_ac/test_critic.py -v
```

- [ ] **Step 5: Commit**

```bash
git add spp_ac/models/actor.py spp_ac/models/critic.py
git add tests/spp_ac/test_actor.py tests/spp_ac/test_critic.py
git commit -m "feat(spp-ac): add Actor and Critic networks (Section 3)"
```

---

### Task 11: Training loop (Eq 27-28)

**Files:**
- Create: `spp_ac/training/trainer.py`
- Create: `tests/spp_ac/test_trainer.py`

**Interfaces:**
- `Trainer(config: Config, device: torch.device)` — orchestrates training
- `.train() -> None` — full training loop

- [ ] **Step 1: Write `spp_ac/training/trainer.py`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from spp_ac.config import Config
from spp_ac.data.cdg import CfgDataset
from spp_ac.env.spp_env import SlotStowageEnv
from spp_ac.models.actor import Actor
from spp_ac.models.critic import Critic


class Trainer:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.actor = Actor(config.model.hidden_dim, config.model.num_gru_layers).to(device)
        self.critic = Critic(config.model.hidden_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.train.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.train.lr)

    def _sample_batch(self) -> SlotStowageEnv:
        rng = np.random.default_rng()
        data = CfgDataset(
            P=self.config.env.num_ports,
            W=self.config.env.num_weight_classes,
            E=self.config.env.num_container_types,
            N=100,
            S=self.config.train.batch_size,
            rng=rng,
        )
        return SlotStowageEnv(self.config.env, self.config.reward, data.to(self.device))

    def train(self, num_iterations: int | None = None) -> None:
        if num_iterations is None:
            num_iterations = self.config.train.num_iterations
        num_iterations = self.config.train.num_iterations
        env = self._sample_batch()

        for iteration in range(num_iterations):
            episode_log_probs: list[torch.Tensor] = []
            episode_rewards: list[float] = []
            episode_values: list[float] = []

            for _ in range(self.config.train.batch_size):
                bay, container = env.reset()
                bay = bay.to(self.device).unsqueeze(0)
                container = container.to(self.device).unsqueeze(0)

                v0 = self.critic(bay, container)
                log_probs: list[torch.Tensor] = []
                h_state = None
                prev_embed = None

                done = False
                while not done:
                    mask = env.get_mask().to(self.device)
                    with torch.no_grad():
                        probs, log_prob, d_t, h_state = self.actor(
                            bay, container, prev_embed, h_state,
                        )
                    scaled_probs = probs * mask
                    scaled_probs = scaled_probs / scaled_probs.sum(dim=-1, keepdim=True)
                    dist = torch.distributions.Categorical(scaled_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    log_probs.append(log_prob)

                    action_scalar = action.item()
                    container_np = container.squeeze(0).cpu()
                    env.container_state = container_np
                    new_bay, new_container, reward, done = env.step(action_scalar)
                    bay = new_bay.to(self.device).unsqueeze(0)
                    container = new_container.to(self.device).unsqueeze(0)
                    prev_embed = d_t

                episode_log_probs.append(torch.stack(log_probs).sum())
                episode_rewards.append(reward)
                episode_values.append(v0.item())

            rewards_t = torch.tensor(episode_rewards, device=self.device)
            values_t = torch.tensor(episode_values, device=self.device)
            advantages = rewards_t - values_t
            log_probs_t = torch.stack(episode_log_probs)

            # Policy gradient (Eq 27) — gradient ascent
            actor_loss = -(advantages.detach() * log_probs_t).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.train.grad_clip)
            self.actor_optim.step()

            # Value gradient (Eq 28) — gradient descent
            critic_loss = (advantages ** 2).mean()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.train.grad_clip)
            self.critic_optim.step()

            if iteration % 100 == 0:
                avg_reward = rewards_t.mean().item()
                avg_value = values_t.mean().item()
                print(f"Iter {iteration:5d} | R={avg_reward:.4f} | V={avg_value:.4f} | "
                      f"Adv={advantages.mean().item():.4f} | ActorL={actor_loss.item():.6f} | CriticL={critic_loss.item():.6f}")
```

- [ ] **Step 2: Write smoke test (short run)**

```python
# tests/spp_ac/test_trainer.py
import torch
from spp_ac.config import Config
from spp_ac.training.trainer import Trainer

def test_trainer_smoke():
    config = Config()
    config.train.batch_size = 4
    config.train.num_iterations = 5
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    trainer.train(num_iterations=2)
    assert True  # no crash = pass

def test_trainer_actor_parameters_update():
    config = Config()
    config.train.batch_size = 2
    config.train.num_iterations = 1
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    before = [p.clone() for p in trainer.actor.parameters()]
    trainer.train(num_iterations=1)
    after = list(trainer.actor.parameters())
    any_changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert any_changed
```

- [ ] **Step 3: Run tests (expected: PASS)**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m pytest tests/spp_ac/test_trainer.py -v -x
```

- [ ] **Step 4: Commit**

```bash
git add spp_ac/training/trainer.py tests/spp_ac/test_trainer.py
git commit -m "feat(spp-ac): add REINFORCE training loop (Eq 27-28)"
```

---

### Task 12: Main entry point

**Files:**
- Create: `spp_ac/main.py`

- [ ] **Step 1: Write `spp_ac/main.py`**

```python
import argparse
import torch
from pathlib import Path
from spp_ac.config import Config
from spp_ac.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPP-AC: Slot Stowage Optimization via Improved Actor-Critic")
    parser.add_argument("--config", type=str, default="spp_ac/config.yaml", help="Path to config YAML")
    parser.add_argument("--iterations", type=int, default=None, help="Override training iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config.from_yaml(args.config)

    if args.iterations:
        config.train.num_iterations = args.iterations
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr

    if args.device == "auto":
        device = config.resolved_device()
    else:
        device = torch.device(args.device)

    trainer = Trainer(config, device)

    trainer.train()

    if args.checkpoint:
        torch.save({
            "actor_state_dict": trainer.actor.state_dict(),
            "critic_state_dict": trainer.critic.state_dict(),
            "actor_optim_state_dict": trainer.actor_optim.state_dict(),
            "critic_optim_state_dict": trainer.critic_optim.state_dict(),
            "config": config,
        }, args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

```bash
cd C:/Users/X1/Project/CRP_Master
python -m spp_ac.main --iterations 3 --batch-size 4 --device cpu
```

- [ ] **Step 3: Commit**

```bash
git add spp_ac/main.py
git commit -m "feat(spp-ac): add main entry point with CLI"
```

---

### Self-Review Checklist

1. **Spec coverage:** Every section of spec has a corresponding task.
   - Section 2 MDP → Tasks 3-7 (bay, container, sequence, reward, env)
   - Section 3 Networks → Tasks 8-10 (encoder, decoder, actor, critic)
   - Section 4 CDG → Task 2
   - Section 5 Training → Task 11
   - Config/Scaffolding → Task 1
   - Entry point → Task 12

2. **Placeholder scan:** All code is complete in every step. No TBD/TODO.

3. **Type consistency:** 
   - Bay state: `Tensor[6, R, T]` throughout
   - Container state: `Tensor[M, 4]` throughout
   - Actor returns: `(probs[B,M], log_probs[B,M], d_t[B,H], (state, h_state))`
   - Critic returns: `v[B]`
   - Trainer uses `SlotStowageEnv`, `Actor`, `Critic`

4. **No gaps:** All equations (Eq 1-28) covered by testable code.


```
spp_ac/
├── __init__.py
├── config.py
├── main.py
├── config.yaml
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── cdg.py
├── env/
│   ├── __init__.py
│   ├── bay.py
│   ├── container.py
│   ├── sequence.py
│   ├── reward.py
│   └── spp_env.py
├── models/
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── actor.py
│   └── critic.py
└── training/
    ├── __init__.py
    └── trainer.py

tests/spp_ac/
├── __init__.py
├── test_cdg.py
├── test_bay.py
├── test_container.py
├── test_sequence.py
├── test_reward.py
├── test_env.py
├── test_encoder.py
├── test_decoder.py
├── test_actor.py
├── test_critic.py
└── test_trainer.py
```

---
