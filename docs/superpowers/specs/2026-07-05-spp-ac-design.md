# Slot Stowage Optimization via Improved Actor-Critic (Du et al. 2026)

**Spec**: Faithful reproduction of Du et al., *An automated slot stowage optimization method for container ship based on improved Actor-Critic algorithm*, Ocean Engineering 350 (2026) 124255.

**Project location**: `C:\Users\X1\Project\CRP_Master\spp_ac\`

---

## 1. Problem Statement

Slot Planning Problem (SPP): Assign containers to individual slots within a bay of a container ship, handling mixed 20-ft and 40-ft containers under seaworthiness constraints.

### Objectives (minimize weighted sum, Eq 1)

| Term | Metric | Eq | Description |
|---|---|---|---|
| `f₁` | REO | (2) | Restow operation count (pairs where upper container discharges earlier than lower) |
| `f₂` | LCGCP | (3) | Lateral offset of cargo CoG from centerline plane |
| `f₃` | HOL | (4) | Heavy-over-light violations (upper heavier than lower) |

### Constraints

| # | Constraint | Eq |
|---|---|---|
| C1 | Each container at most once | (5) |
| C2 | Slot occupancy consistency | (6) |
| C3 | No overhang (containers must be supported) | (7) |
| C4 | No 20-ft on top of 40-ft | (8)-(9) |
| C5 | Container type consistency | (10) |
| C6 | Row weight limit ≤ Wmax_r | (11) |
| C7 | 40-ft spans two adjacent bays | (12) |
| C8 | At most one container per slot | (13) |

### Assumptions (Section 2)

1. Vessel only discharges containers — no loading at intermediate ports
2. Standard 20-ft and 40-ft dry containers only

---

## 2. MDP Formulation (Section 2.2)

### State

**Bay State** — tensor `[6, R, T]`:

| Ch | Feature | Non-loadable | Empty | Loaded |
|---|---|---|---|---|
| 0 | Loadable flag | -1 | 0 | 1 |
| 1 | Row weight remaining (broadcast) | 0 | Wmax_r | Wmax_r − ΣQ |
| 2 | Allowed container types | 0 | 3 (both) | as loaded |
| 3 | POD of loaded container | 0 | 0 | 1..P |
| 4 | Weight class of loaded container | 0 | 0 | 1..W |
| 5 | Type of loaded container | 0 | 0 | 1..2 |

Non-loadable slots (lashing bridges, hatches, structural): channel 0 = -1, channels 1-5 = 0.

**Container State** — tensor `[M, 4]`, M = E·P·W + 1 = 97:

| Col | Feature | Values |
|---|---|---|
| 0 | POD | 1..P, 0 for zero group |
| 1 | Weight class | 1..W, 0 for zero group |
| 2 | Container type | 1=20ft, 2=40ft, 0 for zero group |
| 3 | Remaining quantity | ≥ 0 integer |

Last group (index M-1) = zero-container placeholder (partial loading).

### Action

Select a container group index `a ∈ {0..M-1}` for the current slot.

- If qty[a] > 0: valid. Place container, decrement qty, update bay.
- If qty[a] = 0: mask prevents selection (probability ≈ 0).
- 40-ft container selected: mark paired slot in adjacent bay with same sequence number.
- 40-ft selected in a slot that only accepts 20-ft: **rejected**, returns to pool, agent selects again.

### Stowage Sequence (Fig 4)

Two adjacent 20-ft bays (forming a 40-ft bay pair). Merged horizontally. Loading order:

```
for tier in 0..T-1:
  left = 0, right = 2R-1
  while left < right:
    if slot(left, tier) loadable → assign sequence number
    if slot(right, tier) loadable → assign sequence number
    left++, right--
```

40-ft containers occupy mirrored slots in both virtual bays (same sequence number).

### Reward (Eq 17-25)

R = R₁ + R₂

**R₁ (objectives)** — computed at episode end:

```
m₁ = f₁ / N_loaded                      Eq 17
m₂ = 2·|f₂| / |R|                      Eq 18
m₃ = f₃ / N_loaded                      Eq 19
R₁ = -(λ₁·m₁ + λ₂·m₂ + λ₃·m₃)          Eq 20
```

**R₂ (constraint penalty)**:

```
g₁ = overhang_violations / N_loaded     Eq 21
g₂ = 20TOP40_violations / N_loaded      Eq 22
g₃ = weight_excess / ΣWmax_r            Eq 23
R₂ = -(α₁·g₁ + α₂·g₂ + α₃·g₃)          Eq 24
```

**Note on α signs**: Eq 24 shows R₂ = -(αg). Table 4 lists α = -3. To produce a negative penalty on violation (as paper describes), interpret α = +3 so R₂ = -3·g.

---

## 3. Network Architecture (Section 3)

### 3.1 Container Encoder (Fig 6)

```
Input: [M, 4]
  → Permute(1,0): [4, M]
  → Conv1d(4→64, k=1) + ReLU
  → Conv1d(64→128, k=1) + ReLU
  → Conv1d(128→128, k=1) + ReLU
  → Permute(1,0): [M, 128]
  → LSTM(128→128, batch_first): output all steps → [M, 128]
Output: per-group encoding eⱼ ∈ ℝ¹²⁸
```

### 3.2 Bay Encoder (Fig 7)

```
Input: [6, R, T]
  → Conv2d(6→64, k=3, pad=1) + ReLU
  → MaxPool2d(3, stride=2)
  → Conv2d(64→128, k=3, pad=1) + ReLU
  → MaxPool2d(3, stride=2)
  → Conv2d(128→128, k=3, pad=1) + ReLU
  → AdaptiveAvgPool2d(1): [128, 1, 1]
  → Flatten: [128]
  → LSTM(128→128, batch_first, seq_len=1): [128]
Output: bay context vector b ∈ ℝ¹²⁸
```

### 3.3 Decoder (GRU + Masked Attention, Section 3.2.2-3.2.3)

Initial state: `d₀ = tanh(W·b)`

At step t:
1. GRU: `dₜ = GRU(dₜ₋₁, embed(aₜ₋₁))`
2. Attention: `uⱼᵗ = vᵀ · tanh(W₁·eⱼ + W₂·dₜ)` for j=0..M-1
3. Mask (Eq 26):
   ```
   logitⱼ =
     uⱼᵗ + qtyⱼ      if qtyⱼ > 0
     uⱼᵗ + 1e-9       if qtyⱼ = 0
   ```
4. `P(aₜ=j | ...) = softmax(logitⱼ)`
5. Sample aₜ ~ P(·)
6. Store log P(aₜ)

### 3.4 Policy Network (Actor)

```
class Actor(nn.Module):
    container_encoder: ContainerEncoder
    bay_encoder: BayEncoder
    decoder: Decoder  # GRU + MaskedAttention

    def forward(self, bay: Tensor, container: Tensor,
                prev_embed: Tensor | None, state: Tensor | None)
      → (log_probs: Tensor[M], new_emb: Tensor, new_state: Tensor)
```

### 3.5 Value Network (Critic, Section 3.3)

Same encoder architecture (separate weights). Aggregation:

```
e = container_encoder(container)   # [M, 128]
b = bay_encoder(bay)               # [128]
h = cat(e.mean(0), b)              # [256]
h = h.unsqueeze(-1)                # [256, 1]

h = Conv1d(256→128, k=1) + ReLU   # [128, 1]
h = Conv1d(128→64, k=1)  + ReLU   # [64, 1]
h = Conv1d(64→16, k=1)   + ReLU   # [16, 1]
h = Conv1d(16→1, k=1)             # [1, 1]
v = h.squeeze(-1)                  # scalar

Output: V(s) ∈ ℝ
```

---

## 4. Dataset Generation (Section 4.1)

### RIP — Random Integer Partition

Generate P non-negative integers summing to N.

```
method: exponential(1, P) → normalize → integerize → residual correction
```

### CDG — Container Dataset Generation (Algorithm 1)

```
Input:  P=6, N, W=8, E=2, S (batch size)

1. p ~ Uniform(0, 0.1)
2. N₀ = N · p,  Nᵣ = N − N₀
3. data = zeros(S, M, 4),  M = E·P·W + 1

4. for i in range(S):
     x = RIP(Nᵣ, P)            # containers per port
     for port_idx in range(P):
       ww = RIP(x[port_idx], W) # containers per weight class
       for wc_idx in range(W):
         ee = RIP(ww[wc_idx], E) # per container type
         for t_idx in range(E):
           g = port_idx·W·E + wc_idx·E + t_idx
           data[i, g] = [port_idx+1, wc_idx+1, t_idx+1, ee[t_idx]]

5. data[:, M-1] = [0, 0, 0, N₀]   # zero-container group (last)
6. return data                    # shape [S, M, 4]
```

---

## 5. Training Procedure (Section 3.4, 4.2, Algorithm 1)

### REINFORCE with Baseline

Per iteration:

1. **Sample**: Generate K = 512 container instances via CDG. For each k:
   - Reset env → observe S₀
   - Critic: `Vₖ = V(S₀; φ)`
   - For t = 0..T-1 (stowage sequence):
     - Actor: `π(aₜ | sₜ)`, log π stored
     - Sample aₜ, step env → sₜ₊₁
   - Episode done → compute total Rₖ (Eq 25)
   - Store: `{log π sequence; Rₖ; Vₖ}`

2. **Policy gradient** (Eq 27):
   ```
   Aₖ = Rₖ − Vₖ                           # advantage
   L_A = −1/K · Σₖ Aₖ · (Σₜ log π(aₜ|sₜ))  # REINFORCE
   ∇_θ L_A → Adam step (ascent)
   ```

3. **Value gradient** (Eq 28):
   ```
   L_V = 1/K · Σₖ (Rₖ − Vₖ)²             # MSE
   ∇_φ L_V → Adam step (descent)
   ```

4. Gradient clipping: max norm = 3.0

### Hyperparameters (Table 2)

| Param | Value |
|---|---|
| Hidden dimension | 128 |
| GRU layers | 4 |
| Batch size (K) | 512 |
| Iterations | 20,000 |
| Learning rate | 1e-5 |
| Discount | 0.99 |
| Gradient clip | 3.0 |

### Reward Coefficients (Table 4, Set A)

| Param | Value |
|---|---|
| λ₁ (REO) | 0.7 |
| λ₂ (LCGCP) | 0.3 |
| λ₃ (HOL) | 0.01 |
| α₁ (overhang) | 3 |
| α₂ (20TOP40) | 3 |
| α₃ (weight) | 3 |

---

## 6. Project Structure

```
spp_ac/
├── __init__.py
├── config.py                    # Config dataclass + YAML I/O
├── config.yaml                  # Hyperparameters (Table 2, 4)
├── main.py                      # Entry point: train/test
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   └── cdg.py                   # RIP + CDG (Algorithm 1)
│
├── env/
│   ├── __init__.py
│   ├── bay.py                   # BayState [6, R, T]
│   ├── container.py             # ContainerState [M, 4]
│   ├── sequence.py              # StowageSequence (Fig 4)
│   ├── reward.py                # Eq 17-25
│   └── spp_env.py               # SlotStowageEnv (MDP)
│
├── models/
│   ├── __init__.py
│   ├── encoder.py               # ContainerEncoder + BayEncoder (Fig 6-7)
│   ├── decoder.py               # GRU + MaskedAttention (Eq 26)
│   ├── actor.py                 # Actor network
│   └── critic.py                # Critic network
│
└── training/
    ├── __init__.py
    └── trainer.py               # REINFORCE + baseline (Eq 27-28)
```

---

## 7. Resolved Ambiguities

| Issue | Resolution |
|---|---|
| Eq 24 vs Table 4 (α negative in table) | α interpreted as positive magnitude; R₂ = -(α·g) produces correct penalty |
| Zero-group position | Last group (index M-1), per CDG algorithm (not "g1" as text suggests) |
| Reward timing | Running counters per-step; total R computed at episode end |
| Critic weight sharing | Separate weights from Actor (same architecture) |
| 40-ft in 20-ft slot | Rejected, container returned to pool, re-sample |
| Bay encoder channels | 6→64→128→128 (pattern consistent with container encoder) |

---

## 8. Verification Plan

| Check | Method |
|---|---|
| CDG correctness | Sum quantities = N for each batch; zero-group has N₀ |
| Stowage sequence | Visual inspection of sequence array matches Fig 4 pattern |
| Action mask | Groups with qty=0 have ~0 probability |
| 40-ft slot pairing | Both virtual bays updated with same sequence number |
| Reward components | Hand-compute m₁, m₂, m₃, g₁, g₂, g₃ on small instance |
| Policy gradient | Gradient flows through all modules (no NaN, no zeros) |
| Value gradient | MSE decreases over training |
| Full smoke test | One CDG batch → one training step → converges without errors |
