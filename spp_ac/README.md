# SPP-AC: Slot Stowage Optimization via Improved Actor-Critic

Faithful reproduction of Du et al. (2026), *An automated slot stowage optimization method for container ship based on improved Actor-Critic algorithm*, Ocean Engineering 350, 124255.

## Overview

Solves the Slot Planning Problem (SPP) — assigning containers to individual slots within a bay of a container ship — using Deep Reinforcement Learning. Handles mixed 20-ft and 40-ft containers under seaworthiness constraints.

### Method

- **MDP**: Bay state `[6, R, T]` + Container state `[M, 4]` → action (select container group) → reward (REO + LCGCP + HOL + penalties)
- **Actor** (Pointer Network): CNN+LSTM encoder, GRU decoder with masked attention
- **Critic**: Same encoder + 4×Conv1d FCL → scalar value
- **Training**: REINFORCE with baseline (Eq 27-28)

## Project Structure

```
spp_ac/
├── config.py              Config dataclasses + YAML I/O
├── config.yaml            Hyperparameters (Table 2, 4)
├── main.py                CLI entry point
├── requirements.txt       Dependencies
├── data/
│   └── cdg.py             CDG dataset generator (Algorithm 1)
├── env/
│   ├── bay.py             Bay state [6, R, T] matrix
│   ├── container.py       Container state [M, 4] operations
│   ├── sequence.py        Stowage sequence (Fig 4)
│   ├── reward.py          Reward functions (Eq 17-25)
│   └── spp_env.py         MDP environment
├── models/
│   ├── encoder.py         Container + Bay encoders (Fig 6-7)
│   ├── decoder.py         GRU + Masked Attention (Eq 26)
│   ├── actor.py           Policy network
│   └── critic.py          Value network
└── training/
    └── trainer.py         REINFORCE + baseline (Eq 27-28)
```

## Usage

```bash
# Install dependencies
pip install -r spp_ac/requirements.txt

# Smoke test (3 iterations, batch=4)
python -m spp_ac.main --iterations 3 --batch-size 4 --device cpu

# Full training
python -m spp_ac.main --device cuda --checkpoint checkpoints/model.pt
```

### CLI Arguments

| Flag | Default | Description |
|---|---|---|
| `--config` | `spp_ac/config.yaml` | Config file path |
| `--iterations` | 20000 | Training iterations |
| `--batch-size` | 512 | Batch size (K in Eq 27) |
| `--lr` | 1e-5 | Learning rate |
| `--device` | auto | `auto`, `cpu`, or `cuda` |
| `--checkpoint` | None | Save path |
| `--resume` | None | Load path |

## Configuration

See `config.yaml`. Key parameters from the paper:

| Param | Value | Source |
|---|---|---|
| `hidden_dim` | 128 | Table 2 |
| `num_gru_layers` | 4 | Table 2 |
| `lambda_1` (REO) | 0.7 | Table 4 Set A |
| `lambda_2` (LCGCP) | 0.3 | Table 4 Set A |
| `lambda_3` (HOL) | 0.01 | Table 4 Set A |

## Tests

```bash
python -m pytest tests/spp_ac/ -v
```

## Paper Implementation Notes

| Component | Reference | Status |
|---|---|---|
| Bay state encoding (6 channels) | Section 2.2.1, Fig 2 | ✅ |
| Container state encoding (M×4) | Section 2.2.1, Fig 3 | ✅ |
| Stowage sequence (Fig 4) | Section 2.2.2 | ✅ |
| Reward R₁ + R₂ | Eq 17-25 | ✅ |
| Container encoder (Conv1D+LSTM) | Section 3.2.1, Fig 6 | ✅ |
| Bay encoder (Conv2D+LSTM) | Section 3.2.1, Fig 7 | ✅ |
| GRU decoder + Masked Attention | Section 3.2.2-3.2.3, Eq 26 | ✅ |
| Value network (4×Conv1d) | Section 3.3 | ✅ |
| REINFORCE + baseline | Eq 27-28 | ✅ |
| CDG algorithm | Algorithm 1 | ✅ |

## Code vs Paper: Notable Deviations

- **Eq 26 (Masking)**: Paper shows `softmax(u + 1e-9)` for zero-quantity groups, but text describes "logarithm of the mask". Code uses `attn - 1e9` for masked groups to produce numerically stable zero probability. The paper's Eq 26 and text are ambiguous on this point; the implementation achieves the intended effect.
- **Fig 7 (Bay encoder MaxPool)**: Paper specifies pool kernel 3×3, stride 2. Code adds `padding=1` to prevent dimension collapse for small bay sizes (12×6).
- **Eq 24 α signs**: Table 4 lists α values as negative (-3), but Eq 24 uses `R₂ = -(αg)`. Interpreted as positive magnitude (α=3) to produce penalty on violation.
