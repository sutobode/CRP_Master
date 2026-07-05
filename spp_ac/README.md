# SPP-AC: Slot Stowage Optimization via Improved Actor-Critic

Faithful reproduction of **Du et al. (2026)**, *An automated slot stowage optimization method for container ship based on improved Actor-Critic algorithm*, Ocean Engineering 350, 124255.

## Overview

Solves the **Slot Planning Problem (SPP)** — assigning containers to individual slots within a bay of a container ship — using Deep Reinforcement Learning. Handles mixed 20-ft and 40-ft containers under seaworthiness constraints.

### Method

| Component | Approach | Paper Reference |
|---|---|---|
| Problem | MDP: bay state `[6, R, T]` + container state `[M, 4]` → action → reward | Section 2.2 |
| Policy (Actor) | Pointer Network: CNN+LSTM encoder, GRU decoder, masked attention | Section 3.2, Fig 6-7 |
| Value (Critic) | Same encoder + 4×Conv1d FCL | Section 3.3 |
| Training | REINFORCE with baseline (Eq 27-28) | Section 3.4 |
| Data | CDG algorithm for synthetic container manifests | Section 4.1, Algorithm 1 |

## Project Structure

```
spp_ac/
├── main.py              CLI entry point (train / generate)
├── config.py            Config dataclasses + YAML I/O
├── config.yaml          Default hyperparameters (Table 2, 4)
├── generate.py          Stowage plan generation + heatmap visualization
├── requirements.txt     Dependencies
├── README.md            This file
│
├── data/
│   └── cdg.py           CDG dataset generator (Algorithm 1)
│
├── env/
│   ├── bay.py           BayState — 6-channel matrix [6, R, T]
│   ├── container.py     ContainerState — M×4 matrix operations
│   ├── sequence.py      StowageSequence — bottom-up, sides-to-center (Fig 4)
│   ├── reward.py        Reward functions R₁ + R₂ (Eq 17-25)
│   └── spp_env.py       SlotStowageEnv — MDP environment
│
├── models/
│   ├── encoder.py       ContainerEncoder (Conv1D+LSTM) + BayEncoder (Conv2D+LSTM)
│   ├── decoder.py       GRU decoder + Masked Attention (Eq 26)
│   ├── actor.py         Policy network (encoder + decoder)
│   └── critic.py        Value network (encoder + 4×Conv1d)
│
└── training/
    └── trainer.py       REINFORCE + baseline training loop (Eq 27-28)

tests/spp_ac/
├── test_actor.py        Actor forward + gradient tests
├── test_bay.py          Bay state operations
├── test_cdg.py          CDG data generation
├── test_container.py    Container state operations
├── test_critic.py       Critic forward + gradient
├── test_decoder.py      Attention decoder + mask
├── test_encoder.py      Container + Bay encoders
├── test_env.py          MDP environment
├── test_generate.py     Plan generation + plotting
├── test_reward.py       Reward computation
├── test_sequence.py     Stowage sequence
└── test_trainer.py      Training loop smoke tests
```

## Quick Start

### Installation

```bash
pip install -r spp_ac/requirements.txt
```

If you plan to use visualization:
```bash
pip install matplotlib  # optional, for --plot
```

### Smoke Test (CPU)

```bash
# Train 3 iterations with batch_size=2
python -m spp_ac.main train --iterations 3 --batch-size 2 --device cpu

# Generate a stowage plan
python -m spp_ac.main generate --checkpoint checkpoints/model.pt --num-instances 1 --device cpu

# Generate with sampling (non-greedy) + save plot
python -m spp_ac.main generate --checkpoint checkpoints/model.pt --sample --plot plan.png
```

### Full Training (GPU)

```bash
python -m spp_ac.main train \
    --iterations 20000 \
    --batch-size 512 \
    --device cuda \
    --checkpoint checkpoints/model.pt
```

### Resume Training

```bash
python -m spp_ac.main train \
    --resume checkpoints/model.pt \
    --iterations 10000 \    # additional iterations
    --device cuda
```

## CLI Reference

### `train` subcommand

| Argument | Default | Description |
|---|---|---|
| `--config` | `spp_ac/config.yaml` | Config file path |
| `--iterations` | `20000` | Number of training iterations |
| `--batch-size` | `512` | Batch size (K in Eq 27) |
| `--lr` | `1e-5` | Learning rate |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--checkpoint` | `checkpoints/model.pt` | Checkpoint save path |
| `--resume` | `None` | Resume from checkpoint path |

### `generate` subcommand

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Path to trained model checkpoint |
| `--num-instances` | `1` | Number of instances to generate |
| `--sample` | `False` | Sample from policy (default: greedy argmax) |
| `--plot` | `None` | Save heatmap plot to path |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--config` | `spp_ac/config.yaml` | Config file path |

## Configuration

Edit `spp_ac/config.yaml` or pass a custom path with `--config`.

### Environment (`env`)

| Parameter | Default | Description |
|---|---|---|
| `num_ports` | `6` | Number of ports of discharge (P) |
| `num_weight_classes` | `8` | Number of weight classes (W) |
| `num_container_types` | `2` | Number of container types (E: 20ft, 40ft) |
| `bay_rows` | `12` | Number of rows (stacks) per bay |
| `bay_tiers` | `6` | Number of tiers per row |
| `row_weight_max` | `50.0` | Max weight per row |
| `non_loadable` | `[]` | List of `[row, tier]` slots that cannot be loaded |

### Reward (`reward`)

| Parameter | Default | Paper Reference |
|---|---|---|
| `lambda_1` | `0.7` | REO weight (Table 4 Set A) |
| `lambda_2` | `0.3` | LCGCP weight |
| `lambda_3` | `0.01` | HOL weight |
| `alpha_1` | `3.0` | Overhang penalty coefficient (positive magnitude) |
| `alpha_2` | `3.0` | 20TOP40 penalty coefficient |
| `alpha_3` | `3.0` | Weight limit penalty coefficient |

### Model (`model`)

| Parameter | Default | Description |
|---|---|---|
| `hidden_dim` | `128` | Hidden dimension (Table 2) |
| `num_gru_layers` | `4` | Number of GRU decoder layers (Table 2) |

### Training (`train`)

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | `512` | Batch size K (Eq 27) |
| `num_iterations` | `20000` | Training iterations |
| `lr` | `1e-5` | Learning rate (Adam) |
| `gamma` | `0.99` | Discount factor |
| `grad_clip` | `3.0` | Gradient clipping max norm |
| `seed` | `42` | Random seed |

## Device Management

The `--device` flag supports three modes:

- **`auto`** — auto-detect: uses CUDA if available, otherwise CPU
- **`cpu`** — force CPU (useful for debugging or limited memory)
- **`cuda`** — force CUDA GPU (fails if no GPU available)

Example switching between devices:

```bash
# Debug on CPU
python -m spp_ac.main train --iterations 5 --batch-size 4 --device cpu

# Full training on GPU
python -m spp_ac.main train --device cuda --checkpoint checkpoints/model.pt
```

## Checkpoint System

Checkpoints save:
- Actor state dict
- Critic state dict
- Both optimizer state dicts
- Config at save time
- Current iteration number

To continue training from a checkpoint, use `--resume`:

```bash
python -m spp_ac.main train --resume checkpoints/model.pt --iterations 5000 --device cuda
```

This loads the model, optimizer states, and iteration counter, then trains for 5000 more iterations.

## Tests

```bash
# Run all spp_ac tests
python -m pytest tests/spp_ac/ -v

# Run specific test file
python -m pytest tests/spp_ac/test_env.py -v
```

## Paper Implementation Checklist

| Component | Paper Reference | Status | Notes |
|---|---|---|---|
| Bay state (6 channels) | Section 2.2.1, Fig 2 | ✅ | |
| Container state (M×4) | Section 2.2.1, Fig 3 | ✅ | M = E·P·W + 1 |
| MDP action | Section 2.2.2 | ✅ | Invalid actions rejected, 40ft paired slot |
| Reward R₁ + R₂ | Eq 17-25 | ✅ | See note on α signs below |
| Container encoder | Section 3.2.1, Fig 6 | ✅ | Conv1D(4→64→128→128) + LSTM |
| Bay encoder | Section 3.2.1, Fig 7 | ✅ | Conv2D(6→64→128→128) + 2×MaxPool + AvgPool + LSTM |
| GRU decoder | Section 3.2.2 | ✅ | 4-layer GRU |
| Masked attention | Section 3.2.3, Eq 26 | ✅ | See note on mask value below |
| Value network | Section 3.3 | ✅ | 4×Conv1d(256→128→64→16→1) |
| Policy gradient | Eq 27 | ✅ | REINFORCE with baseline |
| Value gradient | Eq 28 | ✅ | MSE loss |
| CDG algorithm | Algorithm 1 | ✅ | Random integer partition + hierarchical distribution |
| Stowage sequence | Fig 4 | ✅ | Bottom-up, both sides toward center |
| 40-ft handling | Section 2.2.2 | ✅ | Paired slots, same sequence number |
| Partial loading | Section 4.1 | ✅ | Zero-group placeholder |

### Deviations from Paper

| Item | Paper | Implementation | Reason |
|---|---|---|---|
| Eq 26 mask | `softmax(u + 1e-9)` for `qty=0` | `attn - 1e9` | Paper's formula doesn't zero out probabilities — `-1e9` produces numerically stable zero probability for empty groups. The paper text also references "logarithm of the mask" which differs from the equation. |
| Fig 7 MaxPool | No padding | `padding=1` | Prevents dimension collapse for small bay sizes (12×6). Without it, after two MaxPool(3, stride=2) layers, spatial dimension 6 → 2 → 0. |
| Eq 24 α signs | Table 4 lists α = -3 | Config stores α = +3 | Eq 24: `R₂ = -(αg)`. Negative α in table + minus sign = positive penalty on violation. Interpreted as positive magnitude for clarity. |

## Citation

```bibtex
@article{du2026automated,
  title={An automated slot stowage optimization method for container ship
         based on improved Actor-Critic algorithm},
  author={Du, Chen and Sun, Xiaofeng and Liu, Chunlei},
  journal={Ocean Engineering},
  volume={350},
  pages={124255},
  year={2026},
  publisher={Elsevier}
}
```
