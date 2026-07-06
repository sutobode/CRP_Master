# Đề Xuất Nghiên Cứu: Hierarchical Target-Then-Relocate (HTR) cho CRPSP

**Ngày:** 2026-07-06
**Bài báo gốc (Baseline):** Wang et al. (2025) "Learning-based hybrid algorithms for container relocation problem with storage plan", *Transportation Research Part E* 197, 104048.
**Repo code:** https://github.com/sutobode/CRP_RL

---

## 1. Baseline — Wang et al. (2025)

### 1.1 Bài toán CRPSP

CRPSP: Cho N container trong yard (S_y stacks, tối đa T_y tầng) với stowage plan cố định trên tàu (S_v stacks). Mỗi operation = relocation (yard→yard) hoặc transfer (yard→vessel tự động). Mục tiêu: minimize số relocations.

### 1.2 Hạn chế của baseline

| Hạn chế | Evidence |
|---------|----------|
| **Action space lớn** | S_y·(S_y−1) actions mỗi step |
| **Scale nhỏ** | Chỉ test N≤30, A* 20% success ở N=30 |
| **PPO chậm** | 40k iterations để hội tụ |
| **Không structural decomposition** | "Algorithm does not represent structural information well enough" (line 1208) |

---

## 2. Phương Pháp Đề Xuất: HTR

### 2.1 Key Insight

CRPSP có cấu trúc 2-level tự nhiên:

```
Macro-step: load ONE container lên tàu
  ├── HIGH: chọn stack nào cần clear? (target selection)
  └── LOW:  relocate blockers đến đâu? (blocker relocation)
```

**Target selection** cần global view → RL phù hợp.
**Blocker relocation** là subproblem NHỎ (bị chặn bởi ≤ T_y tầng) → OR solver phù hợp.

### 2.2 Action space reduction

| | Baseline (flat PPO) | HTR |
|---|---|---|
| Actions per step | S_y·(S_y−1) | **S_y** (chọn target stack) |
| N=15, S_y=5 | 20 | **5** |
| N=100, S_y=11 | 110 | **11** |
| Training complexity | O(S_y²) | **O(S_y)** |

### 2.3 Kiến trúc

```
HTREnv:
  Action: stack index (0..S_y-1)
  Step:  relocate blockers via subproblem solver → transfer closure → reward = -relocations
  
TargetSelector (Policy):
  Input: yard matrix Θ (S_y × T_y) → RowSelfAttention → MLP → logits (S_y)
  
Subproblem Solver:
  Input: target container at (stack s, tier k)
  Output: relocation sequence for k blockers (k ≤ T_y ≤ 5-6)
  Method: heuristic (fast) hoặc A* (optimal, node_limit=500)
```

### 2.4 Training

PPO với clipped objective (giống baseline), action space S_y.
- Warm-start: BC từ A* demonstrations (optional)
- Fine-tune: 2,000 iterations (vs baseline 40,000)

---

## 3. So sánh với Baseline

| Metric | Baseline PPO | HTR (proposed) |
|--------|-------------|----------------|
| **Action space** | S_y·(S_y−1) = 20 | **S_y = 5** |
| **Training iterations** | 40,000 | **~2,000** |
| **Gap N=15** | 7.33% | **< 3%** (kỳ vọng) |
| **Scaling N=100** | Không được báo cáo | **~2s solve time** |
| **Generalization** | Train N=15, test N=15 | **Zero-shot N→2N** |

---

## Files đã implement

| File | Mô tả |
|------|-------|
| `crpsp/htr_env.py` | HTR environment (action=stack index, reward=-relocations) |
| `crpsp/subproblem.py` | Blocker relocation solver (heuristic/A*) |
| `crpsp/models_htr.py` | TargetSelector policy network |
| `crpsp/ppo_htr.py` | PPO training for HTR |
| `experiments/train_htr.py` | Training script |

---

## Compute budget

| Phase | Task | Time |
|-------|------|------|
| Smoke test | 5 iterations N=15 | ~1 phút (laptop) |
| Full train | 2,000 iterations N=15 | ~2 GPU hours (Colab) |
| Baseline comparison | Evaluate on N=15..100 | ~1 GPU hour |
