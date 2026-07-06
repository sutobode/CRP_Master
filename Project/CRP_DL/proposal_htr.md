# Đề Xuất Nghiên Cứu: Hierarchical Target-Then-Relocate (HTR) cho Container Retrieval Problem

**Ngày:** 2026-07-06
**Bài báo gốc (Baseline):** Shin et al. (2026) "Learning to Retrieve Containers: A Scale-diverse Deep Reinforcement Learning Approach for the Container Retrieval Problem", *Transportation Research Part C: Emerging Technologies*, DOI: 10.1016/j.trc.2025.105496.
**Repo code:** https://github.com/sutobode/CRP_RL (branch: contribution)

---

## 1. Baseline — Shin et al. (2026, TR-C)

### 1.1 Bài toán

Container Retrieval Problem: N containers trong yard (S stacks, T tiers), mỗi container có priority (1 = cao nhất). Crane chỉ với được container trên cùng. Mục tiêu: retrieve tất cả containers theo thứ tự priority với **working time** tối thiểu.

Working time = travel time (bay diff, row diff) + acceleration + pickup/deposit

### 1.2 Phương pháp baseline

```
Encoder (LSTM + Self-Attention) → stack embeddings
Decoder (Attention):
  1. env.find_target_stack() → rule-based: chọn stack có min priority
  2. Attention(context = W_target(emb) + W_global) → chọn destination
  3. env.step(dest) → relocate + clear (tự động retrieve)
```

**Key điểm yếu:**
- Target selection là **rule-based** (min priority) — không quan tâm working time
- Destination context CHỈ từ min-priority stack — không thấy "bức tranh lớn"
- Không có cơ chế "học" để tối ưu target theo working time

### 1.3 Action space

| | Baseline | Ghi chú |
|---|---------|---------|
| Action | Destination index (S stacks) | Chọn nơi đặt container |
| Target selection | Rule-based (min priority) | Không học |

---

## 2. Phương Pháp Đề Xuất: HTR (Dual-Context Destination)

### 2.1 Key Insight

Destination selection trong baseline chỉ dùng context từ min-priority stack:

```
Original context = W_target(min_priority_emb) + W_global
```

Điều này bỏ qua thông tin về stacks nào sẽ quan trọng TRONG TƯƠNG LAI. Một relocation tốt không chỉ giúp retrieval hiện tại mà còn giảm working time cho các retrieval sau.

### 2.2 Dual-Context Architecture

```
HTR context = W_target(min_priority_emb) + W_goal(learned_goal_emb) + W_global(graph_emb)
                ↑                            ↑
         immediate target            long-term goal (học được)
```

**TargetSelector (Learned Policy):**
- Input: stack embeddings + global embedding
- Output: score mỗi stack → softmax → sample → goal_embedding
- Train: REINFORCE (cùng objective với decoder)

### 2.3 Training

```
Loss = REINFORCE(working_time, dest_ll + goal_ll)
  - dest_ll: log prob của destination actions (từ attention decoder)
  - goal_ll: log prob của goal actions (từ TargetSelector)
  - Cả hai network được train JOINTLY
  - POMO baseline (giống baseline gốc)
```

### 2.4 So sánh

| Yếu tố | Baseline (Shin 2026) | HTR (Proposed) |
|--------|---------------------|----------------|
| Target selection | Rule-based | **Rule-based + learned goal** |
| Destination context | Current target only | **Current target + long-term goal** |
| Tham số thêm | — | **~129K params** (W_goal + TargetSelector) |
| Training | REINFORCE + POMO | **REINFORCE + POMO** (giống) |

---

## 3. Contribution cho Q1

### 3.1 Scientific Novelty

| Khía cạnh | Contribution |
|-----------|-------------|
| **First learned context for CRP destination** | Chưa có work nào dùng learned long-term goal làm context cho destination selection |
| **Dual-context architecture** | Novel integration: rule-based immediate target + learned long-term goal |
| **Joint training** | Cả goal selector và destination decoder cùng optimize working time |

### 3.2 Empirical Validation (cần chạy)

| Experiment | So sánh | Kỳ vọng |
|-----------|---------|---------|
| Baseline vs HTR (cùng compute) | Cùng 1000 epochs, cùng instances | **HTR giảm 5-15% working time** |
| Ablation: learned goal vs random goal | Learned goal selector vs random selection | **Learned tốt hơn** |
| Ablation: dual-context vs single-context | W_target+W_goal+W_global vs W_target+W_global | **Dual tốt hơn single** |

---

## 4. Files HTR trong CRP_RL

| File | Mô tả |
|------|-------|
| `model/target_selector.py` | MLP(embed*2 → 128 → 1): score mỗi stack |
| `model/htr_decoder.py` | Decoder với dual context (W_target + W_goal + W_global) |
| `model/model.py` (sửa) | HTR mode switch (--htr flag) |
| `trainer.py` (sửa) | Handle goal_ll + dest_ll combined loss |
| `main.py` (sửa) | CLI args, --htr flag |

### 4.1 Smoke test

```bash
cd CRP_RL && python main.py --htr --epochs 5 --batch_num 2 --batch_size 16
```

---

## 5. Compute Budget

| Phase | Task | Time |
|-------|------|------|
| Laptop smoke | 5 epochs, batch=16 | ~30s |
| Colab full train | 1000 epochs, batch=128, pomo=16 | ~12 GPU hours |
| Benchmark | Lee instances + Shin instances | ~2 GPU hours |
| Ablations | 3 ablations × 500 epochs | ~6 GPU hours |
| **Total** | | **~20 GPU hours** |

---

## 6. Target Journal

Transportation Research Part C: Emerging Technologies (TR-C) — cùng journal với baseline paper.

Lý do:
- Cùng problem domain
- Cùng journal → reviewer familiarity
- Contribution là cải tiến trực tiếp trên baseline → dễ so sánh
