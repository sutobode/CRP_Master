# CRPSP Reproduction — Wang et al. (2025), Transportation Research Part E

Faithful implementation of **"Learning-based hybrid algorithms for container
relocation problem with storage plan"** (Peixiang Wang et al., Transportation
Research Part E 197 (2025) 104048, DOI
[10.1016/j.tre.2025.104048](https://doi.org/10.1016/j.tre.2025.104048)):

| Component | Paper | Module |
|---|---|---|
| Instance generator | Section 4.2 | `crpsp/instance.py` |
| MIP model | Eq (1)–(26) | `crpsp/mip_cpsat.py` (CP-SAT) |
| Lower bound h(n) | Eq (27)–(30) | `crpsp/lower_bound.py` |
| A* solver | Section 3 | `crpsp/astar.py` |
| Environment (state/action/reward Eq 36) | Section 4.2 | `crpsp/env.py` |
| Self-attention actor-critic | Eq (38)–(39), Fig. 9 | `crpsp/models.py` |
| PPO (Algorithm 1, Table 10) | Section 4.1–4.2 | `crpsp/ppo.py` |
| Voting & Stacking ensembles | Section 4.3 | `crpsp/ensemble.py` |
| Greedy heuristic | Section 5.7 | `crpsp/heuristic.py` |
| Tables 2–9 | Section 5 | `experiments/table*.py` |

Extension beyond the paper: a standard-CRP mode (Caserta benchmark format,
restricted/unrestricted) sharing the same solvers.

All ambiguity resolutions vs. the paper: **`REPRODUCTION_NOTES.md`**.

## Setup

    uv venv --python 3.12 .venv
    uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv/Scripts/python.exe ortools xgboost scikit-learn numpy pyyaml pytest

(On the training server install CUDA torch instead of the CPU wheel.)

## Verify (fast, laptop-safe)

    .venv/Scripts/python.exe -m pytest            # full fast suite
    .venv/Scripts/python.exe -m pytest -m slow    # one-time MIP-vs-A* cross-check

Key guarantees checked by the suite: A* optimal == CP-SAT optimal; the Eq-29
lower bound is admissible; A* trajectories replay exactly through the RL
environment; Eq (23) is never violated; GAE matches hand-computed values.

## Training & experiments (SERVER ONLY — heavy)

Every experiment script runs a tiny `--smoke` check by default and requires
an explicit `--full` for paper-scale runs:

    python experiments/train_ppo.py --full --seed 0 --out checkpoints/m0.pt
    # ... repeat for 10 seeds, then:
    python experiments/table3_astar.py --full
    python experiments/table4_ppo_vs_astar.py --full --ckpt checkpoints/m0.pt
    python experiments/table5_attention_ablation.py --full
    python experiments/table6_7_reward_ablation.py --full
    python experiments/table8_ensemble.py --full --ckpt-dir checkpoints/
    python experiments/table9_voting_vs_heuristic.py --full --ckpt-dir checkpoints/

Results land in `results/*.csv`, checkpoints in `checkpoints/`.
