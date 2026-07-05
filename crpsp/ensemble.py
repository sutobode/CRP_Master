"""Ensemble methods: Majority Voting (Eq 40) and Stacking with XGBoost (Eq 41-46).

Stacking uses A*-optimal trajectories as labels (Section 4.3.2): features are
the concatenated per-model action-probability vectors (the matrix P of Eq 45,
flattened row-wise); the aggregator is an XGBoost classifier (Eq 46).
"""
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
    features = concatenated per-model probability vectors (Eq 45)."""
    X, y = [], []
    for inst in instances:
        res = solve_astar(inst)
        if not res.optimal:
            continue
        pairs = [(s, d) for s in range(inst.s_y) for d in range(inst.s_y) if d != s]
        for yard, _vh, action in res.trajectory:
            obs = CRPSPEnv.encode(yard, inst.s_y, inst.t_y, inst.n)
            mask = np.zeros(len(pairs), dtype=bool)
            for i, (s, d) in enumerate(pairs):
                mask[i] = len(yard[s]) > 0 and len(yard[d]) < inst.t_y
            feats = np.concatenate([_probs(a, obs, mask, device) for a in actors])
            X.append(feats)
            y.append(pairs.index(action))
    if not X:
        n_actions = instances[0].s_y * (instances[0].s_y - 1) if instances else 0
        return (np.zeros((0, len(actors) * n_actions), dtype=np.float32),
                np.zeros((0,), dtype=np.int64))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class StackingPolicy:
    def __init__(self, model: xgb.XGBClassifier, n_actions: int, classes: np.ndarray):
        self.model = model
        self.n_actions = n_actions
        self.classes = classes                 # model output columns -> action ids

    def predict(self, features: np.ndarray, mask: np.ndarray) -> int:
        """Eq (46), restricted to valid actions."""
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        full = np.zeros(self.n_actions)
        for col, action in enumerate(self.classes):
            full[int(action)] = proba[col]
        full[~mask] = -1.0
        return int(full.argmax())


def train_stacking(X: np.ndarray, y: np.ndarray, n_actions: int) -> StackingPolicy:
    # XGBoost needs contiguous class labels; action ids may be sparse.
    uniq = np.unique(y)
    y_enc = np.searchsorted(uniq, y)
    model = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss")
    model.fit(X, y_enc)
    return StackingPolicy(model, n_actions, uniq)
