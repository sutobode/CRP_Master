import random

import numpy as np
import torch

from crpsp.ensemble import build_stacking_dataset, train_stacking, vote
from crpsp.instance import generate_instance
from crpsp.models import Actor


def _actors(k, s_y=3, t_y=3, seed0=0):
    out = []
    for i in range(k):
        torch.manual_seed(seed0 + i)
        out.append(Actor(s_y, t_y, hidden_dim=16))
    return out


def test_single_model_vote_equals_argmax():
    s_y, t_y = 3, 3
    actors = _actors(1, s_y, t_y)
    obs = np.random.RandomState(0).rand(s_y, t_y).astype(np.float32)
    mask = np.ones(s_y * (s_y - 1), dtype=bool)
    with torch.no_grad():
        logits = actors[0](torch.as_tensor(obs).unsqueeze(0),
                           torch.as_tensor(mask).unsqueeze(0))
    assert vote(actors, obs, mask, torch.device("cpu")) == int(logits.argmax())


def test_vote_majority_valid_range():
    s_y, t_y = 3, 3
    actors = _actors(5, s_y, t_y)
    obs = np.random.RandomState(1).rand(s_y, t_y).astype(np.float32)
    mask = np.ones(s_y * (s_y - 1), dtype=bool)
    a = vote(actors, obs, mask, torch.device("cpu"))
    assert 0 <= a < s_y * (s_y - 1)


def test_vote_never_selects_masked_action():
    s_y, t_y = 3, 3
    actors = _actors(5, s_y, t_y)
    obs = np.zeros((s_y, t_y), dtype=np.float32)
    mask = np.zeros(s_y * (s_y - 1), dtype=bool)
    mask[3] = True
    assert vote(actors, obs, mask, torch.device("cpu")) == 3


def test_stacking_dataset_and_training():
    rng = random.Random(0)
    instances = [generate_instance(6, 3, 3, 4, rng) for _ in range(8)]
    actors = _actors(3, 3, 4)
    X, y = build_stacking_dataset(actors, instances, torch.device("cpu"))
    n_actions = 3 * 2
    assert len(y) > 0, "need at least one instance requiring relocation"
    assert X.shape[1] == len(actors) * n_actions
    pol = train_stacking(X, y, n_actions)
    mask = np.ones(n_actions, dtype=bool)
    a = pol.predict(X[0], mask)
    assert 0 <= a < n_actions


def test_stacking_respects_mask():
    rng = random.Random(0)
    instances = [generate_instance(6, 3, 3, 4, rng) for _ in range(8)]
    actors = _actors(3, 3, 4)
    X, y = build_stacking_dataset(actors, instances, torch.device("cpu"))
    pol = train_stacking(X, y, 6)
    only = np.zeros(6, dtype=bool)
    only[4] = True
    assert pol.predict(X[0], only) == 4
