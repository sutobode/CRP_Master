import random

import numpy as np
import pytest

from crpsp.astar import solve_astar
from crpsp.evaluate import gap, rollout_policy
from crpsp.instance import generate_instance


def test_gap_matches_table4_examples():
    assert gap(21, 20) == pytest.approx(0.05)      # 5.00%
    assert gap(22, 20) == pytest.approx(0.10)
    assert gap(24, 22) == pytest.approx(2 / 22)    # 9.09%


def test_rollout_random_policy_reports_ops():
    rng = random.Random(0)
    inst = generate_instance(8, 3, 3, 4, rng)

    def rand_policy(obs, mask):
        return int(rng.choice(list(np.flatnonzero(mask))))

    out = rollout_policy(rand_policy, inst, {"max_steps": 50})
    assert "total_ops" in out and "solved" in out and out["seconds"] >= 0
    if out["solved"]:
        assert out["total_ops"] >= solve_astar(inst).total_ops
