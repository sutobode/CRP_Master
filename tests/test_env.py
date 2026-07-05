import random

import numpy as np
import pytest

from crpsp.env import CRPSPEnv
from crpsp.instance import Instance, generate_instance


def _inst(yard, stowage, t_y=3):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)


def test_reset_runs_transfer_closure():
    # top of stack0 is 1 -> immediately transferable; then 2 -> done at reset
    inst = _inst([[2, 1], []], [[1, 2]])
    env = CRPSPEnv()
    env.reset(inst)
    assert env.done and env.n_transfers == 2 and env.n_relocations == 0


def test_step_relocation_then_auto_transfer():
    inst = _inst([[1, 2], []], [[1, 2]])      # 2 blocks 1
    env = CRPSPEnv(reward_mode="simple")
    obs, mask = env.reset(inst)
    assert not env.done
    a = env.action_pairs.index((0, 1))        # move 2 from stack0 to stack1
    obs, mask, r, done, trunc, info = env.step(a)
    assert done and info["n_relocations"] == 1 and info["n_transfers"] == 2
    assert info["total_ops"] == 3
    assert r == -1.0                          # simple reward: exactly -1 per step


def test_designed_reward_eq36():
    inst = _inst([[1, 2], []], [[1, 2]])
    env = CRPSPEnv(reward_mode="designed", terminal_bonus=10.0)
    env.reset(inst)
    # h(s_t): N_L=2 + 1 cycle = 3 ; after step: yard empty -> h=0
    a = env.action_pairs.index((0, 1))
    _, _, r, done, _, _ = env.step(a)
    assert done
    assert r == pytest.approx(-1 + 3 - 0 + 10.0)


def test_eq23_blocks_transfer_order():
    # container 1 designated to SHORE stack; far-stack container 2 must load first
    inst = _inst([[1], [2], []], [[2], [1]])
    env = CRPSPEnv()
    env.reset(inst)
    assert env.vessel == [[2], [1]] and env.done   # closure loads 2 then 1


def test_eq23_never_violated_during_random_play():
    rng = random.Random(0)
    env = CRPSPEnv()
    for _ in range(10):
        inst = generate_instance(10, 4, 4, 5, rng)
        obs, mask = env.reset(inst)
        steps = 0
        while not env.done and steps < 50:
            valid = np.flatnonzero(mask)
            obs, mask, r, done, trunc, _ = env.step(int(rng.choice(list(valid))))
            hs = [len(s) for s in env.vessel]
            assert all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))
            total = sum(len(s) for s in env.yard) + sum(len(s) for s in env.vessel)
            assert total == inst.n                 # container conservation
            steps += 1
            if done or trunc:
                break


def test_mask_correctness():
    inst = _inst([[1, 2], [], [3]], [[1, 2, 3]], t_y=2)
    env = CRPSPEnv()
    obs, mask = env.reset(inst)
    for idx, (s, d) in enumerate(env.action_pairs):
        expected = len(env.yard[s]) > 0 and len(env.yard[d]) < inst.t_y
        assert mask[idx] == expected


def test_invalid_action_raises():
    inst = _inst([[1, 2], []], [[1, 2]])
    env = CRPSPEnv()
    env.reset(inst)
    bad = env.action_pairs.index((1, 0))      # stack1 empty
    with pytest.raises(ValueError):
        env.step(bad)


def test_truncation_at_max_steps():
    # needs >= 2 relocations (2 and 3 both block 1); one step cannot finish
    inst = _inst([[1, 3, 2], []], [[1, 2, 3]], t_y=3)
    env = CRPSPEnv(max_steps=1)
    obs, mask = env.reset(inst)
    assert not env.done
    obs, mask, r, done, trunc, _ = env.step(env.action_pairs.index((0, 1)))
    assert trunc and not done


def test_obs_encoding_normalized():
    inst = _inst([[1, 2], []], [[1, 2]])
    env = CRPSPEnv()
    obs, _ = env.reset(inst)
    assert obs.dtype == np.float32 and obs.shape == (2, 3)
    assert obs[0, 0] == pytest.approx(1 / 2) and obs[0, 1] == pytest.approx(2 / 2)
    assert obs[1, 0] == 0.0
