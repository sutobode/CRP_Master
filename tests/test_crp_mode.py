from crpsp.astar import solve_astar
from crpsp.env import CRPSPEnv
from crpsp.instance import load_caserta


def test_parse_caserta():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    assert inst.mode == "crp"
    assert inst.s_y == 3 and inst.n == 6
    assert inst.yard == ((5, 2), (1, 4), (6, 3))   # rows are bottom -> top
    assert inst.t_y == 4                            # H=2 -> H+2 (CV convention)
    assert inst.priorities == (1, 2, 3, 4, 5, 6)


def test_crp_transfer_rule():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    env = CRPSPEnv()
    obs, mask = env.reset(inst)
    # priority 1 sits under 4 in stack 1 -> nothing retrievable at reset
    assert env.n_transfers == 0 and not env.done
    # relocate 4 (top of stack 1) to stack 0 -> 1 retrieves; 2 is buried under 4
    a = env.action_pairs.index((1, 0))
    obs, mask, r, done, trunc, info = env.step(a)
    assert info["n_transfers"] == 1


def test_crp_episode_completes():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    env = CRPSPEnv()
    obs, mask = env.reset(inst)
    import numpy as np
    import random
    rng = random.Random(3)
    steps = 0
    while not env.done and steps < 50:
        valid = np.flatnonzero(mask)
        obs, mask, r, done, trunc, _ = env.step(int(rng.choice(list(valid))))
        steps += 1
        if done or trunc:
            break
    # random play may or may not finish, but counters must stay consistent
    total = sum(len(s) for s in env.yard) + env.n_transfers
    assert total == inst.n


def test_astar_solves_crp_instance():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    res = solve_astar(inst)
    assert res.optimal and res.total_ops >= 6      # at least one op per container
    assert res.relocations >= 1                    # 1 is blocked by 4 initially


def test_astar_crp_trajectory_replays():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    res = solve_astar(inst)
    env = CRPSPEnv()
    env.reset(inst)
    for _, _, (s, d) in res.trajectory:
        env.step(env.action_pairs.index((s, d)))
    assert env.done
    assert env.n_relocations + env.n_transfers == res.total_ops


def test_restricted_mask():
    inst = load_caserta("tests/fixtures/caserta_sample.dat")
    env = CRPSPEnv(restricted=True)
    obs, mask = env.reset(inst)
    # target=1 sits in stack index 1 -> only sources from stack 1 allowed
    assert mask.any()
    for i, (s, d) in enumerate(env.action_pairs):
        if mask[i]:
            assert s == 1
