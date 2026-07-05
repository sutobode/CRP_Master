import random

from crpsp.astar import solve_astar
from crpsp.instance import Instance, generate_instance
from crpsp.lower_bound import lower_bound, must_precede_pairs


def _inst(yard, stowage, t_y=3):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)


def test_zero_relocation_instance():
    res = solve_astar(_inst([[2, 1], []], [[1, 2]]))
    assert res.relocations == 0 and res.total_ops == 2 and res.optimal
    assert res.trajectory == []


def test_one_forced_relocation():
    res = solve_astar(_inst([[1, 2], []], [[1, 2]]))
    assert res.relocations == 1 and res.total_ops == 3
    assert len(res.trajectory) == 1
    yard, vheights, action = res.trajectory[0]
    assert action[0] == 0                      # must move off stack 0


def test_lb_is_admissible_on_random_instances():
    rng = random.Random(0)
    for _ in range(30):
        inst = generate_instance(8, 3, 3, 4, rng)
        res = solve_astar(inst)
        assert res.optimal
        p = must_precede_pairs(inst)
        assert lower_bound(inst.yard, p) <= res.total_ops


def test_trajectory_replays_to_solution():
    """Replaying the returned actions through the env must solve the instance
    with exactly the reported number of operations."""
    from crpsp.env import CRPSPEnv
    rng = random.Random(5)
    for _ in range(10):
        inst = generate_instance(8, 3, 3, 4, rng)
        res = solve_astar(inst)
        env = CRPSPEnv()
        env.reset(inst)
        for _, _, (s, d) in res.trajectory:
            env.step(env.action_pairs.index((s, d)))
        assert env.done
        assert env.n_relocations + env.n_transfers == res.total_ops


def test_solves_table3_smallest_config():
    rng = random.Random(1)
    inst = generate_instance(12, 4, 4, 5, rng)
    res = solve_astar(inst, time_limit_s=30)
    assert res.optimal and res.total_ops >= 12
