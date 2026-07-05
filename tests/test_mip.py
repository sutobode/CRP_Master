import random

import pytest

from crpsp.astar import solve_astar
from crpsp.instance import Instance, generate_instance
from crpsp.mip_cpsat import solve_mip


def _inst(yard, stowage, t_y=3):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)


def test_trivial_no_relocation():
    inst = _inst([[2, 1], []], [[1, 2]])
    res = solve_mip(inst, horizon=4)
    assert res.status == "OPTIMAL" and res.target_reached
    assert res.total_ops == 2 and res.relocations == 0


def test_one_relocation():
    inst = _inst([[1, 2], []], [[1, 2]])
    res = solve_mip(inst, horizon=5)
    assert res.status == "OPTIMAL" and res.target_reached
    assert res.total_ops == 3 and res.relocations == 1


def test_eq23_enforced():
    # far stack must be loaded first; MIP must still reach target with 0 relocations
    inst = _inst([[1], [2], []], [[2], [1]])
    res = solve_mip(inst, horizon=4)
    assert res.status == "OPTIMAL" and res.target_reached and res.relocations == 0


@pytest.mark.slow
def test_cross_check_vs_astar_n5():
    rng = random.Random(0)
    for _ in range(3):
        inst = generate_instance(5, 3, 3, 5, rng)
        a = solve_astar(inst)
        assert a.optimal
        m = solve_mip(inst, horizon=a.total_ops + 2, time_limit_s=120)
        assert m.status == "OPTIMAL" and m.target_reached
        assert m.total_ops == a.total_ops
