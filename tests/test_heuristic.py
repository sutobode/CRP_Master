import random

from crpsp.astar import solve_astar
from crpsp.heuristic import solve_heuristic
from crpsp.instance import Instance, generate_instance


def test_solves_trivial():
    inst = Instance(((2, 1), ()), ((1, 2),), 3)
    r = solve_heuristic(inst)
    assert r.solved and r.relocations == 0 and r.total_ops == 2


def test_never_beats_optimal_and_always_solves():
    rng = random.Random(0)
    for _ in range(20):
        inst = generate_instance(10, 4, 4, 5, rng)
        h = solve_heuristic(inst)
        a = solve_astar(inst)
        assert a.optimal
        assert h.solved
        assert h.total_ops >= a.total_ops
