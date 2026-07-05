from crpsp.instance import Instance
from crpsp.lower_bound import lower_bound, must_precede_pairs


def _inst(yard, stowage, t_y=4):
    return Instance(tuple(map(tuple, yard)), tuple(map(tuple, stowage)), t_y)


def test_same_stack_precedence():
    inst = _inst([[1, 2]], [[1, 2]])        # vessel stack: 1 below 2
    p = must_precede_pairs(inst)
    assert (1, 2) in p and (2, 1) not in p


def test_shoreline_precedence_eq23():
    # stowage stack 0 = far holds container 2; stack 1 = shore holds container 1
    inst = _inst([[1], [2]], [[2], [1]])
    p = must_precede_pairs(inst)
    assert (2, 1) in p                       # far tier-1 before near tier-1 (Eq 23)
    assert (1, 2) not in p


def test_lb_counts_transfers_only_when_no_blocking():
    inst = _inst([[2, 1]], [[1, 2]])         # yard: 1 on top of 2; no cycle
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) == 2    # N_L=2, zero cycles


def test_lb_detects_forced_relocation():
    inst = _inst([[1, 2]], [[1, 2]])         # 2 sits on 1, but 1 must go first
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) == 3    # 2 transfers + 1 forced relocation


def test_lb_direct_adjacency_only():
    # yard stack bottom->top: 1,3,2 ; stowage single stack (1,2,3).
    # direct pairs: (3 on 1) with precede(1,3) => cycle; (2 on 3) needs precede(3,2) absent.
    inst = _inst([[1, 3, 2]], [[1, 2, 3]])
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) == 3 + 1


def test_lb_admissible_on_known_optimum():
    # optimal: relocate 2, transfer 1, transfer 2 = 3 ops
    inst = _inst([[1, 2], []], [[1, 2]])
    p = must_precede_pairs(inst)
    assert lower_bound(inst.yard, p) <= 3
