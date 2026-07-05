import random

import pytest

from crpsp.instance import Instance, generate_instance, slot_map


def test_generate_basic_properties():
    rng = random.Random(0)
    inst = generate_instance(n=15, s_y=5, s_v=5, t_y=5, rng=rng)
    assert inst.n == 15 and inst.s_y == 5 and inst.s_v == 5
    ids = sorted(c for st in inst.yard for c in st)
    assert ids == list(range(1, 16))                    # every container exactly once in yard
    assert all(len(st) <= inst.t_y for st in inst.yard)  # yard height limit
    assert sorted(c for st in inst.stowage for c in st) == list(range(1, 16))


def test_stowage_heights_nonincreasing_toward_shore():
    rng = random.Random(1)
    for _ in range(20):
        inst = generate_instance(12, 4, 4, 5, rng)
        hs = [len(st) for st in inst.stowage]           # index 0 = furthest from shore
        assert all(hs[i] >= hs[i + 1] for i in range(len(hs) - 1))  # Eq 23 final profile


def test_labeling_shore_first_bottom_up():
    rng = random.Random(2)
    inst = generate_instance(6, 3, 3, 5, rng)
    labels = []
    for s in range(inst.s_v - 1, -1, -1):               # shore -> far
        labels.extend(inst.stowage[s])                  # bottom -> top
    assert labels == list(range(1, 7))


def test_slot_map_roundtrip():
    rng = random.Random(3)
    inst = generate_instance(10, 4, 4, 5, rng)
    sm = slot_map(inst)
    for vs, stack in enumerate(inst.stowage):
        for vt, c in enumerate(stack):
            assert sm[c] == (vs, vt)


def test_capacity_error():
    with pytest.raises(ValueError):
        generate_instance(n=30, s_y=2, t_y=5, s_v=3, rng=random.Random(0))


def test_seeded_determinism():
    a = generate_instance(15, 5, 5, 5, random.Random(7))
    b = generate_instance(15, 5, 5, 5, random.Random(7))
    assert a == b
