import torch
from spp_ac.env.bay import BayState


def test_empty_bay_matrix():
    bay = BayState(R=3, T=2, row_weight_max=[10.0, 10.0, 10.0])
    mat = bay.get_matrix()
    assert mat.shape == (6, 3, 2)
    assert (mat[0] == 0).all()
    assert (mat[1] == 10.0).all()


def test_non_loadable():
    bay = BayState(R=3, T=2, row_weight_max=[10.0, 10.0, 10.0],
                   non_loadable={(1, 0)})
    mat = bay.get_matrix()
    assert mat[0, 1, 0] == -1.0
    assert mat[1, 1, 0] == 0.0


def test_load_container():
    bay = BayState(R=2, T=2, row_weight_max=[20.0, 20.0])
    bay = bay.load(0, 0, pod=3, weight=5, ctype=1)
    mat = bay.get_matrix()
    assert mat[0, 0, 0] == 1.0
    assert mat[3, 0, 0] == 3.0
    assert mat[4, 0, 0] == 5.0
    assert mat[5, 0, 0] == 1.0
    assert mat[1, 0, 0] == 15.0


def test_can_load():
    bay = BayState(R=2, T=2, row_weight_max=[20.0, 20.0])
    assert bay.can_load(0, 0, ctype=1) is True
    assert bay.can_load(0, 0, ctype=2) is True


def test_can_load_non_loadable():
    bay = BayState(R=2, T=2, row_weight_max=[20.0, 20.0],
                   non_loadable={(0, 0)})
    assert bay.can_load(0, 0, ctype=1) is False


def test_can_load_overweight():
    bay = BayState(R=2, T=2, row_weight_max=[3.0, 20.0])
    bay = bay.load(0, 0, pod=1, weight=5, ctype=1)
    assert bay.can_load(0, 0, ctype=1) is False
    assert bay.can_load(0, 1, ctype=1) is False
