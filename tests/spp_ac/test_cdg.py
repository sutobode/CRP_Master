import numpy as np
from spp_ac.data.cdg import rip, CfgDataset


def test_rip_sums_to_n():
    parts = rip(100, 6)
    assert sum(parts) == 100
    assert len(parts) == 6
    assert all(p >= 0 for p in parts)


def test_rip_deterministic_seed():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = rip(100, 6, rng1)
    p2 = rip(100, 6, rng2)
    assert p1 == p2


def test_cdg_output_shape():
    data = CfgDataset(P=2, W=3, E=2, N=100, S=8, rng=np.random.default_rng(42))
    M = 2 * 3 * 2 + 1
    assert data.shape == (8, M, 4)


def test_cdg_quantity_sum():
    data = CfgDataset(P=2, W=3, E=2, N=100, S=4, rng=np.random.default_rng(42))
    total_qty = data[:, :, 3].sum().item()
    assert total_qty == 400


def test_cdg_zero_group():
    data = CfgDataset(P=2, W=3, E=2, N=100, S=1, rng=np.random.default_rng(42))
    zero_group = data[0, -1]
    assert zero_group[0].item() == 0
    assert zero_group[1].item() == 0
    assert zero_group[2].item() == 0
