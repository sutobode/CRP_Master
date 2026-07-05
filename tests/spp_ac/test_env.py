import torch
from spp_ac.config import EnvConfig, RewardConfig
from spp_ac.env.spp_env import SlotStowageEnv


def _make_container_data(P=2, W=2, E=2, N=50, S=1):
    from spp_ac.data.cdg import CfgDataset
    import numpy as np
    return CfgDataset(P, W, E, N, S, rng=np.random.default_rng(42))


def test_env_reset():
    env_cfg = EnvConfig(num_ports=2, num_weight_classes=2, num_container_types=2,
                        bay_rows=2, bay_tiers=2)
    reward_cfg = RewardConfig()
    data = _make_container_data(P=2, W=2, E=2, N=20, S=4)
    env = SlotStowageEnv(env_cfg, reward_cfg, data)
    bay, container = env.reset()
    assert bay.shape == (6, 2, 2)
    assert container.shape == (9, 4)


def test_env_step():
    env_cfg = EnvConfig(num_ports=2, num_weight_classes=2, num_container_types=2,
                        bay_rows=2, bay_tiers=2)
    reward_cfg = RewardConfig()
    data = _make_container_data(P=2, W=2, E=2, N=20, S=4)
    env = SlotStowageEnv(env_cfg, reward_cfg, data)
    env.reset()
    mask = env.get_mask()
    valid_idx = (mask > 0).nonzero(as_tuple=True)[0][0].item()
    bay, container, reward, done = env.step(valid_idx)
    assert bay.shape == (6, 2, 2)
    assert container[valid_idx, 3].item() < data[0, valid_idx, 3].item()
