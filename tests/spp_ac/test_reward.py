from spp_ac.config import RewardConfig
from spp_ac.env.reward import RewardTracker


def test_reward_no_violations():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[20.0, 20.0])
    tracker.record_load(0, 0, pod=1, weight=5, ctype=1)
    tracker.record_load(0, 1, pod=2, weight=3, ctype=1)
    reward = tracker.compute()
    assert reward < 0


def test_reward_with_reo():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[20.0, 20.0])
    tracker.record_load(0, 0, pod=2, weight=3, ctype=1)
    tracker.record_load(0, 1, pod=1, weight=5, ctype=1)
    reward = tracker.compute()
    assert reward < 0


def test_reward_empty():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[20.0, 20.0])
    assert tracker.compute() == 0.0


def test_reward_overweight():
    cfg = RewardConfig()
    tracker = RewardTracker(cfg, R=2, T=2, row_weight_max=[5.0, 20.0])
    tracker.record_load(0, 0, pod=1, weight=10, ctype=1)
    reward = tracker.compute()
    assert reward < -(cfg.alpha_3 * (5.0 / 25.0))
