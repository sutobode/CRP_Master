import random

import torch

from crpsp.ppo import PPOTrainer, TrainConfig, compute_gae, convergence_iteration


def test_gae_hand_computed():
    # gamma=0.5, lam=0.5 ; two steps, terminal at t=1
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.5])
    next_values = torch.tensor([0.5, 0.9])    # bootstrap ignored at terminal
    dones = torch.tensor([0.0, 1.0])
    adv, targets = compute_gae(rewards, values, next_values, dones, 0.5, 0.5)
    # d0 = 1 + 0.5*0.5 - 0.5 = 0.75 ; d1 = 1 + 0 - 0.5 = 0.5
    # A1 = 0.5 ; A0 = 0.75 + (0.25)*0.5 = 0.875
    assert torch.allclose(adv, torch.tensor([0.875, 0.5]), atol=1e-6)
    # y0 = 1 + 0.5*0.5 = 1.25 ; y1 = 1 + 0 (terminal) = 1.0
    assert torch.allclose(targets, torch.tensor([1.25, 1.0]), atol=1e-6)


def _tiny_cfg():
    return TrainConfig(n=5, s_y=3, s_v=3, t_y=3, reward_mode="designed",
                       terminal_bonus=10.0, max_steps=50, hidden_dim=32,
                       use_attention=True, lr=5e-4, gamma=0.4, gae_lambda=0.9,
                       clip_eps=0.15, batch_size=10, instances_per_iter=2,
                       iterations=3, ppo_epochs=1, seed=0, device="cpu")


def test_smoke_train_three_iterations():
    """Machinery check only — NOT a training run (Global Constraints)."""
    cfg = _tiny_cfg()
    tr = PPOTrainer(cfg)
    rng = random.Random(0)
    hist = [tr.train_iteration(rng) for _ in range(3)]
    for m in hist:
        assert torch.isfinite(torch.tensor(m["actor_loss"]))
        assert torch.isfinite(torch.tensor(m["critic_loss"]))
        assert 0.0 <= m["solve_rate"] <= 1.0


def test_config_from_yaml_matches_dataclass():
    import pathlib

    import yaml
    cfg = yaml.safe_load(pathlib.Path("configs/default.yaml").read_text())
    tc = TrainConfig.from_yaml(cfg)
    assert tc.lr == 5e-4 and tc.gamma == 0.4 and tc.hidden_dim == 128
    assert tc.max_steps == 50 and tc.instances_per_iter == 10


def test_save_load_roundtrip(tmp_path):
    cfg = _tiny_cfg()
    tr = PPOTrainer(cfg)
    p = tmp_path / "ck.pt"
    tr.save(p)
    tr2 = PPOTrainer(cfg)
    tr2.load(p)
    for a, b in zip(tr.actor.parameters(), tr2.actor.parameters()):
        assert torch.equal(a, b)


def test_convergence_iteration_definition():
    hist = [{"solve_rate": 0.0, "mean_relocations": 9.0}] * 5 \
         + [{"solve_rate": 1.0, "mean_relocations": 3.0}] * 25
    assert convergence_iteration(hist, window=20) == 25
    assert convergence_iteration(hist[:10], window=20) is None
