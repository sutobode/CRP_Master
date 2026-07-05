import torch
import tempfile
from pathlib import Path
from spp_ac.config import Config
from spp_ac.training.trainer import Trainer


def test_trainer_smoke():
    config = Config()
    config.train.batch_size = 4
    config.train.num_iterations = 5
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    trainer.train(num_iterations=2)
    assert True


def test_trainer_actor_parameters_update():
    config = Config()
    config.train.batch_size = 2
    config.train.num_iterations = 1
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    before = [p.clone() for p in trainer.actor.parameters()]
    trainer.train(num_iterations=1)
    after = list(trainer.actor.parameters())
    any_changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert any_changed


def test_checkpoint_save_load():
    config = Config()
    config.train.batch_size = 2
    config.train.num_iterations = 1
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    trainer.train(num_iterations=1)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    try:
        trainer.save_checkpoint(ckpt_path)
        assert Path(ckpt_path).exists()

        trainer2 = Trainer(config, device)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.start_iteration == 1

        for p1, p2 in zip(trainer.actor.parameters(), trainer2.actor.parameters()):
            assert torch.equal(p1, p2)
        for p1, p2 in zip(trainer.critic.parameters(), trainer2.critic.parameters()):
            assert torch.equal(p1, p2)
    finally:
        Path(ckpt_path).unlink(missing_ok=True)


def test_checkpoint_load_missing():
    config = Config()
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    import pytest
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint("/nonexistent/path.pt")
