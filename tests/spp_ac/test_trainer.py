import torch
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
