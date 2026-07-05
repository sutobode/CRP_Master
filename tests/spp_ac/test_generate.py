import torch
from pathlib import Path
from spp_ac.config import Config
from spp_ac.models.actor import Actor
from spp_ac.generate import generate_plan, plot_stowage_plan


def test_generate_plan_runs():
    config = Config()
    config.train.batch_size = 2
    actor = Actor(config.model.hidden_dim, config.model.num_gru_layers)
    plans = generate_plan(actor, config, num_instances=2, greedy=True)
    assert len(plans) == 2
    assert "reward" in plans[0]
    assert "containers" in plans[0]
    assert len(plans[0]["containers"]) > 0


def test_generate_plan_sampling():
    config = Config()
    config.train.batch_size = 2
    actor = Actor(config.model.hidden_dim, config.model.num_gru_layers)
    plans = generate_plan(actor, config, num_instances=1, greedy=False)
    assert len(plans) == 1


def test_generate_plan_containers_have_keys():
    config = Config()
    config.train.batch_size = 2
    actor = Actor(config.model.hidden_dim, config.model.num_gru_layers)
    plans = generate_plan(actor, config, num_instances=1, greedy=True)
    c = plans[0]["containers"][0]
    assert "action" in c
    assert "pod" in c
    assert "weight_class" in c
    assert "container_type" in c


def test_plot_stowage_plan():
    try:
        from spp_ac.generate import _import_plt
        if _import_plt() is None:
            import pytest
            pytest.skip("matplotlib not installed")
    except ImportError:
        import pytest
        pytest.skip("matplotlib not installed")
    bay_state = torch.zeros(6, 4, 3)
    fig = plot_stowage_plan(bay_state, title="Test")
    assert fig is not None
    import matplotlib
    matplotlib.pyplot.close(fig)


def test_trainer_generate_plan():
    from spp_ac.training.trainer import Trainer
    config = Config()
    config.train.batch_size = 2
    device = torch.device("cpu")
    trainer = Trainer(config, device)
    plans = trainer.generate_plan(num_instances=1, greedy=True)
    assert len(plans) == 1
