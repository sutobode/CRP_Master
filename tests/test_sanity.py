import pathlib

import yaml


def test_config_loads_table10_values():
    cfg = yaml.safe_load(pathlib.Path("configs/default.yaml").read_text())
    assert cfg["train"]["lr"] == 5.0e-4
    assert cfg["train"]["gamma"] == 0.4
    assert cfg["train"]["batch_size"] == 10
    assert cfg["model"]["hidden_dim"] == 128
    assert cfg["train"]["gae_lambda"] == 0.9
    assert cfg["train"]["clip_eps"] == 0.15
    assert cfg["env"]["max_steps"] == 50
    assert cfg["train"]["instances_per_iter"] == 10
