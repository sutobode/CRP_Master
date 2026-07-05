from dataclasses import dataclass, field
from pathlib import Path
import yaml
import torch


@dataclass
class EnvConfig:
    num_ports: int = 6
    num_weight_classes: int = 8
    num_container_types: int = 2
    bay_rows: int = 12
    bay_tiers: int = 6
    row_weight_max: float = 50.0
    non_loadable: list | None = None


@dataclass
class RewardConfig:
    lambda_1: float = 0.7
    lambda_2: float = 0.3
    lambda_3: float = 0.01
    alpha_1: float = 3.0
    alpha_2: float = 3.0
    alpha_3: float = 3.0


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_gru_layers: int = 4


@dataclass
class TrainConfig:
    batch_size: int = 512
    num_iterations: int = 20000
    lr: float = 1e-5
    gamma: float = 0.99
    grad_clip: float = 3.0
    seed: int = 42


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            env=EnvConfig(**data.get("env", {})),
            reward=RewardConfig(**data.get("reward", {})),
            model=ModelConfig(**data.get("model", {})),
            train=TrainConfig(**data.get("train", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        data = {
            "env": {"num_ports": self.env.num_ports, "num_weight_classes": self.env.num_weight_classes,
                    "num_container_types": self.env.num_container_types, "bay_rows": self.env.bay_rows,
                    "bay_tiers": self.env.bay_tiers, "row_weight_max": self.env.row_weight_max,
                    "non_loadable": self.env.non_loadable},
            "reward": {"lambda_1": self.reward.lambda_1, "lambda_2": self.reward.lambda_2,
                       "lambda_3": self.reward.lambda_3, "alpha_1": self.reward.alpha_1,
                       "alpha_2": self.reward.alpha_2, "alpha_3": self.reward.alpha_3},
            "model": {"hidden_dim": self.model.hidden_dim, "num_gru_layers": self.model.num_gru_layers},
            "train": {"batch_size": self.train.batch_size, "num_iterations": self.train.num_iterations,
                      "lr": self.train.lr, "gamma": self.train.gamma,
                      "grad_clip": self.train.grad_clip, "seed": self.train.seed},
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def resolved_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
