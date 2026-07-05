import numpy as np
import torch
from spp_ac.config import EnvConfig, RewardConfig
from spp_ac.env.bay import BayState
from spp_ac.env.container import select_container, remaining_quantity
from spp_ac.env.sequence import StowageSequence
from spp_ac.env.reward import RewardTracker


class SlotStowageEnv:
    def __init__(
        self,
        env_cfg: EnvConfig,
        reward_cfg: RewardConfig,
        container_data: torch.Tensor,
        rng: np.random.Generator | None = None,
    ):
        self.env_cfg = env_cfg
        self.reward_cfg = reward_cfg
        self.container_data = container_data
        self.rng = rng or np.random.default_rng()
        self.num_ports = env_cfg.num_ports
        self.R = env_cfg.bay_rows
        self.T = env_cfg.bay_tiers
        self.current_idx = 0
        self._zero_group_idx = container_data.size(1) - 1

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.container_state = self.container_data[self.current_idx].clone()
        self.current_idx = (self.current_idx + 1) % len(self.container_data)
        row_weight_max = [self.env_cfg.row_weight_max] * self.R
        non_loadable_bay = {(r, t) for r, t in (self.env_cfg.non_loadable or [])}
        non_loadable_seq = {(b, r, t) for b in (0, 1) for r, t in non_loadable_bay}
        self.bay = BayState(self.R, self.T, row_weight_max, non_loadable_bay)
        self.sequence = StowageSequence(self.R, self.R, self.T, non_loadable_seq)
        self.tracker = RewardTracker(self.reward_cfg, self.R, self.T, row_weight_max)
        self._actual_step = 0
        return self.bay.get_matrix(), self.container_state

    def _compute_mask(self) -> torch.Tensor:
        return (self.container_state[:, 3] > 0).float()

    def _is_terminal(self) -> bool:
        return self._actual_step >= len(self.sequence) or remaining_quantity(self.container_state) == 0

    def step(
        self, action: int
    ) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
        if self._is_terminal():
            reward = self.tracker.compute()
            return self.bay.get_matrix(), self.container_state, reward, True

        new_container, valid = select_container(self.container_state, action)
        if not valid:
            done = self._is_terminal()
            return (
                self.bay.get_matrix(),
                self.container_state,
                0.0,
                done,
            )
        self.container_state = new_container
        slot = self.sequence[self._actual_step]
        bay_idx, row, tier = slot
        is_zero_group = action == self._zero_group_idx

        if is_zero_group:
            self._actual_step += 1
            done = self._is_terminal()
            return self.bay.get_matrix(), self.container_state, 0.0, done

        ctype = int(self.container_state[action, 2].item())
        pod = int(self.container_state[action, 0].item())
        weight = int(self.container_state[action, 1].item())

        allowed_type = int(self.bay.state[2, row, tier].item())
        if ctype == 2 and allowed_type not in (2, 3):
            self.container_state = new_container
            done = self._is_terminal()
            return (
                self.bay.get_matrix(),
                self.container_state,
                0.0,
                done,
            )

        self.bay = self.bay.load(row, tier, pod, weight, ctype)
        self.tracker.record_load(row, tier, pod, weight, ctype)
        self._actual_step += 1

        if ctype == 2:
            self.sequence.mark_occupied(self._actual_step - 1)

        done = self._is_terminal()
        reward = self.tracker.compute() if done else 0.0
        return self.bay.get_matrix(), self.container_state, reward, done

    def get_mask(self) -> torch.Tensor:
        return self._compute_mask()
