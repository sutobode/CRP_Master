import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from spp_ac.config import Config, set_seed
from spp_ac.data.cdg import CfgDataset
from spp_ac.env.spp_env import SlotStowageEnv
from spp_ac.models.actor import Actor
from spp_ac.models.critic import Critic


class Trainer:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        set_seed(config.train.seed)
        self.start_iteration = 0
        self.actor = Actor(config.model.hidden_dim, config.model.num_gru_layers).to(device)
        self.critic = Critic(config.model.hidden_dim).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=config.train.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=config.train.lr)

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
            "config": self.config,
            "start_iteration": self.start_iteration,
        }, path)

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        self.start_iteration = ckpt.get("start_iteration", 0)

    def _sample_batch(self) -> SlotStowageEnv:
        rng = np.random.default_rng()
        data = CfgDataset(
            P=self.config.env.num_ports,
            W=self.config.env.num_weight_classes,
            E=self.config.env.num_container_types,
            N=100,
            S=self.config.train.batch_size,
            rng=rng,
        )
        return SlotStowageEnv(self.config.env, self.config.reward, data.to(self.device))

    def train(self, num_iterations: int | None = None) -> None:
        if num_iterations is None:
            num_iterations = self.config.train.num_iterations
        env = self._sample_batch()
        total = self.start_iteration + num_iterations

        for iteration in range(self.start_iteration, total):
            episode_log_probs: list[torch.Tensor] = []
            episode_rewards: list[float] = []
            episode_values: list[float] = []

            for _ in range(self.config.train.batch_size):
                bay, container = env.reset()
                bay = bay.to(self.device).unsqueeze(0)
                container = container.to(self.device).unsqueeze(0)

                v0 = self.critic(bay, container)
                log_probs: list[torch.Tensor] = []
                h_state = None
                prev_embed = None

                done = False
                while not done:
                    mask = env.get_mask().to(self.device)
                    probs, _, d_t, h_state = self.actor(
                        bay, container, prev_embed, h_state,
                    )
                    scaled_probs = probs * mask
                    probs_sum = scaled_probs.sum(dim=-1, keepdim=True)
                    scaled_probs = torch.where(probs_sum > 0, scaled_probs / probs_sum, torch.zeros_like(scaled_probs))
                    dist = torch.distributions.Categorical(scaled_probs.detach())
                    action = dist.sample()
                    log_prob = torch.distributions.Categorical(scaled_probs).log_prob(action)
                    log_probs.append(log_prob)

                    action_scalar = action.item()
                    new_bay, new_container, reward, done = env.step(action_scalar)
                    bay = new_bay.to(self.device).unsqueeze(0)
                    container = new_container.to(self.device).unsqueeze(0)
                    prev_embed = d_t

                episode_log_probs.append(torch.stack(log_probs).sum())
                episode_rewards.append(reward)
                episode_values.append(v0)

            rewards_t = torch.tensor(episode_rewards, device=self.device)
            values_t = torch.stack(episode_values)
            advantages = rewards_t - values_t
            log_probs_t = torch.stack(episode_log_probs)

            # Policy gradient (Eq 27) — gradient ascent
            actor_loss = -(advantages.detach() * log_probs_t).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.train.grad_clip)
            self.actor_optim.step()

            # Value gradient (Eq 28) — gradient descent
            critic_loss = (advantages ** 2).mean()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.train.grad_clip)
            self.critic_optim.step()

            if iteration % 100 == 0 or iteration == total - 1:
                avg_reward = rewards_t.mean().item()
                avg_value = values_t.mean().item()
                print(f"Iter {iteration:5d}/{total} | R={avg_reward:.4f} | V={avg_value:.4f} | "
                      f"Adv={advantages.mean().item():.4f} | ActorL={actor_loss.item():.6f} | CriticL={critic_loss.item():.6f}")

        self.start_iteration = total

    def generate_plan(
        self,
        num_instances: int = 1,
        greedy: bool = True,
    ) -> list[dict]:
        from spp_ac.generate import generate_plan as _generate
        return _generate(self.actor, self.config, num_instances, greedy, self.device)
