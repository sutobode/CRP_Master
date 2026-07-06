"""PPO training for HTR high-level policy.

Same PPO algorithm as baseline (clipped objective, GAE)
but with action space S_y (target selection) instead of S_y*(S_y-1).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .htr_env import HTREnv
from .instance import generate_instance
from .models_htr import TargetSelector
from .ppo import compute_gae


@dataclass
class HTRTrainConfig:
    n: int
    s_y: int
    s_v: int
    t_y: int
    terminal_bonus: float
    max_steps: int
    hidden_dim: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    batch_size: int
    instances_per_iter: int
    iterations: int
    ppo_epochs: int
    seed: int
    device: str = "auto"

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class HTRTrainer:
    def __init__(self, cfg: HTRTrainConfig):
        self.cfg = cfg
        self.device = cfg.resolved_device()
        torch.manual_seed(cfg.seed)
        self.policy = TargetSelector(cfg.s_y, cfg.t_y, cfg.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.env = HTREnv(cfg.terminal_bonus, cfg.max_steps)

    def _rollout(self, rng: random.Random) -> tuple[list[dict], dict]:
        inst = generate_instance(self.cfg.n, self.cfg.s_y, self.cfg.s_v, self.cfg.t_y, rng)
        obs, mask = self.env.reset(inst)

        ep: list[dict] = []
        solved = self.env.done
        while not self.env.done and self.env.t < self.cfg.max_steps:
            to = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            tm = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.policy(to, tm)
            dist = Categorical(logits=logits)
            a = dist.sample()
            nobs, nmask, r, done, trunc, _ = self.env.step(int(a.item()))
            ep.append({"obs": obs, "mask": mask, "action": int(a.item()),
                       "logp": float(dist.log_prob(a).item()), "reward": float(r),
                       "next_obs": nobs, "done": float(done)})
            obs, mask = nobs, nmask
            solved = done
            if done or trunc:
                break
        stats = {"solved": bool(solved), "relocations": self.env.n_relocations,
                 "reward": sum(e["reward"] for e in ep)}
        return ep, stats

    def train_iteration(self, rng: random.Random) -> dict:
        episodes, stats = [], []
        for _ in range(self.cfg.instances_per_iter):
            ep, st = self._rollout(rng)
            if ep:
                episodes.append(ep)
            stats.append(st)
        base = {"mean_reward": float(np.mean([s["reward"] for s in stats])),
                "solve_rate": float(np.mean([s["solved"] for s in stats])),
                "mean_relocations": float(np.mean([s["relocations"] for s in stats]))}
        if not episodes:
            return {**base, "policy_loss": 0.0}

        flat = [e for ep in episodes for e in ep]
        obs = torch.as_tensor(np.array([e["obs"] for e in flat]), device=self.device)
        masks = torch.as_tensor(np.array([e["mask"] for e in flat]), device=self.device)
        acts = torch.tensor([e["action"] for e in flat], device=self.device)
        logp_old = torch.tensor([e["logp"] for e in flat], device=self.device)

        # Compute returns (no critic in simple version)
        returns = []
        i = 0
        for ep in episodes:
            L = len(ep)
            r = torch.tensor([e["reward"] for e in ep])
            # Simple Monte Carlo return
            ret = torch.zeros(L)
            running = 0.0
            for t in reversed(range(L)):
                running = r[t] + self.cfg.gamma * running
                ret[t] = running
            returns.append(ret)
            i += L
        rets = torch.cat(returns).to(self.device)
        adv = rets - rets.mean()
        if adv.std() > 1e-8:
            adv = adv / (adv.std() + 1e-8)

        # PPO update
        policy_losses = []
        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(len(acts))
            for s0 in range(0, len(acts), self.cfg.batch_size):
                idx = perm[s0:s0 + self.cfg.batch_size]
                logits = self.policy(obs[idx], masks[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(acts[idx])
                ratio = torch.exp(logp - logp_old[idx])
                a_b = adv[idx]
                l_clip = -torch.min(
                    ratio * a_b,
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * a_b,
                ).mean()
                self.optimizer.zero_grad()
                l_clip.backward()
                self.optimizer.step()
                policy_losses.append(float(l_clip.item()))

        return {**base, "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0}

    def save(self, path) -> None:
        torch.save({"policy": self.policy.state_dict(), "cfg": self.cfg.__dict__}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
