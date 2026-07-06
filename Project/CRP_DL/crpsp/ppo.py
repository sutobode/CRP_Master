"""PPO for the CRPSP — faithful implementation of the paper's Algorithm 1.

Per iteration (outer loop): generate M fresh instances, roll each out with the
current policy (max T steps), recompute values with the current critic
(lines 13-18), compute GAE advantages (Eq 33-34) and TD(0) value targets
(line 17), then a single shuffled mini-batch pass updating actor (clipped
objective, Eq 35) and critic (squared error, line 23). No entropy bonus, no
gradient clipping (notes #8); targets bootstrap 0 at terminal states (notes #9).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from .env import CRPSPEnv
from .instance import generate_instance
from .models import Actor, Critic


def compute_gae(rewards, values, next_values, dones, gamma: float, lam: float):
    """A_t = sum (gamma*lam)^{k-t} delta_k (Eq 34); y_t = r_t + gamma*V(s_{t+1})(1-done)."""
    T = len(rewards)
    adv = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        nv = next_values[t] * (1.0 - dones[t])
        delta = rewards[t] + gamma * nv - values[t]          # Eq (33)
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        adv[t] = gae
    targets = rewards + gamma * next_values * (1.0 - dones)  # Algorithm 1 line 17
    return adv, targets


@dataclass
class TrainConfig:
    n: int
    s_y: int
    s_v: int
    t_y: int
    reward_mode: str
    terminal_bonus: float
    max_steps: int
    hidden_dim: int
    use_attention: bool
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

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TrainConfig":
        return cls(**cfg["env"], **cfg["model"], **cfg["train"])

    def resolved_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class PPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = cfg.resolved_device()
        torch.manual_seed(cfg.seed)
        self.actor = Actor(cfg.s_y, cfg.t_y, cfg.hidden_dim, cfg.use_attention).to(self.device)
        self.critic = Critic(cfg.s_y, cfg.t_y, cfg.hidden_dim, cfg.use_attention).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.env = CRPSPEnv(cfg.reward_mode, cfg.terminal_bonus, cfg.max_steps)

    # ------------------------------------------------------------- collection
    def _rollout(self, rng: random.Random) -> tuple[list[dict], dict]:
        inst = generate_instance(self.cfg.n, self.cfg.s_y, self.cfg.s_v, self.cfg.t_y, rng)
        obs, mask = self.env.reset(inst)
        ep: list[dict] = []
        solved = self.env.done
        while not self.env.done and self.env.t < self.cfg.max_steps:
            to = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            tm = torch.as_tensor(mask, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.actor(to, tm)
            dist = Categorical(logits=logits)
            a = dist.sample()
            nobs, nmask, r, done, trunc, info = self.env.step(int(a.item()))
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

    # -------------------------------------------------------------- one iter
    def train_iteration(self, rng: random.Random) -> dict:
        episodes, stats = [], []
        for _ in range(self.cfg.instances_per_iter):       # Algorithm 1 lines 3-12
            ep, st = self._rollout(rng)
            if ep:
                episodes.append(ep)
            stats.append(st)
        base = {"mean_reward": float(np.mean([s["reward"] for s in stats])),
                "solve_rate": float(np.mean([s["solved"] for s in stats])),
                "mean_relocations": float(np.mean([s["relocations"] for s in stats]))}
        if not episodes:                                   # all solved at reset
            return {**base, "actor_loss": 0.0, "critic_loss": 0.0}

        flat = [e for ep in episodes for e in ep]
        obs = torch.as_tensor(np.array([e["obs"] for e in flat]), device=self.device)
        masks = torch.as_tensor(np.array([e["mask"] for e in flat]), device=self.device)
        acts = torch.tensor([e["action"] for e in flat], device=self.device)
        logp_old = torch.tensor([e["logp"] for e in flat], device=self.device)

        # Algorithm 1 lines 13-18: values recomputed with the CURRENT critic
        with torch.no_grad():
            all_v = self.critic(obs).cpu()
            next_obs = torch.as_tensor(np.array([e["next_obs"] for e in flat]),
                                       device=self.device)
            all_nv = self.critic(next_obs).cpu()
        adv_list, tgt_list = [], []
        i = 0
        for ep in episodes:
            L = len(ep)
            r = torch.tensor([e["reward"] for e in ep])
            d = torch.tensor([e["done"] for e in ep])
            a, y = compute_gae(r, all_v[i:i + L], all_nv[i:i + L], d,
                               self.cfg.gamma, self.cfg.gae_lambda)
            adv_list.append(a)
            tgt_list.append(y)
            i += L
        adv = torch.cat(adv_list).to(self.device)
        tgt = torch.cat(tgt_list).to(self.device)

        # Algorithm 1 lines 19-25: shuffle, mini-batches, clipped update
        n_tr = len(acts)
        actor_losses, critic_losses = [], []
        for _ in range(self.cfg.ppo_epochs):
            perm = torch.randperm(n_tr)
            for s0 in range(0, n_tr, self.cfg.batch_size):
                idx = perm[s0:s0 + self.cfg.batch_size]
                logits = self.actor(obs[idx], masks[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(acts[idx])
                ratio = torch.exp(logp - logp_old[idx])
                a_b = adv[idx]
                l_clip = -torch.min(
                    ratio * a_b,
                    torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * a_b,
                ).mean()                                    # Eq (35)
                self.opt_actor.zero_grad()
                l_clip.backward()
                self.opt_actor.step()
                v = self.critic(obs[idx])
                l_v = ((tgt[idx] - v) ** 2).mean()          # Algorithm 1 line 23
                self.opt_critic.zero_grad()
                l_v.backward()
                self.opt_critic.step()
                actor_losses.append(float(l_clip.item()))
                critic_losses.append(float(l_v.item()))

        return {**base, "actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses))}

    # -------------------------------------------------------------------- io
    def save(self, path) -> None:
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "cfg": self.cfg.__dict__}, path)

    def load(self, path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])


def convergence_iteration(history: list[dict], window: int = 20) -> int | None:
    """Convergence iteration for Tables 5-7 (paper leaves this undefined; notes #10):
    end index of the first `window`-long run of iterations that are all fully
    solved (solve_rate == 1.0) with stable mean relocations. None if never."""
    if len(history) < window:
        return None
    for end in range(window, len(history) + 1):
        w = history[end - window:end]
        if all(m["solve_rate"] >= 1.0 for m in w):
            rel = [m["mean_relocations"] for m in w]
            if max(rel) - min(rel) < 1e-9 or all(r >= rel[0] - 1e-9 for r in rel[1:]):
                return end
    return None
