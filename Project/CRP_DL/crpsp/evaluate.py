"""Evaluation utilities: optimality gap (notes #11) and policy rollouts."""
from __future__ import annotations

import time
from typing import Callable

from .env import CRPSPEnv
from .instance import Instance


def gap(alg_ops: int, opt_ops: int) -> float:
    """Gap = (total_ops_alg - total_ops_opt) / total_ops_opt (notes #11)."""
    return (alg_ops - opt_ops) / opt_ops


def rollout_policy(select_action: Callable, inst: Instance, env_kwargs: dict) -> dict:
    """Run one episode with `select_action(obs, mask) -> int`; report ops and time."""
    env = CRPSPEnv(**env_kwargs)
    t0 = time.perf_counter()
    obs, mask = env.reset(inst)
    while not env.done and env.t < env.max_steps:
        a = select_action(obs, mask)
        obs, mask, r, done, trunc, info = env.step(a)
        if done or trunc:
            break
    secs = time.perf_counter() - t0
    return {"total_ops": env.n_relocations + env.n_transfers,
            "relocations": env.n_relocations, "solved": env.done, "seconds": secs}
