"""CRPSP environment: relocation actions, automatic transfer closure (Section 4.2)."""
from __future__ import annotations

import numpy as np

from .instance import Instance, slot_map
from .lower_bound import lower_bound, must_precede_pairs
from .transfer import closure_crp, closure_crpsp


class CRPSPEnv:
    """State = yard matrix Theta; action = relocation (source, dest); transfers auto.

    Follows the paper's State Transition: after each relocation, every feasible
    transfer executes immediately (respecting Eq 23), then control returns to
    the agent. Reward per Eq (36) with configurable terminal bonus (notes #1).

    CRP mode (extension): retrieval by priority order instead of a stowage
    plan; `restricted=True` limits relocation sources to the stack holding
    the current retrieval target (restricted CRP variant).
    """

    def __init__(self, reward_mode: str = "designed", terminal_bonus: float = 10.0,
                 max_steps: int = 50, restricted: bool = False):
        assert reward_mode in ("designed", "simple")
        self.reward_mode = reward_mode
        self.terminal_bonus = terminal_bonus
        self.max_steps = max_steps
        self.restricted = restricted

    # ------------------------------------------------------------------ setup
    def reset(self, instance: Instance):
        self.inst = instance
        self.yard: list[list[int]] = [list(s) for s in instance.yard]
        self.vessel: list[list[int]] = [[] for _ in range(instance.s_v)]
        self.slot_of = slot_map(instance) if instance.mode == "crpsp" else None
        self.next_priority = 1
        self.precede = must_precede_pairs(instance)
        self.action_pairs = [(s, d) for s in range(instance.s_y)
                             for d in range(instance.s_y) if d != s]
        self.n_relocations = 0
        self.n_transfers = 0
        self.t = 0
        self._transfer_closure()
        self.done = self._all_placed()
        return self._obs(), self._mask()

    # ------------------------------------------------------------------- core
    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        s, d = self.action_pairs[action]
        if not self.yard[s] or len(self.yard[d]) >= self.inst.t_y:
            raise ValueError(f"invalid action {(s, d)}")
        h_before = lower_bound(self.yard, self.precede)
        self.yard[d].append(self.yard[s].pop())
        self.n_relocations += 1
        self._transfer_closure()
        self.t += 1
        self.done = self._all_placed()
        truncated = (not self.done) and self.t >= self.max_steps
        if self.reward_mode == "designed":
            h_after = lower_bound(self.yard, self.precede)
            reward = -1.0 + h_before - h_after          # Eq (36)
            if self.done:
                reward += self.terminal_bonus           # notes #1
        else:
            reward = -1.0                               # Section 5.5 baseline
        info = {"n_relocations": self.n_relocations, "n_transfers": self.n_transfers,
                "total_ops": self.n_relocations + self.n_transfers}
        return self._obs(), self._mask(), reward, self.done, truncated, info

    # -------------------------------------------------------------- internals
    def _all_placed(self) -> bool:
        return all(not s for s in self.yard)

    def _transfer_closure(self) -> None:
        if self.inst.mode == "crp":
            n, self.next_priority = closure_crp(self.yard, self.next_priority)
        else:
            n = closure_crpsp(self.yard, self.vessel, self.slot_of)
        self.n_transfers += n

    def _obs(self) -> np.ndarray:
        return self.encode(self.yard, self.inst.s_y, self.inst.t_y, self.inst.n)

    def _mask(self) -> np.ndarray:
        m = np.zeros(len(self.action_pairs), dtype=bool)
        allowed_source = None
        if self.restricted and self.inst.mode == "crp" and not self.done:
            for si, stack in enumerate(self.yard):
                if self.next_priority in stack:
                    allowed_source = si
                    break
        for i, (s, d) in enumerate(self.action_pairs):
            ok = bool(self.yard[s]) and len(self.yard[d]) < self.inst.t_y
            if allowed_source is not None:
                ok = ok and s == allowed_source
            m[i] = ok
        return m

    @staticmethod
    def encode(yard, s_y: int, t_y: int, n: int) -> np.ndarray:
        """State matrix Theta (Section 4.2): rows = stacks, entries = id/n, 0 = empty."""
        obs = np.zeros((s_y, t_y), dtype=np.float32)
        for i, stack in enumerate(yard):
            for k, c in enumerate(stack):
                obs[i, k] = c / n
        return obs
