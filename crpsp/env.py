"""CRPSP environment: relocation actions, automatic transfer closure (Section 4.2)."""
from __future__ import annotations

import numpy as np

from .instance import Instance, slot_map
from .lower_bound import lower_bound, must_precede_pairs


class CRPSPEnv:
    """State = yard matrix Theta; action = relocation (source, dest); transfers auto.

    Follows the paper's State Transition: after each relocation, every feasible
    transfer executes immediately (respecting Eq 23), then control returns to
    the agent. Reward per Eq (36) with configurable terminal bonus (notes #1).
    """

    def __init__(self, reward_mode: str = "designed", terminal_bonus: float = 10.0,
                 max_steps: int = 50):
        assert reward_mode in ("designed", "simple")
        self.reward_mode = reward_mode
        self.terminal_bonus = terminal_bonus
        self.max_steps = max_steps

    # ------------------------------------------------------------------ setup
    def reset(self, instance: Instance):
        self.inst = instance
        self.yard: list[list[int]] = [list(s) for s in instance.yard]
        self.vessel: list[list[int]] = [[] for _ in range(instance.s_v)]
        self.slot_of = slot_map(instance)
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

    def _height_ok(self, vs: int) -> bool:
        """Eq (23): after adding to vessel stack vs, nearer-shore stacks stay <= farther."""
        h_new = len(self.vessel[vs]) + 1
        return all(len(self.vessel[j]) >= h_new for j in range(vs))

    def _transfer_closure(self) -> None:
        moved = True
        while moved:
            moved = False
            for stack in self.yard:
                if not stack:
                    continue
                c = stack[-1]
                vs, vt = self.slot_of[c]
                if len(self.vessel[vs]) == vt and self._height_ok(vs):
                    stack.pop()
                    self.vessel[vs].append(c)
                    self.n_transfers += 1
                    moved = True

    def _obs(self) -> np.ndarray:
        return self.encode(self.yard, self.inst.s_y, self.inst.t_y, self.inst.n)

    def _mask(self) -> np.ndarray:
        m = np.zeros(len(self.action_pairs), dtype=bool)
        for i, (s, d) in enumerate(self.action_pairs):
            m[i] = bool(self.yard[s]) and len(self.yard[d]) < self.inst.t_y
        return m

    @staticmethod
    def encode(yard, s_y: int, t_y: int, n: int) -> np.ndarray:
        """State matrix Theta (Section 4.2): rows = stacks, entries = id/n, 0 = empty."""
        obs = np.zeros((s_y, t_y), dtype=np.float32)
        for i, stack in enumerate(yard):
            for k, c in enumerate(stack):
                obs[i, k] = c / n
        return obs
