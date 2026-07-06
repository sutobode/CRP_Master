"""High-level HTR environment: action = target stack, reward = -relocations."""
from __future__ import annotations

import numpy as np

from .instance import Instance, slot_map
from .lower_bound import lower_bound, must_precede_pairs
from .transfer import closure_crpsp
from .subproblem import solve_one_relocation


class HTREnv:
    """High-level env for HTR: action = stack index (0..S_y-1).
    Step: relocate top blocker of chosen stack → transfer closure.
    Reward: -(relocations per macro-step), terminal bonus on done.
    """

    def __init__(self, terminal_bonus: float = 10.0, max_steps: int = 50):
        self.terminal_bonus = terminal_bonus
        self.max_steps = max_steps

    def reset(self, instance: Instance):
        self.inst = instance
        self.yard: list[list[int]] = [list(s) for s in instance.yard]
        self.vessel: list[list[int]] = [[] for _ in range(instance.s_v)]
        self.slot_of = slot_map(instance) if instance.mode == "crpsp" else None
        self.s_y = instance.s_y
        self.t_y = instance.t_y
        self.n_relocations = 0
        self.n_transfers = 0
        self.t = 0
        self._transfer_closure()
        self.done = self._all_placed()
        return self._obs(), self._mask()

    def _is_transferable(self, c: int) -> bool:
        vs, vt = self.slot_of[c]
        return len(self.vessel[vs]) == vt and all(
            len(self.vessel[j]) >= len(self.vessel[vs]) + 1
            for j in range(vs)
        )

    def step(self, action: int):
        if self.done:
            raise RuntimeError("episode finished")
        target_stack = action

        if not self.yard[target_stack]:
            self._transfer_closure()
            self.t += 1
            self.done = self._all_placed()
            return self._obs(), self._mask(), -1.0, self.done, False, {}

        # Try transfers: if top is transferable, transfer directly
        top = self.yard[target_stack][-1]
        if self._is_transferable(top):
            self.vessel[self.slot_of[top][0]].append(self.yard[target_stack].pop())
            self.n_transfers += 1
            self._transfer_closure()
            self.t += 1
            self.done = self._all_placed()
            reward = 0.0
            if self.done:
                reward += self.terminal_bonus
            return self._obs(), self._mask(), reward, self.done, False, {}

        # Not transferable: relocate top blocker
        n_reloc = 0
        for _ in range(self.t_y):
            if not self.yard[target_stack]:
                break
            c = self.yard[target_stack][-1]
            if self._is_transferable(c):
                break
            d = solve_one_relocation(self.yard, target_stack, self.inst)
            if d < 0:
                break
            self.yard[d].append(self.yard[target_stack].pop())
            n_reloc += 1

        self.n_relocations += n_reloc
        self._transfer_closure()
        self.t += 1
        self.done = self._all_placed()

        reward = -float(n_reloc)
        if self.done:
            reward += self.terminal_bonus

        return self._obs(), self._mask(), reward, self.done, False, {}

    def _all_placed(self) -> bool:
        return all(not s for s in self.yard)

    def _transfer_closure(self) -> None:
        n = closure_crpsp(self.yard, self.vessel, self.slot_of)
        self.n_transfers += n

    def _obs(self) -> np.ndarray:
        return self.encode(self.yard, self.s_y, self.t_y, self.inst.n)

    def _mask(self) -> np.ndarray:
        m = np.zeros(self.s_y, dtype=bool)
        for i in range(self.s_y):
            m[i] = bool(self.yard[i])
        return m

    @staticmethod
    def encode(yard, s_y: int, t_y: int, n: int) -> np.ndarray:
        obs = np.zeros((s_y, t_y), dtype=np.float32)
        for i, stack in enumerate(yard):
            for k, c in enumerate(stack):
                obs[i, k] = c / n
        return obs
