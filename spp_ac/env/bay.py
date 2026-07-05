import torch


class BayState:
    def __init__(
        self,
        R: int,
        T: int,
        row_weight_max: list[float],
        non_loadable: set[tuple[int, int]] | None = None,
    ):
        self.R = R
        self.T = T
        self.row_weight_max = torch.tensor(row_weight_max, dtype=torch.float32)
        self.non_loadable = non_loadable or set()
        self.state = torch.zeros(6, R, T, dtype=torch.float32)
        self._init_state()

    def _init_state(self):
        for r in range(self.R):
            for t in range(self.T):
                if (r, t) in self.non_loadable:
                    self.state[0, r, t] = -1.0
                    self.state[1, r, t] = 0.0
                else:
                    self.state[0, r, t] = 0.0
                    self.state[1, r, t] = float(self.row_weight_max[r])
                    self.state[2, r, t] = 3.0

    def get_matrix(self) -> torch.Tensor:
        return self.state.clone()

    def load(
        self, r: int, t: int, pod: int, weight: int, ctype: int
    ) -> "BayState":
        new = BayState.__new__(BayState)
        new.R, new.T = self.R, self.T
        new.row_weight_max = self.row_weight_max.clone()
        new.non_loadable = self.non_loadable
        new.state = self.state.clone()
        new.state[0, r, t] = 1.0
        new.state[3, r, t] = float(pod)
        new.state[4, r, t] = float(weight)
        new.state[5, r, t] = float(ctype)
        new.state[2, r, t] = float(ctype)
        new.state[1, r, :] -= float(weight)
        return new

    def can_load(self, r: int, t: int, ctype: int) -> bool:
        if (r, t) in self.non_loadable:
            return False
        if self.state[0, r, t] != 0.0:
            return False
        allowed = int(self.state[2, r, t].item())
        if allowed not in (3, ctype):
            return False
        row_remaining = self.state[1, r, t].item()
        return row_remaining >= 0
