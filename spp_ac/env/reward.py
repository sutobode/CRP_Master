import torch
from spp_ac.config import RewardConfig


class RewardTracker:
    def __init__(self, config: RewardConfig, R: int, T: int, row_weight_max: list[float]):
        self.config = config
        self.R = R
        self.T = T
        self.row_weight_max = torch.tensor(row_weight_max, dtype=torch.float32)
        self.reset()

    def reset(self):
        self.N_loaded = 0
        self.f1 = 0
        self.f2_numer = 0.0
        self.f2_denom = 0.0
        self.f3 = 0
        self.overhang = 0
        self.twenty_on_forty = 0
        self.row_weights = torch.zeros(self.R)
        self._slots: list[dict] = []

    def record_load(
        self, r: int, t: int, pod: int, weight: int, ctype: int
    ):
        if t > 0 and not any(
            s["r"] == r and s["t"] == t - 1 for s in self._slots
        ):
            self.overhang += 1

        if ctype == 1:
            below = [s for s in self._slots if s["r"] == r and s["t"] == t - 1]
            if below and below[0]["ctype"] == 2:
                self.twenty_on_forty += 1

        for s in self._slots:
            if s["r"] == r and s["t"] < t and s["pod"] > pod:
                self.f1 += 1

        below = [s for s in self._slots if s["r"] == r and s["t"] == t - 1]
        if below and weight > below[0]["weight"]:
            self.f3 += 1

        lateral_dist = r - (self.R - 1) / 2.0
        self.f2_numer += lateral_dist * weight
        self.f2_denom += weight

        self.row_weights[r] += weight

        self._slots.append({"r": r, "t": t, "pod": pod, "weight": weight, "ctype": ctype})
        self.N_loaded += 1

    def compute(self) -> float:
        if self.N_loaded == 0:
            return 0.0

        m1 = self.f1 / self.N_loaded
        f2 = self.f2_numer / self.f2_denom if self.f2_denom > 0 else 0.0
        m2 = 2.0 * abs(f2) / self.R
        m3 = self.f3 / self.N_loaded

        R1 = -(self.config.lambda_1 * m1 + self.config.lambda_2 * m2 + self.config.lambda_3 * m3)

        g1 = self.overhang / self.N_loaded
        g2 = self.twenty_on_forty / self.N_loaded
        weight_excess = torch.clamp(self.row_weights - self.row_weight_max, min=0).sum().item()
        total_max = self.row_weight_max.sum().item()
        g3 = weight_excess / total_max if total_max > 0 else 0.0

        R2 = -(self.config.alpha_1 * g1 + self.config.alpha_2 * g2 + self.config.alpha_3 * g3)

        return R1 + R2
