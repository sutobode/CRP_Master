import numpy as np
import torch


def rip(N: int, P: int, rng: np.random.Generator | None = None) -> list[int]:
    if rng is None:
        rng = np.random.default_rng()
    if P == 1:
        return [N]
    parts = rng.exponential(1.0, size=P)
    parts = (parts / parts.sum() * N).astype(np.int64)
    diff = N - parts.sum()
    idx = rng.choice(P, abs(diff), replace=False)
    for i in idx[:abs(diff)]:
        parts[i] += 1 if diff > 0 else -1
    return parts.tolist()


def CfgDataset(
    P: int, W: int, E: int, N: int, S: int,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    if rng is None:
        rng = np.random.default_rng()
    M = E * P * W + 1
    data = torch.zeros(S, M, 4, dtype=torch.float32)
    p_zero = rng.uniform(0, 0.1)
    N0 = int(N * p_zero)
    Nr = N - N0

    for i in range(S):
        x = rip(Nr, P, rng)
        for port_idx, n_port in enumerate(x):
            ww = rip(n_port, W, rng)
            for wc_idx, n_wc in enumerate(ww):
                ee = rip(n_wc, E, rng)
                for t_idx, n_t in enumerate(ee):
                    g = port_idx * W * E + wc_idx * E + t_idx
                    data[i, g, 0] = float(port_idx + 1)
                    data[i, g, 1] = float(wc_idx + 1)
                    data[i, g, 2] = float(t_idx + 1)
                    data[i, g, 3] = float(n_t)
        data[i, M - 1, 3] = float(N0)

    return data
