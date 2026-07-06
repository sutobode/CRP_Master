"""HTR high-level policy network: selects target stack (S_y actions)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .models import RowSelfAttention


class TargetSelector(nn.Module):
    """High-level policy: given yard state, select which stack to clear next.

    Action space: S_y (one per yard stack).
    Architecture: same self-attention trunk as baseline Actor,
                  but output dim = S_y instead of S_y*(S_y-1).
    """

    def __init__(self, s_y: int, t_y: int, hidden_dim: int = 128):
        super().__init__()
        self.att = RowSelfAttention(t_y, hidden_dim)
        in_dim = s_y * hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, s_y)
        )

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.att(obs)
        logits = self.fc(h.flatten(1))
        return logits.masked_fill(~mask, -1e9)
