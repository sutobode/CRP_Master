"""Actor-critic networks with row-wise self-attention (Eq 38-39, Fig. 9, notes #7).

Separate actor and critic networks per Fig. 9; each applies single-head scaled
dot-product self-attention across yard-stack rows of the state matrix, then a
hidden-dim-128 MLP head (Table 10). The no-attention variants (Section 5.4
ablation) flatten the raw state matrix instead.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RowSelfAttention(nn.Module):
    """Single-head scaled dot-product attention over yard-stack rows (Eq 38-39)."""

    def __init__(self, n_tiers: int, d: int):
        super().__init__()
        self.d = d
        self.wq = nn.Linear(n_tiers, d, bias=False)   # Q = S W_Q
        self.wk = nn.Linear(n_tiers, d, bias=False)   # K = S W_K
        self.wv = nn.Linear(n_tiers, d, bias=False)   # V = S W_V

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B, m, n)
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.d), dim=-1)
        return att @ v                                        # (B, m, d)


class _Trunk(nn.Module):
    def __init__(self, s_y: int, t_y: int, hidden_dim: int, use_attention: bool):
        super().__init__()
        self.use_attention = use_attention
        if use_attention:
            self.att = RowSelfAttention(t_y, hidden_dim)
            in_dim = s_y * hidden_dim
        else:
            self.att = None
            in_dim = s_y * t_y
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:     # obs: (B, s_y, t_y)
        h = self.att(obs) if self.use_attention else obs
        return self.fc(h.flatten(1))


class Actor(nn.Module):
    def __init__(self, s_y: int, t_y: int, hidden_dim: int = 128, use_attention: bool = True):
        super().__init__()
        self.trunk = _Trunk(s_y, t_y, hidden_dim, use_attention)
        self.head = nn.Linear(hidden_dim, s_y * (s_y - 1))

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.trunk(obs))
        return logits.masked_fill(~mask, -1e9)


class Critic(nn.Module):
    def __init__(self, s_y: int, t_y: int, hidden_dim: int = 128, use_attention: bool = True):
        super().__init__()
        self.trunk = _Trunk(s_y, t_y, hidden_dim, use_attention)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs)).squeeze(-1)
