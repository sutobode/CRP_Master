import torch
import torch.nn as nn
from spp_ac.models.encoder import ContainerEncoder, BayEncoder


class Critic(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.container_encoder = ContainerEncoder(hidden_dim)
        self.bay_encoder = BayEncoder(hidden_dim)
        self.fc1 = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1)
        self.fc3 = nn.Conv1d(hidden_dim // 2, 16, kernel_size=1)
        self.fc4 = nn.Conv1d(16, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, bay: torch.Tensor, container: torch.Tensor) -> torch.Tensor:
        e = self.container_encoder(container)
        b = self.bay_encoder(bay)
        h = torch.cat([e.mean(dim=1), b], dim=-1)
        h = h.unsqueeze(-1)
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.fc4(h)
        v = h.squeeze(-1).squeeze(-1)
        return v
