import torch
import torch.nn as nn


class ContainerEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(4, hidden_dim // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return out


class BayEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(6, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.avgpool(x)
        x = x.flatten(1)
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out.squeeze(1)
        return out
