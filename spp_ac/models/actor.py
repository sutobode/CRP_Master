import torch
import torch.nn as nn
from spp_ac.models.encoder import ContainerEncoder, BayEncoder
from spp_ac.models.decoder import AttentionDecoder


class Actor(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_gru_layers: int = 4):
        super().__init__()
        self.container_encoder = ContainerEncoder(hidden_dim)
        self.bay_encoder = BayEncoder(hidden_dim)
        self.decoder = AttentionDecoder(hidden_dim, num_gru_layers)

    def forward(
        self,
        bay: torch.Tensor,
        container: torch.Tensor,
        prev_embed: torch.Tensor | None = None,
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        container_enc = self.container_encoder(container)
        bay_context = self.bay_encoder(bay)
        quantities = container[:, :, 3]
        probs, log_probs, d_t, h_state = self.decoder(
            container_enc, bay_context, prev_embed, quantities, h_state,
        )
        return probs, log_probs, d_t, h_state
