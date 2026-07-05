import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_gru_layers: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        self.gru = nn.GRU(hidden_dim, hidden_dim, num_gru_layers, batch_first=True)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.embed_proj = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_init = nn.Linear(hidden_dim, hidden_dim)

    def get_initial_state(self, bay_context: torch.Tensor) -> torch.Tensor:
        batch_size = bay_context.size(0)
        h0 = torch.tanh(self.decoder_init(bay_context))
        h0 = h0.unsqueeze(0).repeat(self.num_gru_layers, 1, 1).contiguous()
        return h0

    def forward(
        self,
        container_enc: torch.Tensor,
        bay_context: torch.Tensor,
        prev_embed: torch.Tensor | None,
        quantities: torch.Tensor,
        h_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if h_state is None:
            h_state = self.get_initial_state(bay_context)

        if prev_embed is None:
            gru_input = torch.zeros(
                bay_context.size(0), 1, self.hidden_dim,
                device=bay_context.device,
            )
        else:
            gru_input = self.embed_proj(prev_embed).unsqueeze(1)

        gru_out, h_state = self.gru(gru_input, h_state)
        d_t = gru_out.squeeze(1)

        proj_e = self.W1(container_enc)
        proj_d = self.W2(d_t).unsqueeze(1)
        u = torch.tanh(proj_e + proj_d)
        attn = torch.einsum("h,bmh->bm", self.v, u)

        mask = (quantities > 0).float()
        logits = torch.where(mask > 0, attn + quantities, attn - 1e9)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        return probs, log_probs, d_t, h_state
