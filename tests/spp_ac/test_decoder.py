import torch
from spp_ac.models.decoder import AttentionDecoder


def test_decoder_output_shape():
    dec = AttentionDecoder(hidden_dim=128, num_gru_layers=4)
    container_enc = torch.randn(4, 13, 128)
    bay_context = torch.randn(4, 128)
    quantities = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                               [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]], dtype=torch.float32)
    probs, log_probs, d_t, h_state = dec(container_enc, bay_context, None, quantities)
    assert probs.shape == (4, 13)
    assert log_probs.shape == (4, 13)
    assert d_t.shape == (4, 128)


def test_decoder_mask_zero_quantity():
    dec = AttentionDecoder(hidden_dim=128, num_gru_layers=4)
    container_enc = torch.randn(1, 13, 128)
    bay_context = torch.randn(1, 128)
    quantities = torch.zeros(1, 13)
    quantities[0, 5] = 3.0
    probs, log_probs, _, _ = dec(container_enc, bay_context, None, quantities)
    assert probs[0, 5] > 0.5


def test_decoder_gradient_flows():
    dec = AttentionDecoder(hidden_dim=128, num_gru_layers=4)
    container_enc = torch.randn(2, 13, 128, requires_grad=True)
    bay_context = torch.randn(2, 128, requires_grad=True)
    quantities = torch.ones(2, 13) * 2
    probs, log_probs, _, _ = dec(container_enc, bay_context, None, quantities)
    loss = log_probs.mean()
    loss.backward()
    assert container_enc.grad is not None
    assert bay_context.grad is not None
