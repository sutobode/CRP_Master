import torch
from spp_ac.models.encoder import ContainerEncoder, BayEncoder


def test_container_encoder_output_shape():
    enc = ContainerEncoder(hidden_dim=128)
    x = torch.randn(4, 13, 4)
    out = enc(x)
    assert out.shape == (4, 13, 128)


def test_container_encoder_gradient_flows():
    enc = ContainerEncoder(hidden_dim=128)
    x = torch.randn(2, 13, 4, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_bay_encoder_output_shape():
    enc = BayEncoder(hidden_dim=128)
    x = torch.randn(4, 6, 12, 6)
    out = enc(x)
    assert out.shape == (4, 128)


def test_bay_encoder_gradient_flows():
    enc = BayEncoder(hidden_dim=128)
    x = torch.randn(2, 6, 12, 6, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
