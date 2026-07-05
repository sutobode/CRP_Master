import torch
from spp_ac.models.actor import Actor

def test_actor_forward():
    actor = Actor(hidden_dim=128, num_gru_layers=4)
    bay = torch.randn(2, 6, 12, 6)
    container = torch.randn(2, 13, 4)
    container[:, :, 3] = torch.randint(0, 5, (2, 13)).float()
    probs, log_probs, d_t, h_state = actor(bay, container)
    assert probs.shape == (2, 13)
    assert log_probs.shape == (2, 13)
    assert d_t.shape == (2, 128)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))

def test_actor_gradient_flows():
    actor = Actor(hidden_dim=128, num_gru_layers=4)
    bay = torch.randn(1, 6, 12, 6)
    container = torch.randn(1, 13, 4)
    container[:, :, 3] = torch.randint(1, 5, (1, 13)).float()
    _, _, d_t, h_state = actor(bay, container)
    prev_embed = d_t.detach()
    probs, log_probs, _, _ = actor(bay, container, prev_embed, h_state)
    loss = -log_probs.mean()
    loss.backward()
    assert all(p.grad is not None for p in actor.parameters())
