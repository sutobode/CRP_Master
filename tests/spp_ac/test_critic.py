import torch
from spp_ac.models.critic import Critic

def test_critic_output():
    critic = Critic(hidden_dim=128)
    bay = torch.randn(4, 6, 12, 6)
    container = torch.randn(4, 13, 4)
    v = critic(bay, container)
    assert v.shape == (4,)

def test_critic_gradient_flows():
    critic = Critic(hidden_dim=128)
    bay = torch.randn(2, 6, 12, 6)
    container = torch.randn(2, 13, 4)
    v = critic(bay, container)
    loss = v.mean()
    loss.backward()
    assert all(p.grad is not None for p in critic.parameters())
