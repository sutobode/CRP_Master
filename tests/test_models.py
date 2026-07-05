import torch

from crpsp.models import Actor, Critic, RowSelfAttention


def test_attention_shapes_eq38_39():
    att = RowSelfAttention(n_tiers=5, d=128)
    x = torch.randn(7, 6, 5)                      # B=7, m=6 stacks, n=5 tiers
    out = att(x)
    assert out.shape == (7, 6, 128)


def test_attention_weights_are_convex_combinations():
    att = RowSelfAttention(3, 4)
    x = torch.randn(1, 2, 3)
    q, k = att.wq(x), att.wk(x)
    w = torch.softmax(q @ k.transpose(-2, -1) / (4 ** 0.5), dim=-1)
    assert torch.allclose(w.sum(-1), torch.ones(1, 2), atol=1e-6)


def test_actor_masks_invalid_actions():
    actor = Actor(s_y=4, t_y=5)
    obs = torch.randn(2, 4, 5)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, 0] = False
    logits = actor(obs, mask)
    assert logits.shape == (2, 12)
    assert (logits[:, 0] < -1e8).all()
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs[:, 0], torch.zeros(2), atol=1e-6)


def test_critic_scalar_and_grads_flow():
    critic = Critic(s_y=4, t_y=5)
    obs = torch.randn(3, 4, 5, requires_grad=True)
    v = critic(obs)
    assert v.shape == (3,)
    v.sum().backward()
    assert obs.grad is not None


def test_no_attention_variant():
    actor = Actor(s_y=4, t_y=5, use_attention=False)
    logits = actor(torch.randn(2, 4, 5), torch.ones(2, 12, dtype=torch.bool))
    assert logits.shape == (2, 12)
