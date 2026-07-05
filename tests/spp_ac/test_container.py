import torch
from spp_ac.env.container import select_container, remaining_quantity


def test_select_valid():
    state = torch.zeros(13, 4)
    state[0] = torch.tensor([1.0, 1.0, 1.0, 5.0])
    state[1] = torch.tensor([1.0, 2.0, 1.0, 3.0])
    new_state, valid = select_container(state, 0)
    assert valid is True
    assert new_state[0, 3].item() == 4.0


def test_select_invalid():
    state = torch.zeros(13, 4)
    state[0, 3] = 0.0
    new_state, valid = select_container(state, 0)
    assert valid is False
    assert new_state[0, 3].item() == 0.0


def test_remaining():
    state = torch.zeros(13, 4)
    state[0, 3] = 5.0
    state[1, 3] = 3.0
    assert remaining_quantity(state) == 8
