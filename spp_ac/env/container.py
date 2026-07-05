import torch


def select_container(state: torch.Tensor, idx: int) -> tuple[torch.Tensor, bool]:
    new_state = state.clone()
    if new_state[idx, 3] > 0:
        new_state[idx, 3] -= 1
        return new_state, True
    return new_state, False


def remaining_quantity(state: torch.Tensor) -> int:
    return int(state[:, 3].sum().item())
