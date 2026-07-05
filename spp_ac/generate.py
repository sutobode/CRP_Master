import torch
import numpy as np
from pathlib import Path
from spp_ac.config import Config
from spp_ac.data.cdg import CfgDataset
from spp_ac.env.spp_env import SlotStowageEnv
from spp_ac.models.actor import Actor


def _import_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def generate_plan(
    actor: Actor,
    config: Config,
    num_instances: int = 1,
    greedy: bool = True,
    device: torch.device | None = None,
) -> list[dict]:
    if device is None:
        device = torch.device("cpu")
    actor.eval()
    rng = np.random.default_rng(42)
    data = CfgDataset(
        P=config.env.num_ports,
        W=config.env.num_weight_classes,
        E=config.env.num_container_types,
        N=100,
        S=num_instances,
        rng=rng,
    )
    env = SlotStowageEnv(config.env, config.reward, data.to(device))
    plans = []

    with torch.no_grad():
        for _ in range(num_instances):
            bay, container = env.reset()
            bay = bay.to(device).unsqueeze(0)
            container = container.to(device).unsqueeze(0)
            h_state = None
            prev_embed = None
            plan = {"slots": [], "containers": []}

            done = False
            while not done:
                mask = env.get_mask().to(device)
                probs, _, d_t, h_state = actor(bay, container, prev_embed, h_state)
                scaled_probs = probs * mask
                probs_sum = scaled_probs.sum(dim=-1, keepdim=True)
                scaled_probs = scaled_probs / (probs_sum + 1e-10)

                if greedy:
                    action = scaled_probs.argmax(dim=-1)
                else:
                    dist = torch.distributions.Categorical(scaled_probs)
                    action = dist.sample()

                action_scalar = action.item()
                new_bay, new_container, reward, done = env.step(action_scalar)
                container_info = new_container[action_scalar].cpu().tolist()
                plan["containers"].append({
                    "action": action_scalar,
                    "pod": int(container_info[0]),
                    "weight_class": int(container_info[1]),
                    "container_type": int(container_info[2]),
                })
                bay = new_bay.to(device).unsqueeze(0)
                container = new_container.to(device).unsqueeze(0)
                prev_embed = d_t

            plan["reward"] = reward
            plan["bay_state"] = bay.squeeze(0).cpu()
            plans.append(plan)

    return plans


def plot_stowage_plan(
    bay_state: torch.Tensor,
    save_path: str | Path | None = None,
    title: str = "Stowage Plan",
):
    plt = _import_plt()
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

    R = bay_state.size(1)
    T = bay_state.size(2)
    pod_map = bay_state[3].numpy()
    weight_map = bay_state[4].numpy()
    type_map = bay_state[5].numpy()
    loadable = bay_state[0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(title, fontsize=14)

    cmap_pod = plt.cm.Set2
    pod_norm = plt.Normalize(vmin=-0.5, vmax=max(pod_map.max(), 0.5))
    axes[0].imshow(pod_map, cmap=cmap_pod, norm=pod_norm, aspect="auto")
    axes[0].set_title("Port of Discharge")
    axes[0].set_xlabel("Tier")
    axes[0].set_ylabel("Row")
    mask_invalid = loadable == -1
    axes[0].imshow(np.ma.masked_where(~mask_invalid, mask_invalid),
                   cmap="gray", aspect="auto", alpha=0.3)

    cmap_w = plt.cm.Blues
    axes[1].imshow(weight_map, cmap=cmap_w, aspect="auto")
    axes[1].set_title("Weight Class")
    axes[1].set_xlabel("Tier")
    axes[1].set_ylabel("Row")
    axes[1].imshow(np.ma.masked_where(~mask_invalid, mask_invalid),
                   cmap="gray", aspect="auto", alpha=0.3)

    type_display = np.where(type_map > 0, type_map, np.nan)
    cmap_t = plt.cm.Set1
    im3 = axes[2].imshow(type_display, cmap=cmap_t, aspect="auto", vmin=0.5, vmax=2.5)
    axes[2].set_title("Container Type (1=20ft, 2=40ft)")
    axes[2].set_xlabel("Tier")
    axes[2].set_ylabel("Row")
    axes[2].imshow(np.ma.masked_where(~mask_invalid, mask_invalid),
                   cmap="gray", aspect="auto", alpha=0.3)

    fig.colorbar(im3, ax=axes[2], ticks=[1, 2])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
