import argparse
import torch
from pathlib import Path
from spp_ac.config import Config
from spp_ac.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPP-AC: Slot Stowage Optimization via Improved Actor-Critic")
    parser.add_argument("--config", type=str, default="spp_ac/config.yaml", help="Path to config YAML")
    parser.add_argument("--iterations", type=int, default=None, help="Override training iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Save checkpoint path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config.from_yaml(args.config)

    if args.iterations:
        config.train.num_iterations = args.iterations
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr

    if args.device == "auto":
        device = config.resolved_device()
    else:
        device = torch.device(args.device)

    trainer = Trainer(config, device)

    trainer.train()

    if args.checkpoint:
        torch.save({
            "actor_state_dict": trainer.actor.state_dict(),
            "critic_state_dict": trainer.critic.state_dict(),
            "actor_optim_state_dict": trainer.actor_optim.state_dict(),
            "critic_optim_state_dict": trainer.critic_optim.state_dict(),
            "config": config,
        }, args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
