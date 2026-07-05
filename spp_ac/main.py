import argparse
import torch
from pathlib import Path
from spp_ac.config import Config
from spp_ac.training.trainer import Trainer
from spp_ac.generate import generate_plan, plot_stowage_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPP-AC: Slot Stowage Optimization via Improved Actor-Critic")
    parser.add_argument("--config", type=str, default="spp_ac/config.yaml", help="Path to config YAML")
    sub = parser.add_subparsers(dest="command")

    train_parser = sub.add_parser("train", help="Train a model")
    train_parser.add_argument("--iterations", type=int, default=None, help="Override training iterations")
    train_parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train_parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda")
    train_parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt", help="Save checkpoint path")
    train_parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    gen_parser = sub.add_parser("generate", help="Generate stowage plans from trained model")
    gen_parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    gen_parser.add_argument("--num-instances", type=int, default=1, help="Number of instances")
    gen_parser.add_argument("--greedy", action="store_true", default=True, help="Greedy decoding")
    gen_parser.add_argument("--plot", type=str, default=None, help="Save plot to path")
    gen_parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, or cuda")

    return parser.parse_args()


def _resolve_device(device_str: str, config: Config) -> torch.device:
    if device_str == "auto":
        return config.resolved_device()
    return torch.device(device_str)


def cmd_train(args: argparse.Namespace, config: Config) -> None:
    if args.iterations:
        config.train.num_iterations = args.iterations
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.lr = args.lr

    device = _resolve_device(args.device, config)
    trainer = Trainer(config, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} (iteration {trainer.start_iteration})")

    trainer.train()

    if args.checkpoint:
        trainer.save_checkpoint(args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")


def cmd_generate(args: argparse.Namespace, config: Config) -> None:
    device = _resolve_device(args.device, config)
    trainer = Trainer(config, device)
    trainer.load_checkpoint(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    plans = trainer.generate_plan(num_instances=args.num_instances, greedy=args.greedy)

    for i, plan in enumerate(plans):
        print(f"\nInstance {i + 1}: reward={plan['reward']:.4f}")
        for c in plan["containers"]:
            if c["container_type"] == 0:
                ctype = "empty"
            elif c["container_type"] == 2:
                ctype = "40ft"
            else:
                ctype = "20ft"
            print(f"  Group {c['action']:3d} | POD={c['pod']} | W={c['weight_class']} | {ctype}")

        if args.plot:
            plot_path = Path(args.plot)
            if len(plans) > 1:
                save_path = plot_path.parent / f"{plot_path.stem}_{i}{plot_path.suffix}"
            else:
                save_path = plot_path
            plot_stowage_plan(plan["bay_state"], save_path=str(save_path),
                              title=f"Stowage Plan - Instance {i + 1}")
            print(f"Plot saved to {save_path}")


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)

    if args.command == "generate":
        cmd_generate(args, config)
    else:
        cmd_train(args, config)


if __name__ == "__main__":
    main()
