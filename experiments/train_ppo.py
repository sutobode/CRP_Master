"""Full PPO training (Algorithm 1) — SERVER ONLY.

Refuses to run without --full: training is heavy and must not run on the
development laptop (see plan Global Constraints). Saves a checkpoint and a
per-iteration metrics CSV.
"""
import pathlib
import random
import sys

import yaml

from common import parse_args, write_csv

from crpsp.ppo import PPOTrainer, TrainConfig

EXTRA = [
    ("--config", {"default": "configs/default.yaml"}),
    ("--iterations", {"type": int, "default": None}),
    ("--n", {"type": int, "default": None}),
    ("--s-y", {"type": int, "default": None}),
    ("--s-v", {"type": int, "default": None}),
    ("--t-y", {"type": int, "default": None}),
    ("--reward", {"choices": ["designed", "simple"], "default": None}),
    ("--no-attention", {"action": "store_true"}),
]


def main():
    args = parse_args(__doc__, extra=EXTRA)
    if not args.full:
        print("REFUSING to train without --full: training is server-only "
              "(heavy compute). Use tests for local verification.", file=sys.stderr)
        sys.exit(2)
    raw = yaml.safe_load(pathlib.Path(args.config).read_text())
    cfg = TrainConfig.from_yaml(raw)
    if args.iterations is not None:
        cfg.iterations = args.iterations
    if args.n is not None:
        cfg.n = args.n
    if args.s_y is not None:
        cfg.s_y = args.s_y
    if args.s_v is not None:
        cfg.s_v = args.s_v
    if args.t_y is not None:
        cfg.t_y = args.t_y
    if args.reward is not None:
        cfg.reward_mode = args.reward
    if args.no_attention:
        cfg.use_attention = False
    cfg.seed = args.seed

    trainer = PPOTrainer(cfg)
    rng = random.Random(args.seed)
    history = []
    for it in range(cfg.iterations):
        metrics = trainer.train_iteration(rng)
        metrics["iteration"] = it
        history.append(metrics)
        if it % 10 == 0:
            print(f"iter {it}: reward={metrics['mean_reward']:.2f} "
                  f"solve={metrics['solve_rate']:.2f} "
                  f"reloc={metrics['mean_relocations']:.2f}")
    out = pathlib.Path(args.out or "checkpoints/model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(out)
    write_csv(out.with_suffix(".metrics.csv"), history)
    print(f"saved checkpoint {out}")


if __name__ == "__main__":
    main()
