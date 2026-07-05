"""Table 5: attention vs no-attention — 5 models each, convergence + gap.

Full run trains 10 models to (up to) 1000 iterations — SERVER ONLY.
Smoke: builds both variants and runs 2 iterations each (wiring check).
"""
import pathlib
import random

import yaml

from common import parse_args, write_csv

from crpsp.ppo import PPOTrainer, TrainConfig, convergence_iteration

EXTRA = [("--config", {"default": "configs/default.yaml"}),
         ("--models", {"type": int, "default": 5})]


def main():
    args = parse_args(__doc__, extra=EXTRA)
    raw = yaml.safe_load(pathlib.Path(args.config).read_text())
    rows = []
    n_models = args.models if args.full else 1
    iters = raw["train"]["iterations"] if args.full else 2
    for use_att in (True, False):
        for m in range(n_models):
            cfg = TrainConfig.from_yaml(raw)
            cfg.use_attention = use_att
            cfg.iterations = iters
            cfg.seed = args.seed + m
            trainer = PPOTrainer(cfg)
            rng = random.Random(cfg.seed)
            hist = [trainer.train_iteration(rng) for _ in range(iters)]
            conv = convergence_iteration(hist)
            rows.append({"attention": use_att, "model": m,
                         "convergence": conv if conv is not None else -1,
                         "final_solve_rate": hist[-1]["solve_rate"],
                         "final_mean_relocations": hist[-1]["mean_relocations"]})
            print(rows[-1])
            if args.full:
                out = pathlib.Path(f"checkpoints/table5_att{int(use_att)}_m{m}.pt")
                out.parent.mkdir(parents=True, exist_ok=True)
                trainer.save(out)
    write_csv(args.out or "results/table5_attention.csv", rows)


if __name__ == "__main__":
    main()
