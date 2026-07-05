"""Tables 6-7: designed vs simple reward — convergence across the parameter grid.

Full grid (paper): LR {1e-4, 5e-4} x gamma {0.4, 0.5, 0.7} x N {5, 10, 15}
x eps {0.01, 0.05, 0.15} x 6 runs, capped at 1000 iterations — SERVER ONLY.
Smoke: one cell, 2 iterations per reward mode.
"""
import pathlib
import random

import yaml

from common import parse_args, write_csv

from crpsp.ppo import PPOTrainer, TrainConfig, convergence_iteration

LRS = [1e-4, 5e-4]
GAMMAS = [0.4, 0.5, 0.7]
NS = [5, 10, 15]
EPSILONS = [0.01, 0.05, 0.15]
RUNS = 6
CAP = 1000


def main():
    args = parse_args(__doc__, extra=[("--config", {"default": "configs/default.yaml"})])
    raw = yaml.safe_load(pathlib.Path(args.config).read_text())
    if args.full:
        grid = [(lr, g, n, e, run) for lr in LRS for g in GAMMAS
                for n in NS for e in EPSILONS for run in range(RUNS)]
        iters = CAP
    else:
        grid = [(5e-4, 0.4, 5, 0.15, 0)]
        iters = 2
    rows = []
    for (lr, gamma, n, eps, run) in grid:
        for reward_mode in ("simple", "designed"):
            cfg = TrainConfig.from_yaml(raw)
            cfg.lr, cfg.gamma, cfg.n, cfg.clip_eps = lr, gamma, n, eps
            cfg.reward_mode = reward_mode
            cfg.iterations = iters
            cfg.seed = args.seed + run
            trainer = PPOTrainer(cfg)
            rng = random.Random(cfg.seed)
            hist = []
            conv = None
            for _ in range(iters):
                hist.append(trainer.train_iteration(rng))
                conv = convergence_iteration(hist)
                if conv is not None:
                    break                       # paper reports iterations-to-converge
            rows.append({"lr": lr, "gamma": gamma, "N": n, "eps": eps, "run": run,
                         "reward": reward_mode,
                         "convergence": conv if conv is not None else iters})
            print(rows[-1])
    write_csv(args.out or "results/table6_7_reward.csv", rows)


if __name__ == "__main__":
    main()
