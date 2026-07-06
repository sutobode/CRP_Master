#!/usr/bin/env python3
"""HTR training: high-level policy (target selection) + subproblem solver.

Usage:
    # Smoke test (laptop, ~1 min):
    python experiments/train_htr.py --smoke
    
    # Full training (Colab, ~2 GPU hours):
    python experiments/train_htr.py --full --iterations 2000
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np

from crpsp.ppo_htr import HTRTrainConfig, HTRTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--s_y", type=int, default=5)
    parser.add_argument("--s_v", type=int, default=5)
    parser.add_argument("--t_y", type=int, default=5)
    parser.add_argument("--subproblem", type=str, default="heuristic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="htr_checkpoint.pt")
    args = parser.parse_args()

    if args.smoke:
        cfg = HTRTrainConfig(
            n=args.n, s_y=args.s_y, s_v=args.s_v, t_y=args.t_y,
            terminal_bonus=10.0, max_steps=50,
            hidden_dim=128, lr=5e-4, gamma=0.4, gae_lambda=0.9,
            clip_eps=0.15, batch_size=10, instances_per_iter=10,
            iterations=5, ppo_epochs=1, seed=args.seed,
        )
    elif args.full:
        cfg = HTRTrainConfig(
            n=args.n, s_y=args.s_y, s_v=args.s_v, t_y=args.t_y,
            terminal_bonus=10.0, max_steps=50,
            hidden_dim=128, lr=5e-4, gamma=0.4, gae_lambda=0.9,
            clip_eps=0.15, batch_size=10, instances_per_iter=10,
            iterations=args.iterations, ppo_epochs=1, seed=args.seed,
            subproblem_solver=args.subproblem,
        )
    else:
        # Default: balanced run
        cfg = HTRTrainConfig(
            n=args.n, s_y=args.s_y, s_v=args.s_v, t_y=args.t_y,
            terminal_bonus=10.0, max_steps=50,
            hidden_dim=128, lr=5e-4, gamma=0.4, gae_lambda=0.9,
            clip_eps=0.15, batch_size=10, instances_per_iter=10,
            iterations=args.iterations, ppo_epochs=1, seed=args.seed,
            subproblem_solver=args.subproblem,
        )

    print(f"HTR config: n={cfg.n}, s_y={cfg.s_y}, s_v={cfg.s_v}, t_y={cfg.t_y}")
    print(f"Action space: {cfg.s_y} (vs baseline {cfg.s_y * (cfg.s_y - 1)})")
    print(f"Iterations: {cfg.iterations}")
    print(f"Device: {cfg.resolved_device()}")

    trainer = HTRTrainer(cfg)
    rng = random.Random(cfg.seed)
    history = []
    t0 = time.perf_counter()

    for it in range(cfg.iterations):
        result = trainer.train_iteration(rng)
        history.append(result)
        if (it + 1) % 20 == 0 or it == 0 or it == cfg.iterations - 1:
            elapsed = time.perf_counter() - t0
            print(
                f"Iter {it+1:4d}/{cfg.iterations} | "
                f"reward {result['mean_reward']:+.2f} | "
                f"solve {result['solve_rate']:.0%} | "
                f"reloc {result['mean_relocations']:.1f} | "
                f"loss {result.get('policy_loss', 0):.4f} | "
                f"{elapsed:5.1f}s"
            )

    trainer.save(args.out)
    print(f"Saved to {args.out}")

    # Final stats
    solve_rates = [h["solve_rate"] for h in history[-100:]]
    final_mean_reloc = np.mean([h["mean_relocations"] for h in history[-100:]])
    print(f"Final 100 iters: solve rate {np.mean(solve_rates):.0%}, "
          f"mean relocations {final_mean_reloc:.1f}")


if __name__ == "__main__":
    main()
