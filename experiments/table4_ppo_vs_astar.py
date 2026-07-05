"""Table 4: trained PPO policy (greedy argmax) vs A* — gap and time.

Requires a checkpoint from train_ppo.py. Full: 200 instances of (15,5,5,5).
Smoke: 3 instances (still needs --ckpt; use a tiny local checkpoint from the
PPO smoke test if desired).
"""
import random
import time

import numpy as np
import torch

from common import parse_args, write_csv

from crpsp.astar import solve_astar
from crpsp.evaluate import gap, rollout_policy
from crpsp.instance import generate_instance
from crpsp.ppo import PPOTrainer, TrainConfig

EXTRA = [("--ckpt", {"required": True})]


def main():
    args = parse_args(__doc__, extra=EXTRA)
    count = 200 if args.full else 3
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**ck["cfg"])
    cfg.device = "cpu" if not torch.cuda.is_available() else cfg.device
    trainer = PPOTrainer(cfg)
    trainer.load(args.ckpt)
    device = trainer.device

    def policy(obs, mask):
        with torch.no_grad():
            logits = trainer.actor(torch.as_tensor(obs, device=device).unsqueeze(0),
                                   torch.as_tensor(mask, device=device).unsqueeze(0))
        return int(logits.argmax())

    rng = random.Random(args.seed)
    rows = []
    for i in range(count):
        inst = generate_instance(cfg.n, cfg.s_y, cfg.s_v, cfg.t_y, rng)
        t0 = time.perf_counter()
        opt = solve_astar(inst, time_limit_s=1000)
        astar_s = time.perf_counter() - t0
        out = rollout_policy(policy, inst, {"reward_mode": cfg.reward_mode,
                                            "terminal_bonus": cfg.terminal_bonus,
                                            "max_steps": cfg.max_steps})
        g = gap(out["total_ops"], opt.total_ops) if (out["solved"] and opt.optimal) else None
        rows.append({"instance": i, "solved": out["solved"],
                     "ppo_ops": out["total_ops"], "opt_ops": opt.total_ops,
                     "gap": g, "ppo_T_s": round(out["seconds"], 5),
                     "astar_T_s": round(astar_s, 4)})
        print(rows[-1])
    solved = [r for r in rows if r["gap"] is not None]
    if solved:
        print(f"mean gap {np.mean([r['gap'] for r in solved]):.4f} "
              f"over {len(solved)}/{len(rows)} solved")
    write_csv(args.out or "results/table4_ppo_vs_astar.csv", rows)


if __name__ == "__main__":
    main()
