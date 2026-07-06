"""Table 8: ensemble learning (Voting vs Stacking) on (15,5,5,5) instances.

Requires trained checkpoints (--ckpt-dir with *.pt from train_ppo.py).
Full: 10 models, 200 eval instances, stacking trained on A* trajectories.
Smoke: whatever checkpoints exist, 3 eval instances.
"""
import pathlib
import random
import time

import numpy as np
import torch

from common import astar_node_limit, astar_time_limit, parse_args, write_csv

from crpsp.astar import solve_astar
from crpsp.ensemble import build_stacking_dataset, train_stacking, vote
from crpsp.evaluate import gap, rollout_policy
from crpsp.instance import generate_instance
from crpsp.ppo import PPOTrainer, TrainConfig

EXTRA = [("--ckpt-dir", {"required": True}),
         ("--stacking-instances", {"type": int, "default": 200})]


def load_actors(ckpt_dir):
    actors, cfg = [], None
    for p in sorted(pathlib.Path(ckpt_dir).glob("*.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**ck["cfg"])
        cfg.device = "cpu" if not torch.cuda.is_available() else cfg.device
        tr = PPOTrainer(cfg)
        tr.load(p)
        actors.append(tr.actor)
    if not actors:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")
    return actors, cfg


def main():
    args = parse_args(__doc__, extra=EXTRA)
    actors, cfg = load_actors(args.ckpt_dir)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    n_eval = 200 if args.full else 3
    n_stack = args.stacking_instances if args.full else 3
    limit = astar_time_limit(args)
    node_cap = astar_node_limit(args)
    rng = random.Random(args.seed)

    stack_insts = [generate_instance(cfg.n, cfg.s_y, cfg.s_v, cfg.t_y, rng)
                   for _ in range(n_stack)]
    X, y = build_stacking_dataset(actors, stack_insts, device, time_limit_s=limit)
    stack_pol = train_stacking(X, y, cfg.s_y * (cfg.s_y - 1)) if len(y) else None

    env_kwargs = {"reward_mode": cfg.reward_mode,
                  "terminal_bonus": cfg.terminal_bonus, "max_steps": cfg.max_steps}
    rows = []
    for method in ("voting", "stacking"):
        if method == "stacking" and stack_pol is None:
            continue
        gaps, times, solved_n = [], [], 0
        for _ in range(n_eval):
            inst = generate_instance(cfg.n, cfg.s_y, cfg.s_v, cfg.t_y, rng)
            opt = solve_astar(inst, time_limit_s=limit, node_limit=node_cap)

            if method == "voting":
                def pol(obs, mask):
                    return vote(actors, obs, mask, device)
            else:
                def pol(obs, mask):
                    feats = np.concatenate(
                        [tp for tp in ( _softmax_probs(a, obs, mask, device) for a in actors)])
                    return stack_pol.predict(feats, mask)
            out = rollout_policy(pol, inst, env_kwargs)
            times.append(out["seconds"])
            if out["solved"] and opt.optimal:
                solved_n += 1
                gaps.append(gap(out["total_ops"], opt.total_ops))
        rows.append({"method": method, "instances": n_eval, "solved": solved_n,
                     "mean_gap": float(np.mean(gaps)) if gaps else None,
                     "mean_T_s": float(np.mean(times))})
        print(rows[-1])
    write_csv(args.out or "results/table8_ensemble.csv", rows)


def _softmax_probs(actor, obs, mask, device):
    with torch.no_grad():
        logits = actor(torch.as_tensor(obs, device=device).unsqueeze(0),
                       torch.as_tensor(mask, device=device).unsqueeze(0))
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


if __name__ == "__main__":
    main()
