"""Table 9: voting ensemble vs forward-looking greedy heuristic, N = 13..17.

Requires trained checkpoints. Full: 50 instances per N. Smoke: 2 per N,
heuristic-only if no checkpoints are given (--ckpt-dir optional here).
"""
import random
import time

import numpy as np
import torch

from common import parse_args, write_csv

from crpsp.astar import solve_astar
from crpsp.ensemble import vote
from crpsp.evaluate import gap, rollout_policy
from crpsp.heuristic import solve_heuristic
from crpsp.instance import generate_instance

EXTRA = [("--ckpt-dir", {"default": None})]
NS = [13, 14, 15, 16, 17]


def main():
    args = parse_args(__doc__, extra=EXTRA)
    count = 50 if args.full else 2
    s_y = s_v = t_y = 5
    actors, cfg, device = None, None, torch.device("cpu")
    if args.ckpt_dir:
        from table8_ensemble import load_actors
        actors, cfg = load_actors(args.ckpt_dir)
    rng = random.Random(args.seed)
    rows = []
    for n in NS:
        h_gaps, h_times, v_gaps, v_times = [], [], [], []
        for _ in range(count):
            inst = generate_instance(n, s_y, s_v, t_y, rng)
            opt = solve_astar(inst, time_limit_s=1000)
            t0 = time.perf_counter()
            h = solve_heuristic(inst)
            h_times.append(time.perf_counter() - t0)
            if h.solved and opt.optimal:
                h_gaps.append(gap(h.total_ops, opt.total_ops))
            if actors is not None:
                def pol(obs, mask):
                    return vote(actors, obs, mask, device)
                out = rollout_policy(pol, inst, {"max_steps": 50})
                v_times.append(out["seconds"])
                if out["solved"] and opt.optimal:
                    v_gaps.append(gap(out["total_ops"], opt.total_ops))
        row = {"N": n, "instances": count,
               "heuristic_gap": float(np.mean(h_gaps)) if h_gaps else None,
               "heuristic_T_s": float(np.mean(h_times))}
        if actors is not None:
            row["voting_gap"] = float(np.mean(v_gaps)) if v_gaps else None
            row["voting_T_s"] = float(np.mean(v_times)) if v_times else None
        rows.append(row)
        print(row)
    write_csv(args.out or "results/table9_voting_vs_heuristic.csv", rows)


if __name__ == "__main__":
    main()
