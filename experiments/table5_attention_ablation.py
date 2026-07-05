"""Table 5: attention vs no-attention — 5 models each, convergence + gap + T(s).

Paper Section 5.4 evaluates both groups on the SAME fixed set of 50 problem
instances and reports Convergence, Gap (vs A*), and T(s) (review finding:
the original version of this script only reported training-time solve_rate/
mean_relocations on fresh random instances, and reused the Table 6-7
iteration cap of 1000 — too low for this table, since the paper's own
no-attention group converges as late as iteration 2241). `--iterations`
below defaults with headroom above that; see REPRODUCTION_NOTES.md #22.

Full run trains models to (up to) `--iterations` — SERVER ONLY.
Smoke: builds both variants and runs 2 iterations each (wiring check).
"""
import pathlib
import random

import numpy as np
import torch
import yaml

from common import astar_node_limit, astar_time_limit, parse_args, write_csv

from crpsp.astar import solve_astar
from crpsp.evaluate import gap, rollout_policy
from crpsp.instance import generate_instance
from crpsp.ppo import PPOTrainer, TrainConfig, convergence_iteration

EXTRA = [("--config", {"default": "configs/default.yaml"}),
         ("--models", {"type": int, "default": 5}),
         ("--iterations", {"type": int, "default": 3000}),
         ("--eval-instances", {"type": int, "default": 50})]


def main():
    args = parse_args(__doc__, extra=EXTRA)
    raw = yaml.safe_load(pathlib.Path(args.config).read_text())
    n_models = args.models if args.full else 1
    iters = args.iterations if args.full else 2
    n_eval = args.eval_instances if args.full else 2

    base_cfg = TrainConfig.from_yaml(raw)
    eval_rng = random.Random(args.seed)
    eval_insts = [generate_instance(base_cfg.n, base_cfg.s_y, base_cfg.s_v,
                                     base_cfg.t_y, eval_rng)
                  for _ in range(n_eval)]
    limit = astar_time_limit(args)
    node_cap = astar_node_limit(args)
    eval_opts = [solve_astar(inst, time_limit_s=limit, node_limit=node_cap)
                 for inst in eval_insts]

    rows = []
    for use_att in (True, False):
        for m in range(n_models):
            cfg = TrainConfig.from_yaml(raw)
            cfg.use_attention = use_att
            cfg.iterations = iters
            cfg.seed = args.seed + m + 1
            trainer = PPOTrainer(cfg)
            train_rng = random.Random(cfg.seed)
            hist = [trainer.train_iteration(train_rng) for _ in range(iters)]
            conv = convergence_iteration(hist)

            def policy(obs, mask):
                with torch.no_grad():
                    logits = trainer.actor(
                        torch.as_tensor(obs, device=trainer.device).unsqueeze(0),
                        torch.as_tensor(mask, device=trainer.device).unsqueeze(0))
                return int(logits.argmax())

            gaps, times, solved_n = [], [], 0
            for inst, opt in zip(eval_insts, eval_opts):
                out = rollout_policy(policy, inst, {"reward_mode": cfg.reward_mode,
                                                     "terminal_bonus": cfg.terminal_bonus,
                                                     "max_steps": cfg.max_steps})
                times.append(out["seconds"])
                if out["solved"] and opt.optimal:
                    solved_n += 1
                    gaps.append(gap(out["total_ops"], opt.total_ops))

            rows.append({"attention": use_att, "model": m,
                         "convergence": conv if conv is not None else -1,
                         "final_solve_rate": hist[-1]["solve_rate"],
                         "final_mean_relocations": hist[-1]["mean_relocations"],
                         "eval_instances": n_eval, "eval_solved": solved_n,
                         "gap": float(np.mean(gaps)) if gaps else None,
                         "T_s": float(np.mean(times))})
            print(rows[-1])
            if args.full:
                out = pathlib.Path(f"checkpoints/table5_att{int(use_att)}_m{m}.pt")
                out.parent.mkdir(parents=True, exist_ok=True)
                trainer.save(out)
    write_csv(args.out or "results/table5_attention.csv", rows)


if __name__ == "__main__":
    main()
