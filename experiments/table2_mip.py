"""Table 2: CP-SAT (paper: Gurobi) solve times on small CRPSP instances.

Full grid straight from the paper; one instance per row. Smoke: two smallest
configs with a 60 s limit.
"""
import random
import time

from common import parse_args, write_csv

from crpsp.heuristic import solve_heuristic
from crpsp.instance import generate_instance
from crpsp.mip_cpsat import solve_mip

FULL_GRID = [(5, 2, 2, 4), (5, 3, 3, 5), (6, 3, 3, 5),
             (7, 3, 3, 5), (9, 3, 3, 5), (10, 3, 3, 5)]


def main():
    args = parse_args(__doc__)
    grid = FULL_GRID if args.full else FULL_GRID[:2]
    limit = 10800.0 if args.full else 60.0
    rng = random.Random(args.seed)
    rows = []
    for (n, s_y, s_v, t_y) in grid:
        inst = generate_instance(n, s_y, s_v, t_y, rng)
        h = solve_heuristic(inst)
        horizon = (h.total_ops if h.solved else 3 * n) + 2
        t0 = time.perf_counter()
        res = solve_mip(inst, horizon=horizon, time_limit_s=limit)
        rows.append({"N": n, "S_y": s_y, "S_v": s_v, "T_y": t_y,
                     "status": res.status, "target_reached": res.target_reached,
                     "total_ops": res.total_ops, "relocations": res.relocations,
                     "T_s": round(time.perf_counter() - t0, 3)})
        print(rows[-1])
    write_csv(args.out or "results/table2_mip.csv", rows)


if __name__ == "__main__":
    main()
