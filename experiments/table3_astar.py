"""Table 3: A* solve times. Paper grid: (N,S_y,S_v,T_y) x 200 instances.

Smoke: first two configs x 3 instances.
"""
import random
import time

from common import parse_args, write_csv

from crpsp.astar import solve_astar
from crpsp.instance import generate_instance

FULL_GRID = [(12, 4, 4, 5), (12, 4, 5, 5), (15, 4, 4, 5), (15, 4, 5, 5),
             (15, 5, 5, 5), (15, 5, 6, 5), (20, 5, 5, 5), (20, 5, 6, 5)]
FULL_COUNT = 200


def main():
    args = parse_args(__doc__)
    grid = FULL_GRID if args.full else FULL_GRID[:2]
    count = FULL_COUNT if args.full else 3
    rng = random.Random(args.seed)
    rows = []
    for (n, s_y, s_v, t_y) in grid:
        times, ok = [], 0
        for _ in range(count):
            inst = generate_instance(n, s_y, s_v, t_y, rng)
            t0 = time.perf_counter()
            res = solve_astar(inst, time_limit_s=1000)
            times.append(time.perf_counter() - t0)
            ok += int(res.optimal)
        rows.append({"N": n, "S_y": s_y, "S_v": s_v, "T_y": t_y,
                     "instances": count, "solved": ok,
                     "mean_T_s": round(sum(times) / len(times), 4)})
        print(rows[-1])
    write_csv(args.out or "results/table3_astar.csv", rows)


if __name__ == "__main__":
    main()
