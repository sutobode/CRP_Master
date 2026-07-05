"""Shared CLI plumbing for experiment scripts.

--smoke (default) = tiny sizes to verify wiring on a laptop;
--full = paper-scale runs, intended for the training server ONLY.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def parse_args(description: str, extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--full", action="store_true",
                   help="paper-scale run (server only); default is a tiny smoke run")
    p.add_argument("--out", default=None, help="CSV output path")
    p.add_argument("--seed", type=int, default=42)
    for flag, kwargs in (extra or []):
        p.add_argument(flag, **kwargs)
    args = p.parse_args()
    args.smoke = not args.full
    return args


def astar_time_limit(args: argparse.Namespace) -> float:
    """A* per-instance time budget: paper uses up to 1000s (Table 3/4 methodology)
    for --full runs; smoke runs cap tightly so a hard random instance can never
    turn a wiring check into a multi-minute, multi-GB search (Global Constraints)."""
    return 1000.0 if args.full else 5.0


def astar_node_limit(args: argparse.Namespace) -> int | None:
    """Extra memory safety net for smoke runs (the time check only happens once
    per popped node, so an unlucky hard instance could still balloon memory)."""
    return None if args.full else 200_000


def write_csv(path: str | pathlib.Path, rows: list[dict]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
