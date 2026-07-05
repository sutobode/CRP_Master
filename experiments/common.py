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
