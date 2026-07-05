"""Wiring checks only: scripts must run their smoke path quickly, and the
training entry point must REFUSE to run without --full (Global Constraints)."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(script, extra=(), timeout=300):
    r = subprocess.run([PY, str(ROOT / "experiments" / script), *extra],
                       capture_output=True, text=True, timeout=timeout,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    return r


def test_table3_smoke():
    _run("table3_astar.py")


def test_table2_smoke():
    _run("table2_mip.py")


def test_train_ppo_refuses_without_full():
    r = subprocess.run([PY, str(ROOT / "experiments" / "train_ppo.py")],
                       capture_output=True, text=True, timeout=60, cwd=str(ROOT))
    assert r.returncode != 0
    assert "REFUSING" in r.stderr


def test_table5_smoke():
    _run("table5_attention_ablation.py", timeout=600)


def test_table9_smoke_heuristic_only():
    _run("table9_voting_vs_heuristic.py", timeout=600)
