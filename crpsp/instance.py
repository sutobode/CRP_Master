"""Instance definition and random generator (paper Section 2 and 4.2)."""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    yard: tuple[tuple[int, ...], ...]     # bottom->top container ids per yard stack
    stowage: tuple[tuple[int, ...], ...]  # bottom->top ids per vessel stack; 0 = furthest from shore
    t_y: int                              # yard max height
    mode: str = "crpsp"                   # "crpsp" | "crp"
    priorities: tuple[int, ...] = ()      # crp mode only: ids in retrieval order

    @property
    def n(self) -> int:
        return sum(len(s) for s in self.yard)

    @property
    def s_y(self) -> int:
        return len(self.yard)

    @property
    def s_v(self) -> int:
        return len(self.stowage)


def generate_instance(n: int, s_y: int, s_v: int, t_y: int, rng: random.Random) -> Instance:
    """Random CRPSP instance per Section 4.2.

    Stowage height profile is non-increasing toward shore so Eq (23) is
    satisfiable; labels 1..N assigned shore->far, bottom->top (notes #2, #3).
    """
    if n > s_y * t_y:
        raise ValueError(f"yard capacity {s_y * t_y} cannot hold {n} containers")
    counts = [0] * s_v
    for _ in range(n):
        counts[rng.randrange(s_v)] += 1
    counts.sort(reverse=True)             # index 0 (furthest from shore) tallest
    stowage: list[list[int]] = [[0] * h for h in counts]
    label = 1
    for s in range(s_v - 1, -1, -1):      # shore side first
        for k in range(counts[s]):        # bottom up
            stowage[s][k] = label
            label += 1
    ids = list(range(1, n + 1))
    rng.shuffle(ids)
    yard: list[list[int]] = [[] for _ in range(s_y)]
    for c in ids:
        open_stacks = [i for i in range(s_y) if len(yard[i]) < t_y]
        yard[rng.choice(open_stacks)].append(c)
    return Instance(
        yard=tuple(tuple(s) for s in yard),
        stowage=tuple(tuple(s) for s in stowage),
        t_y=t_y,
    )


def slot_map(inst: Instance) -> dict[int, tuple[int, int]]:
    """Container id -> (vessel stack index, tier index)."""
    return {c: (vs, vt) for vs, stack in enumerate(inst.stowage) for vt, c in enumerate(stack)}


def load_caserta(path) -> Instance:
    """Standard-CRP loader for the Caserta/Schwarze/Voss BRP format.

    Line 1: '<n_stacks> <n_containers>'; line i+1: '<count> <ids bottom->top>'.
    Yard height limit follows the CV convention H_max = H + 2. Container ids
    double as retrieval priorities (1 = retrieved first).
    """
    import pathlib

    lines = [ln.split() for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
    n_stacks, n = int(lines[0][0]), int(lines[0][1])
    yard = []
    for row in lines[1:1 + n_stacks]:
        cnt = int(row[0])
        yard.append(tuple(int(v) for v in row[1:1 + cnt]))
    h = max((len(s) for s in yard), default=0)
    return Instance(yard=tuple(yard), stowage=(), t_y=h + 2, mode="crp",
                    priorities=tuple(range(1, n + 1)))
