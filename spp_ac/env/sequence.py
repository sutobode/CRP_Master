class StowageSequence:
    def __init__(
        self,
        R_left: int,
        R_right: int,
        T: int,
        non_loadable: set[tuple[int, int, int]] | None = None,
    ):
        self.R_left = R_left
        self.R_right = R_right
        self.T = T
        self.non_loadable = non_loadable or set()
        self._slots: list[tuple[int, int, int]] = []
        self._occupied: set[int] = set()
        self._paired_map: dict[int, int] = {}
        self._slot_of: dict[tuple[int, int, int], int] = {}
        self._build()

    def _build(self):
        R = max(self.R_left, self.R_right)
        for tier in range(self.T):
            left = 0
            right = 2 * R - 1
            while left < right:
                if left < self.R_left and (0, left, tier) not in self.non_loadable:
                    self._add_slot(0, left, tier)
                if right >= R:
                    rrow = right - R
                    if rrow < self.R_right and (1, rrow, tier) not in self.non_loadable:
                        self._add_slot(1, rrow, tier)
                left += 1
                right -= 1

    def _add_slot(self, bay: int, row: int, tier: int):
        idx = len(self._slots)
        self._slots.append((bay, row, tier))
        self._slot_of[(bay, row, tier)] = idx
        paired = (1 - bay, row, tier)
        if paired in self._slot_of:
            self._paired_map[idx] = self._slot_of[paired]
            self._paired_map[self._slot_of[paired]] = idx

    def mark_occupied(self, idx: int) -> list[int]:
        self._occupied.add(idx)
        released: list[int] = []
        if idx in self._paired_map:
            paired_idx = self._paired_map[idx]
            if paired_idx not in self._occupied:
                self._occupied.add(paired_idx)
                released = [paired_idx]
        return released

    def get_paired_slot(self, idx: int) -> tuple[int, int, int] | None:
        if idx in self._paired_map:
            return self._slots[self._paired_map[idx]]
        return None

    def __len__(self) -> int:
        return len(self._slots) - len(self._occupied)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        actual = 0
        for i in range(len(self._slots)):
            if i not in self._occupied:
                if actual == idx:
                    return self._slots[i]
                actual += 1
        raise IndexError(f"Index {idx} out of range")
