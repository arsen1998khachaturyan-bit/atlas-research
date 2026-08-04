from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Program:
    name: str
    complexity_bits: int
    fn: Callable[[tuple[int, ...]], int]

    def run(self, sequence: Sequence[int]) -> int:
        return int(self.fn(tuple(int(x) for x in sequence)))
