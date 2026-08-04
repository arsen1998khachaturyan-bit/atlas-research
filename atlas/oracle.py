from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence


class Oracle(ABC):
    @abstractmethod
    def query(self, sequence: Sequence[int]) -> int:
        raise NotImplementedError


@dataclass
class CachedOracle(Oracle):
    fn: Callable[[tuple[int, ...]], int]
    budget: int = 100_000
    cache: dict[tuple[int, ...], int] = field(default_factory=dict)
    query_count: int = 0
    cache_hits: int = 0

    def query(self, sequence: Sequence[int]) -> int:
        key = tuple(int(x) for x in sequence)

        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        if self.query_count >= self.budget:
            raise RuntimeError("QUERY_BUDGET_EXHAUSTED")

        value = int(self.fn(key))
        self.cache[key] = value
        self.query_count += 1
        return value
