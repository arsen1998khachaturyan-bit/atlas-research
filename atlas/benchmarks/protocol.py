from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationProtocol:
    pool_size: int = 1200
    max_length: int = 8
    tasks_per_seed: int = 30
    seeds: tuple[int, ...] = (101, 202, 303, 404, 505)
