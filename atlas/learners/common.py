from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LearningResult:
    recovered_index: int
    queries: int
    remaining_candidates: int
    history: list[dict]


def fidelity(
    outputs: np.ndarray,
    recovered_index: int,
    task_index: int,
) -> float:
    return float(
        np.mean(outputs[recovered_index] == outputs[task_index])
    )
