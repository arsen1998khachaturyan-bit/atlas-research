from __future__ import annotations

import random

import numpy as np

from atlas.learners.common import LearningResult


class RandomLearner:
    def __init__(self, domain, seed: int = 0):
        self.domain = list(domain)
        self.seed = int(seed)

    def fit(
        self,
        outputs: np.ndarray,
        task_index: int,
    ) -> LearningResult:
        rng = random.Random(self.seed)
        order = list(range(len(self.domain)))
        rng.shuffle(order)

        version = np.arange(outputs.shape[0])
        history = []

        for query_count, sequence_index in enumerate(order, start=1):
            label = outputs[task_index, sequence_index]
            version = version[
                outputs[version, sequence_index] == label
            ]

            history.append({
                "queries": query_count,
                "remaining_candidates": int(len(version)),
            })

            if len(version) <= 1:
                break

        recovered = int(version[0]) if len(version) else 0

        return LearningResult(
            recovered_index=recovered,
            queries=len(history),
            remaining_candidates=int(len(version)),
            history=history,
        )
