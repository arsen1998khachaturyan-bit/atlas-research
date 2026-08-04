from __future__ import annotations

import numpy as np

from atlas.learners.common import LearningResult


class ActiveInformationGainLearner:
    def __init__(self, domain, max_queries: int = 100):
        self.domain = list(domain)
        self.max_queries = int(max_queries)

    def fit(
        self,
        outputs: np.ndarray,
        task_index: int,
    ) -> LearningResult:
        version = np.arange(outputs.shape[0])
        unqueried = np.ones(len(self.domain), dtype=bool)
        history = []

        while (
            len(version) > 1
            and unqueried.any()
            and len(history) < self.max_queries
        ):
            candidate_indices = np.flatnonzero(unqueried)
            matrix = outputs[np.ix_(version, candidate_indices)]

            ones = matrix.sum(axis=0)
            zeros = len(version) - ones
            scores = np.minimum(ones, zeros)

            best_local = int(np.argmax(scores))
            sequence_index = int(candidate_indices[best_local])

            label = outputs[task_index, sequence_index]
            version = version[
                outputs[version, sequence_index] == label
            ]
            unqueried[sequence_index] = False

            history.append({
                "queries": len(history) + 1,
                "remaining_candidates": int(len(version)),
                "sequence": list(self.domain[sequence_index]),
            })

        recovered = int(version[0]) if len(version) else 0

        return LearningResult(
            recovered_index=recovered,
            queries=len(history),
            remaining_candidates=int(len(version)),
            history=history,
        )
