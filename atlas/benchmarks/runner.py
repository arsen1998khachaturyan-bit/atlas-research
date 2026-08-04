from __future__ import annotations

import json
import random
from pathlib import Path

from atlas.domain import all_binary_sequences
from atlas.hypothesis import (
    behavior_matrix,
    generate_hypothesis_pool,
)
from atlas.learners import (
    ActiveInformationGainLearner,
    PassiveLearner,
    RandomLearner,
)
from atlas.learners.common import fidelity


def run_multiseed_validation(
    *,
    pool_size: int,
    max_length: int,
    tasks_per_seed: int,
    seeds: tuple[int, ...],
) -> dict:
    domain = all_binary_sequences(max_length)
    rows = []
    seed_summaries = []

    for seed in seeds:
        pool = generate_hypothesis_pool(
            target_size=pool_size,
            domain=domain,
            seed=seed,
        )
        outputs = behavior_matrix(pool, domain)

        task_rng = random.Random(seed + 99_999)
        eligible = list(range(len(pool) // 3, len(pool)))
        task_indices = task_rng.sample(
            eligible,
            min(tasks_per_seed, len(eligible)),
        )

        seed_rows = []

        for task_number, task_index in enumerate(task_indices):
            learners = {
                "passive": PassiveLearner(
                    domain,
                    seed=seed + 1_000 + task_number,
                ),
                "random": RandomLearner(
                    domain,
                    seed=seed + 2_000 + task_number,
                ),
                "active": ActiveInformationGainLearner(domain),
            }

            result_row = {
                "seed": seed,
                "task_number": task_number,
                "task": pool[task_index].name,
            }

            for name, learner in learners.items():
                result = learner.fit(outputs, task_index)
                result_row[name] = {
                    "queries": result.queries,
                    "remaining_candidates": result.remaining_candidates,
                    "fidelity": fidelity(
                        outputs,
                        result.recovered_index,
                        task_index,
                    ),
                }

            rows.append(result_row)
            seed_rows.append(result_row)

        seed_summaries.append({
            "seed": seed,
            "tasks": len(seed_rows),
            "mean_passive_queries": sum(
                row["passive"]["queries"] for row in seed_rows
            ) / len(seed_rows),
            "mean_random_queries": sum(
                row["random"]["queries"] for row in seed_rows
            ) / len(seed_rows),
            "mean_active_queries": sum(
                row["active"]["queries"] for row in seed_rows
            ) / len(seed_rows),
            "active_wins_vs_passive": sum(
                row["active"]["queries"] < row["passive"]["queries"]
                for row in seed_rows
            ),
            "active_wins_vs_random": sum(
                row["active"]["queries"] < row["random"]["queries"]
                for row in seed_rows
            ),
            "all_active_full_fidelity": all(
                row["active"]["fidelity"] == 1.0
                for row in seed_rows
            ),
        })

    return {
        "experiment": "atlas-multiseed-validation",
        "pool_size": pool_size,
        "max_length": max_length,
        "tasks_per_seed": tasks_per_seed,
        "seeds": list(seeds),
        "total_tasks": len(rows),
        "mean_passive_queries": sum(
            row["passive"]["queries"] for row in rows
        ) / len(rows),
        "mean_random_queries": sum(
            row["random"]["queries"] for row in rows
        ) / len(rows),
        "mean_active_queries": sum(
            row["active"]["queries"] for row in rows
        ) / len(rows),
        "active_wins_vs_passive": sum(
            row["active"]["queries"] < row["passive"]["queries"]
            for row in rows
        ),
        "active_wins_vs_random": sum(
            row["active"]["queries"] < row["random"]["queries"]
            for row in rows
        ),
        "all_active_full_fidelity": all(
            row["active"]["fidelity"] == 1.0
            for row in rows
        ),
        "seed_summaries": seed_summaries,
        "results": rows,
    }


def save_json(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
