"""Post-processes results/atlas_nn_stage_b_capacity_metric.json with a
stricter "did training actually succeed" filter than the one used when that
file was generated, and reports per-layer + overall correlations between
effective-rank shrinkage and the compression-gain magnitude.

Why this exists: analyze_stage_b_capacity_metric.py's original filter
(trained_accuracy - random_accuracy >= 0.15) let a partially-failed
training run through as "succeeded" -- xor2_h256 seed 33 reached only 68%
held-out accuracy (vs. ~87% for a properly trained run at that width/task),
which is a >0.15 gain over its ~49% random-init baseline but still a
partial-training artifact of the same kind Experiment 6 already flagged for
seed 22 at the same width. This script fixes that by additionally requiring
each run's trained accuracy to be within 8 points of the best trained
accuracy observed for that exact condition (a per-condition "near ceiling"
check), re-derives the correlations, and reports both the fix and the
resulting numbers -- rather than silently regenerating the base JSON.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atlas_nn.stage_b.experiment import save_json

MIN_ACCURACY_GAIN = 0.15
MAX_GAP_FROM_CONDITION_CEILING = 0.08


def rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    raw = json.loads(Path("results/atlas_nn_stage_b_capacity_metric.json").read_text())
    all_rows = raw["rows"]

    ceiling_by_condition: dict[str, float] = defaultdict(float)
    for r in all_rows:
        ceiling_by_condition[r["condition"]] = max(
            ceiling_by_condition[r["condition"]], r["trained_accuracy"]
        )

    def training_ok(r: dict) -> bool:
        gained_enough = (r["trained_accuracy"] - r["random_accuracy"]) >= MIN_ACCURACY_GAIN
        near_ceiling = r["trained_accuracy"] >= ceiling_by_condition[r["condition"]] - MAX_GAP_FROM_CONDITION_CEILING
        return gained_enough and near_ceiling

    newly_excluded = sorted(
        {(r["condition"], r["seed"]) for r in all_rows if r["training_succeeded"] and not training_ok(r)}
    )
    print(f"Rows the original filter accepted but the stricter filter rejects: {newly_excluded}")

    valid = [
        r for r in all_rows
        if training_ok(r) and r["compression_gain"] and r["rank_shrinkage"] is not None
    ]

    per_layer = {}
    for layer in ("0", "2", "4"):
        sub = [r for r in valid if r["layer"] == layer]
        y = np.log(np.array([r["compression_gain"] for r in sub]))
        abs_shrink = np.array([
            r["capacity_random"]["shannon_effective_rank"] - r["capacity_trained"]["shannon_effective_rank"]
            for r in sub
        ])
        erank_trained = np.array([r["capacity_trained"]["shannon_effective_rank"] for r in sub])

        per_layer[layer] = {
            "n": len(sub),
            "pearson_abs_shrink_vs_log_gain": pearson(abs_shrink, y),
            "pearson_erank_trained_vs_log_gain": pearson(erank_trained, y),
            "spearman_abs_shrink_vs_log_gain": pearson(rank_transform(abs_shrink), rank_transform(y)),
        }
        print(f"layer={layer} n={len(sub)} "
              f"pearson(abs_shrink,log_gain)={per_layer[layer]['pearson_abs_shrink_vs_log_gain']:.3f} "
              f"pearson(erank_trained,log_gain)={per_layer[layer]['pearson_erank_trained_vs_log_gain']:.3f} "
              f"spearman(abs_shrink,log_gain)={per_layer[layer]['spearman_abs_shrink_vs_log_gain']:.3f}")

    y_all = np.log(np.array([r["compression_gain"] for r in valid]))
    abs_shrink_all = np.array([
        r["capacity_random"]["shannon_effective_rank"] - r["capacity_trained"]["shannon_effective_rank"]
        for r in valid
    ])
    overall = {
        "n": len(valid),
        "pearson_abs_shrink_vs_log_gain": pearson(abs_shrink_all, y_all),
        "spearman_abs_shrink_vs_log_gain": pearson(rank_transform(abs_shrink_all), rank_transform(y_all)),
    }
    print(f"\noverall n={overall['n']} "
          f"pearson(abs_shrink,log_gain)={overall['pearson_abs_shrink_vs_log_gain']:.3f} "
          f"spearman(abs_shrink,log_gain)={overall['spearman_abs_shrink_vs_log_gain']:.3f}")

    save_json(
        {
            "experiment": "atlas_nn-stage_b_capacity_metric_summary",
            "min_accuracy_gain": MIN_ACCURACY_GAIN,
            "max_gap_from_condition_ceiling": MAX_GAP_FROM_CONDITION_CEILING,
            "newly_excluded_vs_original_filter": newly_excluded,
            "n_valid": len(valid),
            "per_layer": per_layer,
            "overall": overall,
        },
        "results/atlas_nn_stage_b_capacity_metric_summary.json",
    )


if __name__ == "__main__":
    main()
