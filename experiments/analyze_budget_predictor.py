"""Experiment 34: turns Experiments 21-33's central finding (compression
family choice depends on layer depth and training origin) into a cheap
PREDICTOR that could replace a full 31-config budget search on a new
model, and measures how much of the achievable compression ratio it
recovers, and how much compute it saves, via leave-one-model-out
cross-validation on the five real pretrained checkpoints already
budget-searched in this project (zero new compression sweeps -- this
reuses existing result JSONs entirely).

Also stress-tests the predictor against the fine-tuning-trajectory
checkpoints from Experiments 31-33, where "training origin" is not a
clean binary category, to see where an origin-blind deployment would
go wrong.

Reproduce with `python -m experiments.analyze_budget_predictor`.
"""
from __future__ import annotations

import json

from atlas_nn.stage_c_real.budget_predictor import evaluate_on_trajectory, evaluate_predictor


def summarize(rows: list[dict], label: str) -> None:
    n = len(rows)
    exact = sum(r["exact_match"] for r in rows)
    mean_recovery = sum(r["recovery_fraction"] for r in rows) / n
    print(f"\n=== {label} (n={n}) ===")
    print(f"exact family match: {exact}/{n} ({100*exact/n:.0f}%)")
    print(f"mean recovery fraction (unweighted, per-row): {mean_recovery:.3f}")


def byte_weighted_summary(rows: list[dict], label: str) -> None:
    """The practically meaningful number: total storage footprint if you
    used the predictor's choices vs. the true full-search optimum,
    across all layers combined -- not an average of per-layer ratios,
    which an outlier miss on one large layer can distort either way."""
    total_original = sum(r["original_bytes"] for r in rows)
    total_predicted = sum(r["predicted_compressed_bytes"] for r in rows)
    total_true = sum(r["true_compressed_bytes"] for r in rows)
    aggregate_predicted_ratio = total_original / total_predicted
    aggregate_true_ratio = total_original / total_true
    print(f"\n=== {label}: byte-weighted aggregate ===")
    print(f"aggregate compression ratio, predictor: {aggregate_predicted_ratio:.2f}x")
    print(f"aggregate compression ratio, full search (optimal): {aggregate_true_ratio:.2f}x")
    print(f"predictor achieves {100*aggregate_predicted_ratio/aggregate_true_ratio:.1f}% of optimal aggregate compression")


def main() -> None:
    result = evaluate_predictor()
    rows = result["per_row"]
    summarize(rows, "Leave-one-model-out, 5 real pretrained models")
    byte_weighted_summary(rows, "Leave-one-model-out, 5 real pretrained models")

    mean_compute = sum(r["compute_fraction"] for r in rows) / len(rows)
    print(f"\nmean compute fraction used (vs. full 31-config search): {mean_compute:.3f}")

    print("\nPer-row detail:")
    for r in rows:
        match = "OK" if r["exact_match"] else "MISS"
        print(
            f"  [{match}] {r['model']:<24} {r['depth']:<5} {r['layer_type']:<12} "
            f"pred={r['predicted_family']:<17} true={r['true_family']:<17} "
            f"recovery={r['recovery_fraction']:.2f} compute={r['compute_fraction']:.2f}"
        )

    print("\nBy origin category:")
    for origin in ("from_scratch", "derived"):
        subset = [r for r in rows if r["origin"] == origin]
        summarize(subset, f"origin={origin}")
        byte_weighted_summary(subset, f"origin={origin}")

    traj_rows = evaluate_on_trajectory()
    print("\n\n=== Trajectory stress test (Experiments 31-33 checkpoints) ===")
    for assumed_origin in ("from_scratch", "derived"):
        subset = [r for r in traj_rows if r["assumed_origin"] == assumed_origin]
        summarize(subset, f"assumed_origin={assumed_origin}, all checkpoints")
        for ckpt in sorted({r["checkpoint"] for r in subset}):
            ckpt_rows = [r for r in subset if r["checkpoint"] == ckpt]
            summarize(ckpt_rows, f"  assumed_origin={assumed_origin}, checkpoint={ckpt}")

    with open("results/atlas_nn_budget_predictor_analysis.json", "w") as f:
        json.dump({"real_models": rows, "trajectory": traj_rows}, f, indent=2, default=str)
    print("\nWrote results/atlas_nn_budget_predictor_analysis.json")


if __name__ == "__main__":
    main()
