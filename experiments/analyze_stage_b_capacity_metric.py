"""Turns Experiment 6's qualitative "spare capacity" story into a
quantitative one. For each (condition, layer, seed) already characterized
by run_atlas_nn_stage_b_capacity_sweep.py, this computes each layer's
effective rank (a capacity-usage proxy, see
atlas_nn.stage_b.capacity_metrics) in both the random-init and trained
state, and checks whether the *shrinkage* in effective rank after training
predicts the *magnitude* of the compression-ratio gain already measured --
not just its sign.

This reproduces the same 6 conditions (deterministic given the same fixed
seeds already used) purely to extract this additional diagnostic from
models already behaviorally characterized in results/atlas_nn_stage_b_
capacity_sweep.json; it does not re-test whether training succeeds -- that
was already established there, and the same held-out-accuracy check is
repeated here only to exclude the known width-256 training failures from
the correlation (see docs/RESEARCH_LOG.md Experiment 6).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_nn.stage_b.capacity_metrics import capacity_metrics
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from experiments.run_atlas_nn_stage_b_capacity_sweep import SEEDS, TASKS, WIDTHS

# A trained network that isn't meaningfully better than its own random-init
# baseline didn't really "train" -- exclude it from the correlation, same
# spirit as Experiment 6's exclusion of the width-256 training failures.
MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED = 0.15


def load_compression_lookup() -> dict:
    path = Path("results/atlas_nn_stage_b_capacity_sweep.json")
    data = json.loads(path.read_text())
    return {
        (s["condition"], s["layer"], s["model_state"], s["seed"]): s["overall_best_ratio"]
        for s in data["searches"]
    }


def rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def main() -> None:
    compression_lookup = load_compression_lookup()
    rows = []

    for task_name, task_cfg in TASKS.items():
        make_dataset = task_cfg["make_dataset"]
        dataset_kwargs = task_cfg["dataset_kwargs"]
        input_dim = dataset_kwargs["n_features"]

        for hidden_dim in WIDTHS:
            condition_name = f"{task_name}_h{hidden_dim}"

            for seed in SEEDS:
                x_train, y_train = make_dataset(task_cfg["n_train"], seed=seed, **dataset_kwargs)
                x_eval, y_eval = make_dataset(task_cfg["n_eval"], seed=seed + 50_000, **dataset_kwargs)

                model = build_mlp(seed, input_dim=input_dim, hidden_dim=hidden_dim, n_hidden_layers=2)
                random_state = snapshot(model)
                random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

                train_mlp(model, x_train, y_train, epochs=task_cfg["epochs"], lr=task_cfg["lr"])
                trained_state = snapshot(model)
                trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

                training_succeeded = (trained_acc - random_acc) >= MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED

                for layer_name in linear_layer_names(model):
                    load_snapshot(model, random_state)
                    w_random = get_weight(model, layer_name).detach().numpy().copy()
                    load_snapshot(model, trained_state)
                    w_trained = get_weight(model, layer_name).detach().numpy().copy()

                    cm_random = capacity_metrics(w_random)
                    cm_trained = capacity_metrics(w_trained)

                    rank_shrinkage = None
                    if cm_random["shannon_effective_rank"] > 0:
                        rank_shrinkage = 1.0 - (
                            cm_trained["shannon_effective_rank"] / cm_random["shannon_effective_rank"]
                        )

                    ratio_random = compression_lookup.get((condition_name, layer_name, "random_init", seed))
                    ratio_trained = compression_lookup.get((condition_name, layer_name, "trained", seed))
                    compression_gain = None
                    if ratio_random and ratio_trained:
                        compression_gain = ratio_trained / ratio_random

                    row = {
                        "condition": condition_name,
                        "task": task_name,
                        "hidden_dim": hidden_dim,
                        "layer": layer_name,
                        "seed": seed,
                        "training_succeeded": training_succeeded,
                        "random_accuracy": random_acc,
                        "trained_accuracy": trained_acc,
                        "capacity_random": cm_random,
                        "capacity_trained": cm_trained,
                        "rank_shrinkage": rank_shrinkage,
                        "compression_ratio_random": ratio_random,
                        "compression_ratio_trained": ratio_trained,
                        "compression_gain": compression_gain,
                    }
                    rows.append(row)

                    rs_str = f"{rank_shrinkage:.3f}" if rank_shrinkage is not None else "None"
                    cg_str = f"{compression_gain:.3f}" if compression_gain is not None else "None"
                    print(
                        f"{condition_name:<14} layer={layer_name:<3} seed={seed:>3} "
                        f"trained_ok={training_succeeded!s:<5} "
                        f"erank_random={cm_random['shannon_effective_rank']:6.2f} "
                        f"erank_trained={cm_trained['shannon_effective_rank']:6.2f} "
                        f"rank_shrinkage={rs_str:<8} compression_gain={cg_str}"
                    )

    valid = [
        r for r in rows
        if r["training_succeeded"]
        and r["rank_shrinkage"] is not None
        and r["compression_gain"] is not None
        and r["compression_gain"] > 0
    ]
    excluded_count = len(rows) - len(valid)

    x = np.array([r["rank_shrinkage"] for r in valid])
    y = np.log(np.array([r["compression_gain"] for r in valid]))

    pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else None
    spearman_r = (
        float(np.corrcoef(rank_transform(x), rank_transform(y))[0, 1]) if len(x) > 1 else None
    )

    print(f"\nn={len(valid)} valid points, {excluded_count} excluded (failed training or missing data)")
    print(f"Pearson  r(rank_shrinkage, log compression_gain) = {pearson_r}")
    print(f"Spearman r(rank_shrinkage, log compression_gain) = {spearman_r}")

    save_json(
        {
            "experiment": "atlas_nn-stage_b_capacity_metric_analysis",
            "min_accuracy_gain_to_count_as_trained": MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED,
            "n_total": len(rows),
            "n_valid": len(valid),
            "n_excluded": excluded_count,
            "pearson_r_rank_shrinkage_vs_log_gain": pearson_r,
            "spearman_r_rank_shrinkage_vs_log_gain": spearman_r,
            "rows": rows,
        },
        "results/atlas_nn_stage_b_capacity_metric.json",
    )


if __name__ == "__main__":
    main()
