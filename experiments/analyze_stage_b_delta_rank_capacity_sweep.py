"""Cross-check for Experiment 25: does delta-rank fraction (the effective
rank of `W_trained - W_random`, as a fraction of that layer's max rank)
predict compression-gain MAGNITUDE, not just its direction on one task --
the same kind of replication that turned Experiment 7's final-matrix
effective-rank finding from "one observation" into "a quantitative,
multi-condition relationship" (via Experiments 5-6's capacity sweep).

Reproduces the same 6 conditions (2 tasks x 3 widths) x 3 seeds already
characterized by run_atlas_nn_stage_b_capacity_sweep.py (deterministic,
same fixed seeds), computes delta-rank fraction per layer, and correlates
it against the compression_gain already measured there -- using the same
"near ceiling" training-success filter Experiment 7's analysis needed
(see docs/RESEARCH_LOG.md Experiment 6/7 and
experiments/summarize_stage_b_capacity_metric.py) from the start, rather
than as a two-pass fix.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atlas_nn.stage_b.capacity_metrics import capacity_metrics
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from experiments.run_atlas_nn_stage_b_capacity_sweep import SEEDS, TASKS, WIDTHS

MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED = 0.15
MAX_GAP_FROM_CONDITION_CEILING = 0.08


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


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    compression_lookup = load_compression_lookup()
    rows = []
    ceiling_by_condition: dict[str, float] = defaultdict(float)

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
                ceiling_by_condition[condition_name] = max(ceiling_by_condition[condition_name], trained_acc)

                for layer_name in linear_layer_names(model):
                    load_snapshot(model, random_state)
                    w_random = get_weight(model, layer_name).detach().numpy().copy()
                    load_snapshot(model, trained_state)
                    w_trained = get_weight(model, layer_name).detach().numpy().copy()

                    delta = w_trained - w_random
                    cm_delta = capacity_metrics(delta)
                    delta_rank_fraction = (
                        cm_delta["shannon_effective_rank"] / cm_delta["max_rank"] if cm_delta["max_rank"] > 0 else None
                    )
                    relative_change = (
                        float(np.linalg.norm(delta) / np.linalg.norm(w_random)) if np.linalg.norm(w_random) > 0 else None
                    )

                    ratio_random = compression_lookup.get((condition_name, layer_name, "random_init", seed))
                    ratio_trained = compression_lookup.get((condition_name, layer_name, "trained", seed))
                    compression_gain = None
                    if ratio_random and ratio_trained:
                        compression_gain = ratio_trained / ratio_random

                    rows.append({
                        "condition": condition_name,
                        "task": task_name,
                        "hidden_dim": hidden_dim,
                        "layer": layer_name,
                        "seed": seed,
                        "random_accuracy": random_acc,
                        "trained_accuracy": trained_acc,
                        "delta_rank_fraction": delta_rank_fraction,
                        "relative_weight_change": relative_change,
                        "compression_ratio_random": ratio_random,
                        "compression_ratio_trained": ratio_trained,
                        "compression_gain": compression_gain,
                    })

    def training_ok(r: dict) -> bool:
        gained_enough = (r["trained_accuracy"] - r["random_accuracy"]) >= MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED
        near_ceiling = r["trained_accuracy"] >= ceiling_by_condition[r["condition"]] - MAX_GAP_FROM_CONDITION_CEILING
        return gained_enough and near_ceiling

    for r in rows:
        r["training_succeeded"] = training_ok(r)
        drf_str = f"{r['delta_rank_fraction']:.3f}" if r["delta_rank_fraction"] is not None else "None"
        cg_str = f"{r['compression_gain']:.3f}" if r["compression_gain"] else "None"
        print(
            f"{r['condition']:<14} layer={r['layer']:<3} seed={r['seed']:>3} "
            f"trained_ok={r['training_succeeded']!s:<5} delta_rank_frac={drf_str:<8} "
            f"rel_change={r['relative_weight_change']:.3f} compression_gain={cg_str}",
            flush=True,
        )

    valid = [
        r for r in rows
        if r["training_succeeded"]
        and r["delta_rank_fraction"] is not None
        and r["compression_gain"] is not None
        and r["compression_gain"] > 0
    ]
    excluded_count = len(rows) - len(valid)

    x = np.array([r["delta_rank_fraction"] for r in valid])
    y = np.log(np.array([r["compression_gain"] for r in valid]))
    overall_pearson = pearson(x, y)
    overall_spearman = pearson(rank_transform(x), rank_transform(y))

    print(f"\nn={len(valid)} valid points, {excluded_count} excluded (failed training or missing data)", flush=True)
    print(f"overall: pearson(delta_rank_fraction, log_gain)={overall_pearson} spearman={overall_spearman}", flush=True)

    per_layer = {}
    for layer in ("0", "2", "4"):
        sub = [r for r in valid if r["layer"] == layer]
        if len(sub) < 3:
            continue
        xs = np.array([r["delta_rank_fraction"] for r in sub])
        ys = np.log(np.array([r["compression_gain"] for r in sub]))
        per_layer[layer] = {
            "n": len(sub),
            "pearson": pearson(xs, ys),
            "spearman": pearson(rank_transform(xs), rank_transform(ys)),
            "mean_delta_rank_fraction": float(np.mean(xs)),
        }
        print(f"layer={layer} n={len(sub)} pearson={per_layer[layer]['pearson']} "
              f"spearman={per_layer[layer]['spearman']} "
              f"mean_delta_rank_fraction={per_layer[layer]['mean_delta_rank_fraction']:.3f}", flush=True)

    per_condition = {}
    for condition_name in sorted({r["condition"] for r in valid}):
        sub = [r for r in valid if r["condition"] == condition_name]
        for layer in ("0", "2"):
            layer_sub = [r for r in sub if r["layer"] == layer]
            if not layer_sub:
                continue
            key = f"{condition_name}_layer{layer}"
            per_condition[key] = {
                "n": len(layer_sub),
                "mean_delta_rank_fraction": float(np.mean([r["delta_rank_fraction"] for r in layer_sub])),
                "mean_compression_gain": float(np.mean([r["compression_gain"] for r in layer_sub])),
            }

    save_json(
        {
            "experiment": "atlas_nn-stage_b_delta_rank_capacity_sweep",
            "min_accuracy_gain_to_count_as_trained": MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED,
            "max_gap_from_condition_ceiling": MAX_GAP_FROM_CONDITION_CEILING,
            "n_total": len(rows),
            "n_valid": len(valid),
            "n_excluded": excluded_count,
            "overall_pearson": overall_pearson,
            "overall_spearman": overall_spearman,
            "per_layer": per_layer,
            "per_condition_layer0_vs_layer2": per_condition,
            "rows": rows,
        },
        "results/atlas_nn_stage_b_delta_rank_capacity_sweep.json",
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
