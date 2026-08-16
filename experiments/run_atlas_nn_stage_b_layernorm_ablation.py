"""Cross-architecture check for Experiment 11's LayerNorm finding
(docs/RESEARCH_LOG.md): on the Stage C-lite Transformer, removing
LayerNorm consistently RAISED the correlation between effective-rank
shrinkage and the post-training compression/behavioral-robustness gain
(0.23->0.55, 0.29->0.37, at 8 seeds) -- i.e. LayerNorm amplifies the gain
and decouples it from the layer's own rank structure.

This tests whether that's a property of LayerNorm specifically, or
something particular to attention/Transformer architectures: adds an
optional LayerNorm to the plain MLP (atlas_nn.stage_b.model.build_mlp,
use_layernorm=True) and repeats Experiment 7's methodology (budget-search
achievable ratio + effective-rank shrinkage, correlated) with LayerNorm on
vs. off, same task (2-XOR), same width, 8 seeds each for direct comparison
with Experiment 11's power.

If adding LayerNorm to the MLP raises its rank-shrinkage correlation
(mirroring, in reverse, what removing it did on the Transformer), that's
cross-architecture confirmation LayerNorm itself is the operative factor.
If it does nothing, LayerNorm's effect is likely Transformer/attention-
specific instead.
"""
from __future__ import annotations

import numpy as np

from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.stage_b.capacity_metrics import capacity_metrics
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import run_layer_experiment, save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
N_TRAIN = 2000
N_EVAL = 500
N_FEATURES = 32
HIDDEN_DIM = 64
EPOCHS = 500
LR = 2e-2
MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED = 0.15

CONDITIONS = [("layernorm_off", False), ("layernorm_on", True)]


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
    all_rows = []

    for condition_name, use_layernorm in CONDITIONS:
        for seed in SEEDS:
            x_train, y_train = make_xor_dataset(N_TRAIN, n_features=N_FEATURES, seed=seed)
            x_eval, y_eval = make_xor_dataset(N_EVAL, n_features=N_FEATURES, seed=seed + 50_000)

            model = build_mlp(
                seed, input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM,
                n_hidden_layers=2, use_layernorm=use_layernorm,
            )
            random_state = snapshot(model)
            random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

            train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
            trained_state = snapshot(model)
            trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]
            training_succeeded = (trained_acc - random_acc) >= MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED

            print(
                f"[{condition_name}] seed={seed} random_acc={random_acc:.3f} "
                f"trained_acc={trained_acc:.3f} ok={training_succeeded}"
            )

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

                load_snapshot(model, random_state)
                row_random = run_layer_experiment(
                    model=model, layer_name=layer_name, model_state_label="random_init",
                    method_name="svd_rank4", compress_fn=lambda m: svd_low_rank(m, rank=4),
                    x_eval=x_eval, y_eval=y_eval, seed=seed,
                    get_weight=get_weight, set_weight=set_weight, evaluate=evaluate,
                    snapshot=snapshot, load_snapshot=load_snapshot,
                    experiment_name="atlas_nn-stage_b_layernorm_ablation",
                )

                load_snapshot(model, trained_state)
                row_trained = run_layer_experiment(
                    model=model, layer_name=layer_name, model_state_label="trained",
                    method_name="svd_rank4", compress_fn=lambda m: svd_low_rank(m, rank=4),
                    x_eval=x_eval, y_eval=y_eval, seed=seed,
                    get_weight=get_weight, set_weight=set_weight, evaluate=evaluate,
                    snapshot=snapshot, load_snapshot=load_snapshot,
                    experiment_name="atlas_nn-stage_b_layernorm_ablation",
                )

                robustness_gain = None
                if row_trained["relative_logit_error"] > 0:
                    robustness_gain = row_random["relative_logit_error"] / row_trained["relative_logit_error"]

                all_rows.append({
                    "condition": condition_name,
                    "use_layernorm": use_layernorm,
                    "layer": layer_name,
                    "seed": seed,
                    "training_succeeded": training_succeeded,
                    "random_accuracy": random_acc,
                    "trained_accuracy": trained_acc,
                    "capacity_random": cm_random,
                    "capacity_trained": cm_trained,
                    "rank_shrinkage": rank_shrinkage,
                    "random_rel_logit_error": row_random["relative_logit_error"],
                    "trained_rel_logit_error": row_trained["relative_logit_error"],
                    "robustness_gain": robustness_gain,
                })
                print(
                    f"  layer={layer_name:<3} rank_shrinkage={rank_shrinkage} "
                    f"robustness_gain={robustness_gain}"
                )

    print("\n=== summary: mean robustness_gain and rank/gain correlation by condition ===")
    summary = {}
    for condition_name, _ in CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == condition_name and r["training_succeeded"]]
        gains = [r["robustness_gain"] for r in rows if r["robustness_gain"] and r["robustness_gain"] > 0]
        mean_gain = float(np.mean(gains)) if gains else None

        valid = [r for r in rows if r["robustness_gain"] and r["robustness_gain"] > 0 and r["rank_shrinkage"] is not None]
        x = np.array([r["rank_shrinkage"] for r in valid])
        y = np.log(np.array([r["robustness_gain"] for r in valid]))
        p = pearson(x, y)
        s = pearson(rank_transform(x), rank_transform(y)) if len(x) > 1 else None

        summary[condition_name] = {"n": len(valid), "mean_robustness_gain": mean_gain, "pearson": p, "spearman": s}
        print(f"{condition_name:<16} n={len(valid):>3} mean_gain={mean_gain} pearson={p} spearman={s}")

    save_json(
        {
            "experiment": "atlas_nn-stage_b_layernorm_ablation",
            "seeds": list(SEEDS),
            "conditions": [c[0] for c in CONDITIONS],
            "summary": summary,
            "rows": all_rows,
        },
        "results/atlas_nn_stage_b_layernorm_ablation.json",
    )


if __name__ == "__main__":
    main()
