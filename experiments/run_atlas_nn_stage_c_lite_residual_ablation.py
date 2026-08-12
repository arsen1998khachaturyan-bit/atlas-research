"""Tests Experiment 10's hypothesis (docs/RESEARCH_LOG.md): does the
Transformer's behavioral-robustness gain (and its depth gradient) come
from residual connections / LayerNorm providing an error-absorbing
pathway, independent of any single layer's own weight-matrix rank
structure -- rather than from effective-rank reduction, which explained
the MLP's gain (Experiment 7) but not the Transformer's (Experiment 10)?

Trains the same architecture in 4 conditions (residual x layernorm, each
on/off) and for each: (a) measures the behavioral-robustness gain
(relative-logit-error ratio, random-init vs trained, at fixed svd_rank4 --
the method most sensitive to rank structure in every prior experiment) per
layer, and (b) correlates effective-rank shrinkage against that gain, the
same way Experiments 7 and 10 did.

If disabling residual/LayerNorm shrinks the robustness gain and/or
restores a positive rank-shrinkage correlation, that supports the
architectural-connectivity hypothesis. If the gain and the (lack of)
correlation persist regardless, the hypothesis is wrong and something else
explains the Transformer's gain.

Uses 8 seeds per condition (raised from the original 3 after Experiment 11
found a real but thinly-evidenced LayerNorm magnitude effect, n=2
comparison pairs) for better statistical power on that specific question.
"""
from __future__ import annotations

import numpy as np

from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.stage_b.capacity_metrics import capacity_metrics
from atlas_nn.stage_b.experiment import run_layer_experiment, save_json
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from atlas_nn.stage_c_lite.dataset import (
    build_vocab,
    generate_sentiment_dataset,
    make_sentiment_arrays,
)
from atlas_nn.stage_c_lite.model import (
    build_ablation_transformer_classifier,
    get_weight,
    linear_layer_names,
    set_weight,
)

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
N_EXAMPLES = 200
N_TRAIN = 150
MAX_LEN = 16
EPOCHS = 150
LR = 3e-3
MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED = 0.15

CONDITIONS = [
    ("residual_and_layernorm", True, True),
    ("no_residual", False, True),
    ("no_layernorm", True, False),
    ("no_residual_no_layernorm", False, False),
]


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

    for condition_name, use_residual, use_layernorm in CONDITIONS:
        for seed in SEEDS:
            all_examples = generate_sentiment_dataset(N_EXAMPLES, seed=seed)
            train_examples, eval_examples = all_examples[:N_TRAIN], all_examples[N_TRAIN:]
            vocab = build_vocab([text for text, _label in train_examples])

            x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=MAX_LEN)
            x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=MAX_LEN)

            model = build_ablation_transformer_classifier(
                seed=seed, vocab_size=len(vocab), max_len=MAX_LEN,
                use_residual=use_residual, use_layernorm=use_layernorm,
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
                    experiment_name="atlas_nn-stage_c_lite_residual_ablation",
                )

                load_snapshot(model, trained_state)
                row_trained = run_layer_experiment(
                    model=model, layer_name=layer_name, model_state_label="trained",
                    method_name="svd_rank4", compress_fn=lambda m: svd_low_rank(m, rank=4),
                    x_eval=x_eval, y_eval=y_eval, seed=seed,
                    get_weight=get_weight, set_weight=set_weight, evaluate=evaluate,
                    snapshot=snapshot, load_snapshot=load_snapshot,
                    experiment_name="atlas_nn-stage_c_lite_residual_ablation",
                )

                robustness_gain = None
                if row_trained["relative_logit_error"] > 0:
                    robustness_gain = row_random["relative_logit_error"] / row_trained["relative_logit_error"]

                all_rows.append({
                    "condition": condition_name,
                    "use_residual": use_residual,
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
                    f"  layer={layer_name:<38} rank_shrinkage={rank_shrinkage} "
                    f"robustness_gain={robustness_gain}"
                )

    print("\n=== summary: mean robustness_gain and rank/gain correlation by condition ===")
    summary = {}
    for condition_name, _, _ in CONDITIONS:
        rows = [r for r in all_rows if r["condition"] == condition_name and r["training_succeeded"]]
        gains = [r["robustness_gain"] for r in rows if r["robustness_gain"] and r["robustness_gain"] > 0]
        mean_gain = float(np.mean(gains)) if gains else None

        valid = [r for r in rows if r["robustness_gain"] and r["robustness_gain"] > 0 and r["rank_shrinkage"] is not None]
        x = np.array([r["rank_shrinkage"] for r in valid])
        y = np.log(np.array([r["robustness_gain"] for r in valid]))
        p = pearson(x, y)
        s = pearson(rank_transform(x), rank_transform(y)) if len(x) > 1 else None

        summary[condition_name] = {"n": len(valid), "mean_robustness_gain": mean_gain, "pearson": p, "spearman": s}
        print(f"{condition_name:<28} n={len(valid):>3} mean_gain={mean_gain} pearson={p} spearman={s}")

    save_json(
        {
            "experiment": "atlas_nn-stage_c_lite_residual_ablation",
            "seeds": list(SEEDS),
            "conditions": [c[0] for c in CONDITIONS],
            "summary": summary,
            "rows": all_rows,
        },
        "results/atlas_nn_stage_c_lite_residual_ablation.json",
    )


if __name__ == "__main__":
    main()
