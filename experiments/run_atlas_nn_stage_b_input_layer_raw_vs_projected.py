"""Tests a new hypothesis for the MLP input layer's still-unexplained
behavior (open since Experiment 4; noise-fraction ruled out at good power
in Experiments 12/15, effective rank only weakly related in Experiment 7):
is it about literally being the first layer to see *raw, untransformed*
task input, rather than a capacity/slack effect the way every other layer
shows?

Every OTHER layer in the network (hidden, output) receives an already-
processed representation from an upstream layer, not raw task features.
This tests whether that difference matters: `atlas_nn.stage_b.model.
build_mlp(use_frozen_input_projection=True)` prepends a frozen (never
trained), orthogonally-initialized linear projection before the trainable
stack, so the first *trainable* layer sees a fixed linear transform of the
raw input instead of the raw input directly -- same task, same shapes,
only the "rawness" of what the first trainable layer sees changes.

If the first trainable layer's compressibility gain becomes more
hidden-layer-like (a real, positive gain) once it no longer sees raw
input, that supports the raw-input hypothesis. If it stays flat/negative
like every previous measurement of the true input layer, that rules this
out too, narrowing the search further.
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
N_TRAIN = 2000
N_EVAL = 500
N_FEATURES = 32
HIDDEN_DIM = 64
EPOCHS = 500
LR = 2e-2
QUALITY_THRESHOLD = 0.05

CONDITIONS = [("raw_input", False), ("projected_input", True)]


def main() -> None:
    all_searches = []
    training_summaries = []

    for condition_name, use_projection in CONDITIONS:
        for seed in SEEDS:
            x_train, y_train = make_xor_dataset(N_TRAIN, n_features=N_FEATURES, seed=seed)
            x_eval, y_eval = make_xor_dataset(N_EVAL, n_features=N_FEATURES, seed=seed + 50_000)

            model = build_mlp(
                seed, input_dim=N_FEATURES, hidden_dim=HIDDEN_DIM,
                n_hidden_layers=2, use_frozen_input_projection=use_projection,
            )
            random_init_state = snapshot(model)
            random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

            train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
            trained_state = snapshot(model)
            trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

            training_summaries.append({
                "condition": condition_name, "seed": seed,
                "random_accuracy": random_acc, "trained_accuracy": trained_acc,
            })
            print(
                f"[{condition_name}] seed={seed} random_acc={random_acc:.3f} trained_acc={trained_acc:.3f}"
            )

            layers = linear_layer_names(model)
            # first TRAINABLE layer: index 0 when there's no frozen
            # projection, index 1 (skipping the frozen projection) when
            # there is one.
            target_layer = layers[1] if use_projection else layers[0]

            for state_label, state in (("random_init", random_init_state), ("trained", trained_state)):
                load_snapshot(model, state)
                layer_shape = tuple(get_weight(model, target_layer).shape)
                search = run_budget_search(
                    model=model,
                    layer_name=target_layer,
                    layer_shape=layer_shape,
                    model_state_label=state_label,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    seed=seed,
                    get_weight=get_weight,
                    set_weight=set_weight,
                    evaluate=evaluate,
                    snapshot=snapshot,
                    load_snapshot=load_snapshot,
                    quality_threshold=QUALITY_THRESHOLD,
                )
                search["condition"] = condition_name
                search["target_layer"] = target_layer
                all_searches.append(search)

                best = search["overall_best_ratio"]
                best_str = f"{best:.2f}" if best is not None else "NONE"
                print(
                    f"  [{condition_name}] seed={seed:>3} state={state_label:<11} "
                    f"target_layer={target_layer} shape={layer_shape} best_ratio={best_str}"
                )

    save_json(
        {
            "experiment": "atlas_nn-stage_b_input_layer_raw_vs_projected",
            "seeds": list(SEEDS),
            "conditions": [c[0] for c in CONDITIONS],
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "training_summaries": training_summaries,
            "searches": all_searches,
        },
        "results/atlas_nn_stage_b_input_layer_raw_vs_projected.json",
    )


if __name__ == "__main__":
    main()
