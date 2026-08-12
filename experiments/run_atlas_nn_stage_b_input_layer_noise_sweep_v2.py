"""Controlled re-run of the input-layer noise-dimension sweep
(experiments/run_atlas_nn_stage_b_input_layer_noise_sweep.py, Experiment
12), fixing the confound that experiment flagged but didn't resolve:
training data size was held fixed while `n_features` varied, so
higher-noise conditions were simultaneously harder generalization
problems with proportionally less data (held-out accuracy ranged
73.6%-98.4% across the sweep) -- noise fraction and task difficulty were
never cleanly separated.

This version raises N_TRAIN to 8000 (vs. 2000), which brings held-out
accuracy to a much tighter 91.0%-99.6% band across the same n_features
range (verified empirically before committing to the full run), and adds
n_features=2 (zero noise dimensions) as a proper member of the main sweep
rather than an ad hoc addendum. If the input layer's flat-to-negative
compressibility (Experiments 4, 6, 7, 12) was actually being masked by
the task-difficulty confound, a real noise-fraction trend should now be
visible where Experiment 12 saw none. If it still isn't, that's a
cleaner, stronger negative result on the noise-fraction hypothesis than
Experiment 12 alone could provide.
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
N_TRAIN = 8000
N_EVAL = 500
HIDDEN_DIM = 64
N_HIDDEN_LAYERS = 2
EPOCHS = 500
LR = 2e-2
QUALITY_THRESHOLD = 0.05
N_FEATURES_SWEEP = (2, 4, 8, 16, 32, 64)


def main() -> None:
    all_searches = []
    training_summaries = []

    for n_features in N_FEATURES_SWEEP:
        for seed in SEEDS:
            x_train, y_train = make_xor_dataset(N_TRAIN, n_features=n_features, seed=seed)
            x_eval, y_eval = make_xor_dataset(N_EVAL, n_features=n_features, seed=seed + 50_000)

            model = build_mlp(seed, input_dim=n_features, hidden_dim=HIDDEN_DIM, n_hidden_layers=N_HIDDEN_LAYERS)
            random_init_state = snapshot(model)
            random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

            train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
            trained_state = snapshot(model)
            trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

            noise_fraction = (n_features - 2) / n_features
            training_summaries.append({
                "n_features": n_features, "seed": seed, "noise_fraction": noise_fraction,
                "random_accuracy": random_acc, "trained_accuracy": trained_acc,
            })
            print(
                f"n_features={n_features:>3} seed={seed:>3} noise_frac={noise_fraction:.3f} "
                f"random_acc={random_acc:.3f} trained_acc={trained_acc:.3f}"
            )

            layers = linear_layer_names(model)
            input_layer = layers[0]  # first Linear layer, the one that sees raw input

            for state_label, state in (("random_init", random_init_state), ("trained", trained_state)):
                load_snapshot(model, state)
                layer_shape = tuple(get_weight(model, input_layer).shape)
                search = run_budget_search(
                    model=model,
                    layer_name=input_layer,
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
                search["n_features"] = n_features
                search["noise_fraction"] = noise_fraction
                all_searches.append(search)

                best = search["overall_best_ratio"]
                best_str = f"{best:.2f}" if best is not None else "NONE"
                print(
                    f"  n_features={n_features:>3} seed={seed:>3} state={state_label:<11} "
                    f"input_layer_shape={layer_shape} best_ratio={best_str}"
                )

    save_json(
        {
            "experiment": "atlas_nn-stage_b_input_layer_noise_sweep_v2",
            "seeds": list(SEEDS),
            "n_train": N_TRAIN,
            "n_features_sweep": list(N_FEATURES_SWEEP),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "training_summaries": training_summaries,
            "searches": all_searches,
        },
        "results/atlas_nn_stage_b_input_layer_noise_sweep_v2.json",
    )


if __name__ == "__main__":
    main()
