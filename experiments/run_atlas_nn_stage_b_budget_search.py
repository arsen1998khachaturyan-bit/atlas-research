"""Follow-up to Stage B (see docs/NEXT_RESEARCH_DECISION.md): Stage B found
that trained-network *behavior* is far more robust to weight-compression
error than the raw tensor error predicts. This script turns that into an
actionable question: instead of a fixed bit-budget, how far can each
method's compression ratio actually go before behavioral quality
(relative output-logit error) crosses a threshold -- and is that
achievable ratio higher for trained weights than for the same layer at
random initialization?

For each layer, in both the random-init and trained state, every method
family (quantization, SVD, pruning, vector-codebook, Atlas block-dictionary)
is swept over its own parameter grid, and the best compression ratio that
still meets the quality bar is kept.
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names
from atlas_nn.stage_b.train import load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33)
N_TRAIN = 2000
N_EVAL = 500
N_FEATURES = 32
QUALITY_THRESHOLD = 0.05


def main() -> None:
    all_searches = []

    for seed in SEEDS:
        x_train, y_train = make_xor_dataset(N_TRAIN, n_features=N_FEATURES, seed=seed)
        x_eval, y_eval = make_xor_dataset(N_EVAL, n_features=N_FEATURES, seed=seed + 50_000)

        model = build_mlp(seed, input_dim=N_FEATURES)
        random_init_state = snapshot(model)
        train_mlp(model, x_train, y_train, epochs=500, lr=2e-2)
        trained_state = snapshot(model)

        layers = linear_layer_names(model)

        for state_label, state in (("random_init", random_init_state), ("trained", trained_state)):
            load_snapshot(model, state)
            for layer_name in layers:
                layer_shape = tuple(get_weight(model, layer_name).shape)
                search = run_budget_search(
                    model=model,
                    layer_name=layer_name,
                    layer_shape=layer_shape,
                    model_state_label=state_label,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    seed=seed,
                    quality_threshold=QUALITY_THRESHOLD,
                )
                all_searches.append(search)

                best = search["overall_best_ratio"]
                best_family = search["overall_best_family"]
                best_ratio_str = f"{best:.2f}" if best is not None else "NONE (no config met threshold)"
                print(
                    f"seed={seed:>3} state={state_label:<11} layer={layer_name:<3} "
                    f"shape={str(layer_shape):<10} best_family={str(best_family):<18} "
                    f"best_ratio={best_ratio_str}"
                )

    save_json(
        {
            "experiment": "atlas_nn-stage_b_budget_search",
            "seeds": list(SEEDS),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "searches": all_searches,
        },
        "results/atlas_nn_stage_b_budget_search.json",
    )


if __name__ == "__main__":
    main()
