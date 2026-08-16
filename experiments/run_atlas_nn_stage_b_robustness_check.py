"""Robustness check for Experiment 4 (see docs/NEXT_RESEARCH_DECISION.md):
the finding "post-training compression headroom rises ~1.5-2x on deeper
layers but not on the input layer" rests on a single 3-Linear-layer MLP
trained on one easy synthetic task (2-XOR). Before treating that as a
property of training in general, this script repeats the same
behavior-budgeted search under two different conditions:

  (a) harder_task -- the *same* 3-Linear-layer architecture, trained on a
      harder synthetic task (3-way parity) instead of 2-XOR.
  (b) deeper_net -- a *deeper* 6-Linear-layer architecture (5 hidden
      layers), trained on the original 2-XOR task.

If the "deeper layers gain more, input layer gains nothing" pattern
reproduces in both conditions, that's real evidence it's a property of
training depth-dependence rather than of one specific setup. If it doesn't,
that's equally informative -- see docs/NEXT_RESEARCH_DECISION.md.
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.dataset import make_parity_dataset, make_xor_dataset
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33)
QUALITY_THRESHOLD = 0.05


def run_condition(
    *,
    condition_name: str,
    make_dataset,
    dataset_kwargs: dict,
    n_train: int,
    n_eval: int,
    model_kwargs: dict,
    train_kwargs: dict,
) -> list[dict]:
    searches = []

    for seed in SEEDS:
        x_train, y_train = make_dataset(n_train, seed=seed, **dataset_kwargs)
        x_eval, y_eval = make_dataset(n_eval, seed=seed + 50_000, **dataset_kwargs)

        model = build_mlp(seed, **model_kwargs)
        random_init_state = snapshot(model)
        history = train_mlp(model, x_train, y_train, **train_kwargs)
        trained_state = snapshot(model)
        held_out_accuracy = evaluate(model, x_eval, y_eval)["accuracy"]

        print(
            f"[{condition_name}] seed={seed} train_acc={history['final_train_accuracy']:.3f} "
            f"held_out_acc={held_out_accuracy:.3f}"
        )

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
                    get_weight=get_weight,
                    set_weight=set_weight,
                    evaluate=evaluate,
                    snapshot=snapshot,
                    load_snapshot=load_snapshot,
                    quality_threshold=QUALITY_THRESHOLD,
                )
                search["condition"] = condition_name
                search["held_out_accuracy"] = held_out_accuracy
                searches.append(search)

                best = search["overall_best_ratio"]
                best_str = f"{best:.2f}" if best is not None else "NONE"
                print(
                    f"  [{condition_name}] seed={seed:>3} state={state_label:<11} "
                    f"layer={layer_name:<3} shape={str(layer_shape):<10} best_ratio={best_str}"
                )

    return searches


def main() -> None:
    all_searches = []

    all_searches += run_condition(
        condition_name="harder_task_parity3",
        make_dataset=make_parity_dataset,
        dataset_kwargs={"n_features": 8, "k": 3},
        n_train=4000,
        n_eval=500,
        model_kwargs={"input_dim": 8, "hidden_dim": 64, "n_hidden_layers": 2},
        train_kwargs={"epochs": 800, "lr": 1e-2},
    )

    all_searches += run_condition(
        condition_name="deeper_net_xor2",
        make_dataset=make_xor_dataset,
        dataset_kwargs={"n_features": 32},
        n_train=2000,
        n_eval=500,
        model_kwargs={"input_dim": 32, "hidden_dim": 64, "n_hidden_layers": 5},
        train_kwargs={"epochs": 500, "lr": 2e-2},
    )

    save_json(
        {
            "experiment": "atlas_nn-stage_b_robustness_check",
            "seeds": list(SEEDS),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "searches": all_searches,
        },
        "results/atlas_nn_stage_b_robustness_check.json",
    )


if __name__ == "__main__":
    main()
