"""Direct test of the "slack/capacity" hypothesis raised in
docs/NEXT_RESEARCH_DECISION.md after Experiment 5: Experiment 4's
post-training compression-headroom gain (easy 2-XOR task) did not
reproduce on a harder task (3-way parity) with the same network width.
The working hypothesis was that the gain tracks *unused representational
capacity left after training*, not "training" as a general property --
an easy task leaves more of a fixed-width network unused than a hard task
does.

This script tests that directly: for both tasks (easy 2-XOR, hard
3-parity), sweep network width (hidden_dim = 16 / 64 / 256) while holding
depth fixed at 2 hidden layers (avoiding the depth-related random-init
degeneracy found in Experiment 5). If the slack hypothesis is right, the
post-training compression gain should track *available capacity relative
to task difficulty* -- e.g. shrink for 2-XOR at hidden_dim=16 (little spare
capacity even for the easy task) and reappear for 3-parity at
hidden_dim=256 (now oversized for the hard task).
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.dataset import make_parity_dataset, make_xor_dataset
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33)
QUALITY_THRESHOLD = 0.05

TASKS = {
    "xor2": dict(make_dataset=make_xor_dataset, dataset_kwargs={"n_features": 32},
                 n_train=2000, n_eval=500, epochs=500, lr=2e-2),
    "parity3": dict(make_dataset=make_parity_dataset, dataset_kwargs={"n_features": 8, "k": 3},
                     n_train=4000, n_eval=500, epochs=800, lr=1e-2),
}
WIDTHS = (16, 64, 256)


def run_condition(*, condition_name, task_cfg, hidden_dim) -> list[dict]:
    make_dataset = task_cfg["make_dataset"]
    dataset_kwargs = task_cfg["dataset_kwargs"]
    input_dim = dataset_kwargs["n_features"]
    searches = []

    for seed in SEEDS:
        x_train, y_train = make_dataset(task_cfg["n_train"], seed=seed, **dataset_kwargs)
        x_eval, y_eval = make_dataset(task_cfg["n_eval"], seed=seed + 50_000, **dataset_kwargs)

        model = build_mlp(seed, input_dim=input_dim, hidden_dim=hidden_dim, n_hidden_layers=2)
        random_init_state = snapshot(model)
        random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        history = train_mlp(model, x_train, y_train, epochs=task_cfg["epochs"], lr=task_cfg["lr"])
        trained_state = snapshot(model)
        trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        print(
            f"[{condition_name}] seed={seed} random_acc={random_acc:.3f} "
            f"trained_train_acc={history['final_train_accuracy']:.3f} trained_held_out_acc={trained_acc:.3f}"
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
                search["task"] = condition_name.split("_h")[0]
                search["hidden_dim"] = hidden_dim
                search["random_init_accuracy"] = random_acc
                search["trained_accuracy"] = trained_acc
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

    for task_name, task_cfg in TASKS.items():
        for hidden_dim in WIDTHS:
            condition_name = f"{task_name}_h{hidden_dim}"
            all_searches += run_condition(
                condition_name=condition_name, task_cfg=task_cfg, hidden_dim=hidden_dim
            )

    save_json(
        {
            "experiment": "atlas_nn-stage_b_capacity_sweep",
            "seeds": list(SEEDS),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "widths": list(WIDTHS),
            "tasks": list(TASKS),
            "searches": all_searches,
        },
        "results/atlas_nn_stage_b_capacity_sweep.json",
    )


if __name__ == "__main__":
    main()
