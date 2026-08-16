"""Follow-up to Experiment 8 (docs/RESEARCH_LOG.md): the Stage C-lite smoke
test found that trained-Transformer behavior is far more robust to
weight-compression error than tensor error predicts, and that this effect
strengthens with depth across two Transformer blocks. This turns that into
the same kind of actionable, quantitative question Experiments 4/6 asked
of the MLP: instead of a fixed bit-budget, how far can each method's
compression ratio actually go before behavioral quality (relative
output-logit error) crosses a threshold -- and does the achievable ratio
increase with depth the way the raw robustness gain did?

Reuses atlas_nn.stage_b.budget_search.run_budget_search unchanged (it's
architecture-agnostic since the Experiment 8 refactor) on the Stage C-lite
Transformer's 7 Linear layers, in both random-init and trained states.
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from atlas_nn.stage_c_lite.dataset import (
    build_vocab,
    generate_sentiment_dataset,
    make_sentiment_arrays,
)
from atlas_nn.stage_c_lite.model import (
    build_transformer_classifier,
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
QUALITY_THRESHOLD = 0.05


def main() -> None:
    all_searches = []

    for seed in SEEDS:
        all_examples = generate_sentiment_dataset(N_EXAMPLES, seed=seed)
        train_examples, eval_examples = all_examples[:N_TRAIN], all_examples[N_TRAIN:]
        vocab = build_vocab([text for text, _label in train_examples])

        x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=MAX_LEN)
        x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=MAX_LEN)

        model = build_transformer_classifier(seed=seed, vocab_size=len(vocab), max_len=MAX_LEN)
        random_init_state = snapshot(model)
        random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
        trained_state = snapshot(model)
        trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        print(f"seed={seed} vocab={len(vocab)} random_acc={random_acc:.3f} trained_acc={trained_acc:.3f}")

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
                search["random_accuracy"] = random_acc
                search["trained_accuracy"] = trained_acc
                all_searches.append(search)

                best = search["overall_best_ratio"]
                best_family = search["overall_best_family"]
                best_ratio_str = f"{best:.2f}" if best is not None else "NONE (no config met threshold)"
                print(
                    f"seed={seed:>3} state={state_label:<11} layer={layer_name:<38} "
                    f"best_family={str(best_family):<18} best_ratio={best_ratio_str}"
                )

    save_json(
        {
            "experiment": "atlas_nn-stage_c_lite_budget_search",
            "seeds": list(SEEDS),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "searches": all_searches,
        },
        "results/atlas_nn_stage_c_lite_budget_search.json",
    )


if __name__ == "__main__":
    main()
