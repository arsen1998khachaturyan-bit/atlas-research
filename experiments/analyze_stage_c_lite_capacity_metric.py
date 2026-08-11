"""Experiment 7's analog for the Stage C-lite Transformer: does effective
rank (a capacity-usage proxy, computed independently of any compression
method) predict the magnitude of the compression-gain pattern found in
Experiment 9, the way it did (r~0.67) for the MLP's hidden layer?

Reproduces the same 3 seeds already characterized by
run_atlas_nn_stage_c_lite_budget_search.py (deterministic, same seeds),
extracts each Linear layer's weight tensor in both random-init and trained
states, computes effective rank, and correlates rank shrinkage against the
already-measured compression gain (log scale, Pearson + Spearman) -- both
overall and split by Transformer block, since Experiment 9 found the two
blocks behave differently (block 1 gains more than block 0).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_nn.stage_b.capacity_metrics import capacity_metrics
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
)

SEEDS = (11, 22, 33)
N_EXAMPLES = 200
N_TRAIN = 150
MAX_LEN = 16
EPOCHS = 150
LR = 3e-3
MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED = 0.15


def load_compression_lookup() -> dict:
    path = Path("results/atlas_nn_stage_c_lite_budget_search.json")
    data = json.loads(path.read_text())
    return {
        (s["layer"], s["model_state"], s["seed"]): s["overall_best_ratio"]
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


def block_of(layer_name: str) -> str:
    if layer_name == "classifier":
        return "head"
    return "block0" if "layers.0" in layer_name else "block1"


def main() -> None:
    compression_lookup = load_compression_lookup()
    rows = []

    for seed in SEEDS:
        all_examples = generate_sentiment_dataset(N_EXAMPLES, seed=seed)
        train_examples, eval_examples = all_examples[:N_TRAIN], all_examples[N_TRAIN:]
        vocab = build_vocab([text for text, _label in train_examples])

        x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=MAX_LEN)
        x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=MAX_LEN)

        model = build_transformer_classifier(seed=seed, vocab_size=len(vocab), max_len=MAX_LEN)
        random_state = snapshot(model)
        random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
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

            ratio_random = compression_lookup.get((layer_name, "random_init", seed))
            ratio_trained = compression_lookup.get((layer_name, "trained", seed))
            compression_gain = None
            if ratio_random and ratio_trained:
                compression_gain = ratio_trained / ratio_random

            row = {
                "layer": layer_name,
                "block": block_of(layer_name),
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
                f"seed={seed:>3} layer={layer_name:<38} trained_ok={training_succeeded!s:<5} "
                f"erank_random={cm_random['shannon_effective_rank']:7.2f} "
                f"erank_trained={cm_trained['shannon_effective_rank']:7.2f} "
                f"rank_shrinkage={rs_str:<8} compression_gain={cg_str}"
            )

    valid = [
        r for r in rows
        if r["training_succeeded"]
        and r["rank_shrinkage"] is not None
        and r["compression_gain"] is not None
        and r["compression_gain"] > 0
    ]

    def correlate(subset, label):
        x = np.array([r["rank_shrinkage"] for r in subset])
        y = np.log(np.array([r["compression_gain"] for r in subset]))
        p = pearson(x, y)
        s = pearson(rank_transform(x), rank_transform(y)) if len(x) > 1 else None
        print(f"{label}: n={len(subset)} pearson={p} spearman={s}")
        return {"n": len(subset), "pearson": p, "spearman": s}

    print()
    overall = correlate(valid, "overall")
    by_block = {
        block: correlate([r for r in valid if r["block"] == block], block)
        for block in ("block0", "block1", "head")
    }

    save_json(
        {
            "experiment": "atlas_nn-stage_c_lite_capacity_metric_analysis",
            "min_accuracy_gain_to_count_as_trained": MIN_ACCURACY_GAIN_TO_COUNT_AS_TRAINED,
            "n_total": len(rows),
            "n_valid": len(valid),
            "overall": overall,
            "by_block": by_block,
            "rows": rows,
        },
        "results/atlas_nn_stage_c_lite_capacity_metric.json",
    )


if __name__ == "__main__":
    main()
