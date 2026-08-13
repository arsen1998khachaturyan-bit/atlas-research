"""Experiment 25: a genuinely new angle on the MLP input-layer mystery,
per docs/NEXT_RESEARCH_DECISION.md's own note that resolving it "would
need a different kind of tool ... not another ablation of the same
shape." Experiments 7, 12, 15, 16 all examined properties of the
*trained* weight matrix (its effective rank, or how the task's input
noise/preprocessing was structured). None asked the more basic question:
how much does each layer's weight matrix actually MOVE during training,
and is that movement itself structured (low-rank) or diffuse?

Hypothesis: the input layer's flat-to-negative post-training
compressibility (Experiments 4, 6, 7, 12, 15, 16) might simply reflect
that it changes *less* during training, relative to its own random-init
scale, than deeper layers do -- staying closer to its starting point,
with less new training-induced structure for a compression method to
exploit in the first place. This also asks whether the *change itself*
(the weight delta) is low-rank, a distinct question from Experiment 7's
"is the final trained matrix low-rank."

Uses the exact original Stage B setup (Experiments 3/4/6/7's 3-Linear
2-XOR MLP) so this connects directly to the layer-by-layer compression-
gain pattern already established there, at the project's 8-seed standard.
"""
from __future__ import annotations

import statistics as st
from collections import defaultdict

import numpy as np

from atlas_nn.stage_b.capacity_metrics import capacity_metrics
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
N_TRAIN = 2000
N_EVAL = 500
N_FEATURES = 32
EPOCHS = 500
LR = 2e-2

# From Experiments 4/6: layer 0 (input) shows no/negative post-training
# compression gain; layers 2 and 4 (hidden, output) show a clear positive
# gain. Recorded here as a fixed reference so this experiment's own
# weight-movement numbers can be checked against the already-established
# compression pattern without rerunning any compression search.
KNOWN_COMPRESSION_GAIN_DIRECTION = {
    "0": "flat_to_negative",
    "2": "positive",
    "4": "positive",
}


def frobenius_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix))


def read_weight_at(model, state: dict, layer_name: str) -> np.ndarray:
    load_snapshot(model, state)
    return get_weight(model, layer_name).detach().numpy().copy()


def main() -> None:
    rows = []

    for seed in SEEDS:
        x_train, y_train = make_xor_dataset(N_TRAIN, n_features=N_FEATURES, seed=seed)
        x_eval, y_eval = make_xor_dataset(N_EVAL, n_features=N_FEATURES, seed=seed + 10_000)

        model = build_mlp(seed=seed, input_dim=N_FEATURES)
        random_state = snapshot(model)
        random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        history = train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
        trained_state = snapshot(model)
        trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        print(
            f"seed={seed} random_acc={random_acc:.3f} "
            f"train_acc={history['final_train_accuracy']:.3f} eval_acc={trained_acc:.3f}",
            flush=True,
        )

        for layer_name in linear_layer_names(model):
            w_random = read_weight_at(model, random_state, layer_name)
            w_trained = read_weight_at(model, trained_state, layer_name)

            delta = w_trained - w_random
            random_norm = frobenius_norm(w_random)
            trained_norm = frobenius_norm(w_trained)
            delta_norm = frobenius_norm(delta)
            relative_change = delta_norm / random_norm if random_norm > 0 else float("inf")

            cm_delta = capacity_metrics(delta)
            cm_random = capacity_metrics(w_random)
            cm_trained = capacity_metrics(w_trained)
            delta_rank_fraction = (
                cm_delta["shannon_effective_rank"] / cm_delta["max_rank"] if cm_delta["max_rank"] > 0 else None
            )

            row = {
                "seed": seed,
                "layer": layer_name,
                "layer_shape": list(w_random.shape),
                "known_compression_gain_direction": KNOWN_COMPRESSION_GAIN_DIRECTION.get(layer_name, "unknown"),
                "random_norm": random_norm,
                "trained_norm": trained_norm,
                "delta_norm": delta_norm,
                "relative_weight_change": relative_change,
                "delta_effective_rank": cm_delta["shannon_effective_rank"],
                "delta_max_rank": cm_delta["max_rank"],
                "delta_rank_fraction": delta_rank_fraction,
                "random_effective_rank": cm_random["shannon_effective_rank"],
                "trained_effective_rank": cm_trained["shannon_effective_rank"],
            }
            rows.append(row)
            print(
                f"  layer={layer_name} rel_change={relative_change:.4f} "
                f"delta_erank={cm_delta['shannon_effective_rank']:.2f} "
                f"delta_rank_frac={delta_rank_fraction:.3f} "
                f"(known: {row['known_compression_gain_direction']})",
                flush=True,
            )

    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r)

    summary = {}
    print("\n=== summary by layer (mean over 8 seeds) ===", flush=True)
    for layer_name, layer_rows in sorted(by_layer.items(), key=lambda kv: int(kv[0])):
        mean_rel_change = st.mean(r["relative_weight_change"] for r in layer_rows)
        mean_delta_rank_frac = st.mean(r["delta_rank_fraction"] for r in layer_rows)
        summary[layer_name] = {
            "known_compression_gain_direction": layer_rows[0]["known_compression_gain_direction"],
            "mean_relative_weight_change": mean_rel_change,
            "mean_delta_rank_fraction": mean_delta_rank_frac,
        }
        print(
            f"layer={layer_name:<3} known_gain={layer_rows[0]['known_compression_gain_direction']:<18} "
            f"mean_rel_change={mean_rel_change:.4f} mean_delta_rank_frac={mean_delta_rank_frac:.4f}",
            flush=True,
        )

    save_json(
        {
            "experiment": "atlas_nn-stage_b_weight_delta_analysis",
            "seeds": list(SEEDS),
            "n_train": N_TRAIN,
            "n_features": N_FEATURES,
            "rows": rows,
            "summary": summary,
        },
        "results/atlas_nn_stage_b_weight_delta_analysis.json",
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
