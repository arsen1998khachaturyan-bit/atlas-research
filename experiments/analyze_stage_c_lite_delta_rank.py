"""Experiment 27: does delta-rank fraction (Experiments 25-26's new MLP
mechanism metric -- the effective rank of `W_trained - W_random`, as a
fraction of max rank) transfer to the Stage C-lite Transformer, the way
Experiments 8-9 showed the underlying capacity-sweep/behavioral-
robustness finding transfers? Never checked on this architecture before
-- Experiment 10 checked the *final*-matrix effective rank on this same
Transformer and found it did NOT transfer as cleanly as it did for the
MLP (r=-0.25 to -0.54 initially, later corrected to a real but weaker
positive r=0.23-0.55 after Experiment 11's larger seed count). This asks
the same question of the *update's* rank instead of the final matrix's.

Reproduces Experiment 9's exact training setup at 8 seeds (extended from
the original 3, mirroring how Experiment 11's addendum resolved a
similarly weak/ambiguous correlation on this same Transformer by raising
seed count), correlating against compression_gain from a matching 8-seed
rerun of run_atlas_nn_stage_c_lite_budget_search.py -- no new compression
methodology, just more seeds on both sides of the lookup.
"""
from __future__ import annotations

import json
from collections import defaultdict
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
from atlas_nn.stage_c_lite.model import build_transformer_classifier, get_weight, linear_layer_names

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
N_EXAMPLES = 200
N_TRAIN = 150
MAX_LEN = 16
EPOCHS = 150
LR = 3e-3


def load_compression_lookup() -> dict:
    path = Path("results/atlas_nn_stage_c_lite_budget_search.json")
    data = json.loads(path.read_text())
    return {
        (s["layer"], s["model_state"], s["seed"]): s["overall_best_ratio"]
        for s in data["searches"]
    }


def block_of(layer_name: str) -> str:
    if layer_name == "classifier":
        return "head"
    return "block0" if "layers.0" in layer_name else "block1"


def sublayer_type_of(layer_name: str) -> str:
    if layer_name == "classifier":
        return "classifier"
    return layer_name.split(".")[-1]


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


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

        print(f"seed={seed} random_acc={random_acc:.3f} trained_acc={trained_acc:.3f}", flush=True)

        for layer_name in linear_layer_names(model):
            load_snapshot(model, random_state)
            w_random = get_weight(model, layer_name).detach().numpy().copy()
            load_snapshot(model, trained_state)
            w_trained = get_weight(model, layer_name).detach().numpy().copy()

            delta = w_trained - w_random
            cm_delta = capacity_metrics(delta)
            delta_rank_fraction = (
                cm_delta["shannon_effective_rank"] / cm_delta["max_rank"] if cm_delta["max_rank"] > 0 else None
            )
            relative_change = (
                float(np.linalg.norm(delta) / np.linalg.norm(w_random)) if np.linalg.norm(w_random) > 0 else None
            )

            ratio_random = compression_lookup.get((layer_name, "random_init", seed))
            ratio_trained = compression_lookup.get((layer_name, "trained", seed))
            compression_gain = ratio_trained / ratio_random if (ratio_random and ratio_trained) else None

            row = {
                "seed": seed,
                "layer": layer_name,
                "block": block_of(layer_name),
                "sublayer_type": sublayer_type_of(layer_name),
                "delta_rank_fraction": delta_rank_fraction,
                "relative_weight_change": relative_change,
                "compression_ratio_random": ratio_random,
                "compression_ratio_trained": ratio_trained,
                "compression_gain": compression_gain,
            }
            rows.append(row)

            drf_str = f"{delta_rank_fraction:.3f}" if delta_rank_fraction is not None else "None"
            cg_str = f"{compression_gain:.3f}" if compression_gain else "None"
            print(
                f"  layer={layer_name:<32} delta_rank_frac={drf_str:<8} "
                f"rel_change={relative_change:.3f} compression_gain={cg_str}",
                flush=True,
            )

    valid = [r for r in rows if r["delta_rank_fraction"] is not None and r["compression_gain"] and r["compression_gain"] > 0]

    x_all = np.array([r["delta_rank_fraction"] for r in valid])
    y_all = np.log(np.array([r["compression_gain"] for r in valid]))
    overall_pearson = pearson(x_all, y_all)
    overall_spearman = pearson(rank_transform(x_all), rank_transform(y_all))
    print(f"\nn={len(valid)} valid points", flush=True)
    print(f"overall: pearson={overall_pearson} spearman={overall_spearman}", flush=True)

    by_block = defaultdict(list)
    for r in valid:
        by_block[r["block"]].append(r)

    per_block = {}
    for block_name, block_rows in by_block.items():
        xs = np.array([r["delta_rank_fraction"] for r in block_rows])
        ys = np.log(np.array([r["compression_gain"] for r in block_rows]))
        per_block[block_name] = {
            "n": len(block_rows),
            "pearson": pearson(xs, ys),
            "mean_delta_rank_fraction": float(np.mean(xs)),
            "mean_compression_gain": float(np.mean([r["compression_gain"] for r in block_rows])),
        }
        print(f"block={block_name} n={len(block_rows)} pearson={per_block[block_name]['pearson']} "
              f"mean_delta_rank_fraction={per_block[block_name]['mean_delta_rank_fraction']:.3f} "
              f"mean_compression_gain={per_block[block_name]['mean_compression_gain']:.3f}", flush=True)

    by_sublayer = defaultdict(list)
    for r in valid:
        by_sublayer[r["sublayer_type"]].append(r)
    per_sublayer = {}
    for sub_name, sub_rows in by_sublayer.items():
        xs = np.array([r["delta_rank_fraction"] for r in sub_rows])
        per_sublayer[sub_name] = {
            "n": len(sub_rows),
            "mean_delta_rank_fraction": float(np.mean(xs)),
            "mean_compression_gain": float(np.mean([r["compression_gain"] for r in sub_rows])),
        }
        print(f"sublayer={sub_name} n={len(sub_rows)} "
              f"mean_delta_rank_fraction={per_sublayer[sub_name]['mean_delta_rank_fraction']:.3f} "
              f"mean_compression_gain={per_sublayer[sub_name]['mean_compression_gain']:.3f}", flush=True)

    save_json(
        {
            "experiment": "atlas_nn-stage_c_lite_delta_rank",
            "seeds": list(SEEDS),
            "n_valid": len(valid),
            "overall_pearson": overall_pearson,
            "overall_spearman": overall_spearman,
            "per_block": per_block,
            "per_sublayer_type": per_sublayer,
            "rows": rows,
        },
        "results/atlas_nn_stage_c_lite_delta_rank.json",
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
