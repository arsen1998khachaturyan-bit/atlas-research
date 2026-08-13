"""Experiment 28: does delta-rank fraction (Experiments 25-27's metric --
effective rank of the training-induced weight change, as a fraction of
max rank) predict compression-gain magnitude on REAL pretrained models?

Two prior results point in different directions on what to expect here.
Experiment 26 found delta-rank fraction is the strongest mechanism
correlation in the project on the from-scratch MLP (r=-0.75). Experiment
27 (+ its 8-seed addendum) found it does NOT transfer to the from-scratch
Stage C-lite Transformer (r=-0.06 pooled, no clean per-block signal
either). Real pretrained models are architecturally Transformers (like
Stage C-lite) but trained very differently (real data, real optimizers,
far more steps) -- this is the first chance to learn whether Experiment
27's non-replication is about the *architecture* (Transformers in
general) or something specific to Stage C-lite's small-scale, short,
from-scratch training.

Cheap by design, like Experiment 24: reuses compression_gain already
computed by the Experiment 19/21/23 budget searches (no new compression
sweeps), and delta-rank fraction is just one more SVD (of
pretrained-minus-random, rather than of the raw matrices Experiment 24
used) on each already-tested layer of each already-downloaded model.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from atlas_nn.stage_b.capacity_metrics import capacity_metrics
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_c_real.model import get_weight, load_pretrained, load_random_init

MODELS = {
    "distilgpt2": "results/atlas_nn_stage_c_real_budget_search.json",
    "gpt2": "results/atlas_nn_stage_c_real_budget_search_gpt2.json",
    "gpt2-medium": "results/atlas_nn_stage_c_real_budget_search_gpt2_medium.json",
}
SEEDS = (11, 22, 33)


def load_compression_lookup(path: str) -> tuple[dict, list[str]]:
    data = json.loads(Path(path).read_text())
    lookup = {
        (s["layer"], s["model_state"], s["seed"]): s["overall_best_ratio"]
        for s in data["searches"]
    }
    layers = list(dict.fromkeys(s["layer"] for s in data["searches"]))
    return lookup, layers


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def block_of(layer_name: str) -> int:
    return int(layer_name.split(".")[2])


def sublayer_type_of(layer_name: str) -> str:
    return ".".join(layer_name.split(".")[3:])


def main() -> None:
    rows = []

    for model_name, results_path in MODELS.items():
        print(f"=== {model_name} ===", flush=True)
        compression_lookup, layers = load_compression_lookup(results_path)

        pretrained_model = load_pretrained(model_name)
        weights_pretrained = {layer: get_weight(pretrained_model, layer).detach().numpy().copy() for layer in layers}
        del pretrained_model

        for seed in SEEDS:
            random_model = load_random_init(seed, model_name)
            for layer_name in layers:
                w_random = get_weight(random_model, layer_name).detach().numpy().copy()
                w_pretrained = weights_pretrained[layer_name]

                delta = w_pretrained - w_random
                cm_delta = capacity_metrics(delta)
                delta_rank_fraction = (
                    cm_delta["shannon_effective_rank"] / cm_delta["max_rank"] if cm_delta["max_rank"] > 0 else None
                )
                relative_change = (
                    float(np.linalg.norm(delta) / np.linalg.norm(w_random)) if np.linalg.norm(w_random) > 0 else None
                )

                ratio_random = compression_lookup.get((layer_name, "random_init", seed))
                ratio_pretrained = compression_lookup.get((layer_name, "pretrained", 0))
                compression_gain = ratio_pretrained / ratio_random if (ratio_random and ratio_pretrained) else None

                row = {
                    "model": model_name,
                    "layer": layer_name,
                    "block": block_of(layer_name),
                    "sublayer_type": sublayer_type_of(layer_name),
                    "seed": seed,
                    "delta_rank_fraction": delta_rank_fraction,
                    "relative_weight_change": relative_change,
                    "compression_ratio_random": ratio_random,
                    "compression_ratio_pretrained": ratio_pretrained,
                    "compression_gain": compression_gain,
                }
                rows.append(row)

                drf_str = f"{delta_rank_fraction:.3f}" if delta_rank_fraction is not None else "None"
                cg_str = f"{compression_gain:.3f}" if compression_gain else "None"
                print(
                    f"  seed={seed:>3} layer={layer_name:<32} delta_rank_frac={drf_str:<8} "
                    f"rel_change={relative_change:.3f} compression_gain={cg_str}",
                    flush=True,
                )
            del random_model

    valid = [r for r in rows if r["delta_rank_fraction"] is not None and r["compression_gain"] and r["compression_gain"] > 0]
    x_all = np.array([r["delta_rank_fraction"] for r in valid])
    y_all = np.log(np.array([r["compression_gain"] for r in valid]))
    overall_pearson = pearson(x_all, y_all)
    overall_spearman = pearson(rank_transform(x_all), rank_transform(y_all))
    print(f"\nn={len(valid)} valid points", flush=True)
    print(f"overall (pooled across all 3 models): pearson={overall_pearson} spearman={overall_spearman}", flush=True)

    summary = {"overall": {"n": len(valid), "pearson": overall_pearson, "spearman": overall_spearman}}

    for model_name in MODELS:
        model_valid = [r for r in valid if r["model"] == model_name]
        if len(model_valid) < 3:
            continue
        xs = np.array([r["delta_rank_fraction"] for r in model_valid])
        ys = np.log(np.array([r["compression_gain"] for r in model_valid]))
        summary[model_name] = {
            "n": len(model_valid),
            "pearson": pearson(xs, ys),
            "spearman": pearson(rank_transform(xs), rank_transform(ys)),
            "mean_delta_rank_fraction": float(np.mean(xs)),
        }
        print(f"{model_name}: n={len(model_valid)} pearson={summary[model_name]['pearson']} "
              f"spearman={summary[model_name]['spearman']} "
              f"mean_delta_rank_fraction={summary[model_name]['mean_delta_rank_fraction']:.3f}", flush=True)

    by_block = defaultdict(list)
    for r in valid:
        by_block[(r["model"], r["block"])].append(r)
    per_block = {}
    for (model_name, block), block_rows in by_block.items():
        if len(block_rows) < 3:
            continue
        xs = np.array([r["delta_rank_fraction"] for r in block_rows])
        ys = np.log(np.array([r["compression_gain"] for r in block_rows]))
        key = f"{model_name}_block{block}"
        per_block[key] = {
            "n": len(block_rows),
            "pearson": pearson(xs, ys),
            "mean_delta_rank_fraction": float(np.mean(xs)),
            "mean_compression_gain": float(np.mean([r["compression_gain"] for r in block_rows])),
        }
        print(f"{key}: n={len(block_rows)} pearson={per_block[key]['pearson']} "
              f"mean_delta_rank_fraction={per_block[key]['mean_delta_rank_fraction']:.3f} "
              f"mean_compression_gain={per_block[key]['mean_compression_gain']:.3f}", flush=True)

    save_json(
        {
            "experiment": "atlas_nn-stage_c_real_delta_rank",
            "models": list(MODELS.keys()),
            "seeds": list(SEEDS),
            "rows": rows,
            "summary": summary,
            "per_block": per_block,
        },
        "results/atlas_nn_stage_c_real_delta_rank.json",
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
