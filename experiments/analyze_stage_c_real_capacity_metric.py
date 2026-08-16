"""Experiment 24: does effective rank predict the compression-gain
magnitude on REAL pretrained models, the way it did (r~0.67) for the
from-scratch MLP (Experiment 7) and, more weakly and LayerNorm-dependently
(r~0.23-0.55), for the from-scratch Stage C-lite Transformer (Experiments
10-11)? Never checked against an actual pretrained model before.

Cheap by design: reuses the compression_gain numbers already computed by
the Experiment 19/21/23 budget searches (no new compression sweeps), and
effective rank is just an SVD of each already-downloaded model's already-
tested layers -- no training, no multi-hour compute, safe to run without
the checkpointing/restart concerns of the budget searches themselves.
"""
from __future__ import annotations

import json
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


def block_of(layer_name: str, model_name: str) -> int:
    return int(layer_name.split(".")[2])


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

                cm_random = capacity_metrics(w_random)
                cm_pretrained = capacity_metrics(w_pretrained)

                rank_shrinkage = None
                if cm_random["shannon_effective_rank"] > 0:
                    rank_shrinkage = 1.0 - (
                        cm_pretrained["shannon_effective_rank"] / cm_random["shannon_effective_rank"]
                    )

                ratio_random = compression_lookup.get((layer_name, "random_init", seed))
                ratio_pretrained = compression_lookup.get((layer_name, "pretrained", 0))
                compression_gain = None
                if ratio_random and ratio_pretrained:
                    compression_gain = ratio_pretrained / ratio_random

                row = {
                    "model": model_name,
                    "layer": layer_name,
                    "block": block_of(layer_name, model_name),
                    "seed": seed,
                    "erank_random": cm_random["shannon_effective_rank"],
                    "erank_pretrained": cm_pretrained["shannon_effective_rank"],
                    "rank_shrinkage": rank_shrinkage,
                    "compression_ratio_random": ratio_random,
                    "compression_ratio_pretrained": ratio_pretrained,
                    "compression_gain": compression_gain,
                }
                rows.append(row)

                rs_str = f"{rank_shrinkage:.3f}" if rank_shrinkage is not None else "None"
                cg_str = f"{compression_gain:.3f}" if compression_gain is not None else "None"
                print(
                    f"  seed={seed:>3} layer={layer_name:<32} "
                    f"erank_random={cm_random['shannon_effective_rank']:8.2f} "
                    f"erank_pretrained={cm_pretrained['shannon_effective_rank']:8.2f} "
                    f"rank_shrinkage={rs_str:<8} compression_gain={cg_str}",
                    flush=True,
                )
            del random_model

    valid = [r for r in rows if r["rank_shrinkage"] is not None and r["compression_gain"] is not None]
    log_gain = np.array([np.log(r["compression_gain"]) for r in valid])
    shrink = np.array([r["rank_shrinkage"] for r in valid])

    summary = {"overall": {"n": len(valid), "pearson_r": pearson(shrink, log_gain)}}
    print(f"\noverall: n={len(valid)} pearson_r={summary['overall']['pearson_r']}", flush=True)

    for model_name in MODELS:
        model_valid = [r for r in valid if r["model"] == model_name]
        if len(model_valid) < 3:
            continue
        s = np.array([r["rank_shrinkage"] for r in model_valid])
        g = np.array([np.log(r["compression_gain"]) for r in model_valid])
        r_value = pearson(s, g)
        summary[model_name] = {"n": len(model_valid), "pearson_r": r_value}
        print(f"{model_name}: n={len(model_valid)} pearson_r={r_value}", flush=True)

    save_json(
        {
            "experiment": "atlas_nn-stage_c_real_capacity_metric",
            "models": list(MODELS.keys()),
            "seeds": list(SEEDS),
            "rows": rows,
            "summary": summary,
        },
        "results/atlas_nn_stage_c_real_capacity_metric.json",
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
