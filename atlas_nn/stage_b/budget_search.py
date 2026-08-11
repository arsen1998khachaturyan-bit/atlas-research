from __future__ import annotations

from typing import Callable

import numpy as np

from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize
from atlas_nn.stage_b.experiment import run_layer_experiment
from atlas_nn.structural import block_dictionary_transform

Config = tuple[str, object, Callable[[np.ndarray], object]]


def quantize_configs() -> list[Config]:
    return [
        (f"quantize_{b}bit", b, (lambda m, b=b: uniform_quantize(m, bits=b, block_size=None)))
        for b in (1, 2, 3, 4, 6, 8)
    ]


def svd_configs(max_rank: int) -> list[Config]:
    ranks = sorted({r for r in (1, 2, 4, 8, 16, 32, 64) if r <= max_rank})
    return [(f"svd_rank{r}", r, (lambda m, r=r: svd_low_rank(m, rank=r))) for r in ranks]


def prune_configs() -> list[Config]:
    return [
        (f"prune_{int(s * 100)}pct", s, (lambda m, s=s: magnitude_prune(m, sparsity=s)))
        for s in (0.3, 0.5, 0.7, 0.9, 0.95)
    ]


def codebook_configs(seed: int) -> list[Config]:
    return [
        (f"vector_codebook_k{k}", k, (lambda m, k=k: vector_codebook(m, vector_len=8, k=k, seed=seed)))
        for k in (4, 8, 16, 32)
    ]


def atlas_configs(seed: int) -> list[Config]:
    configs = []
    for d in (4, 8, 16):
        for rb in (3, 4, 6):
            configs.append((
                f"atlas_dict{d}_res{rb}bit",
                (d, rb),
                (lambda m, d=d, rb=rb: block_dictionary_transform(
                    m, block_shape=(8, 8), dict_size=d, residual_bits=rb, seed=seed
                )),
            ))
    return configs


def all_families(seed: int, max_rank: int) -> dict[str, list[Config]]:
    return {
        "quantize": quantize_configs(),
        "svd": svd_configs(max_rank),
        "prune": prune_configs(),
        "vector_codebook": codebook_configs(seed),
        "atlas_block_dict": atlas_configs(seed),
    }


def sweep_family(
    *,
    model,
    layer_name: str,
    model_state_label: str,
    family_name: str,
    configs: list[Config],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    quality_metric: str = "relative_logit_error",
    quality_threshold: float = 0.05,
) -> dict:
    """Runs every config in a method family on one layer, then reports the
    highest compression ratio among configs whose behavioral error is within
    `quality_threshold` (mission section 8's storage-vs-quality tradeoff,
    applied per family rather than at one fixed bit-width)."""
    rows = []
    for method_name, param, fn in configs:
        row = run_layer_experiment(
            model=model,
            layer_name=layer_name,
            model_state_label=model_state_label,
            method_name=method_name,
            compress_fn=fn,
            x_eval=x_eval,
            y_eval=y_eval,
            seed=seed,
        )
        row["family"] = family_name
        row["param"] = param
        rows.append(row)

    passing = [r for r in rows if r[quality_metric] <= quality_threshold]

    if passing:
        best = max(passing, key=lambda r: r["compression_ratio"])
        status = "met_threshold"
    else:
        best = min(rows, key=lambda r: r[quality_metric])
        status = "threshold_not_met"

    return {"family": family_name, "status": status, "best": best, "all_rows": rows}


def run_budget_search(
    *,
    model,
    layer_name: str,
    layer_shape: tuple[int, int],
    model_state_label: str,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    quality_threshold: float = 0.05,
) -> dict:
    max_rank = min(layer_shape)
    families = all_families(seed, max_rank)

    results = {
        family_name: sweep_family(
            model=model,
            layer_name=layer_name,
            model_state_label=model_state_label,
            family_name=family_name,
            configs=configs,
            x_eval=x_eval,
            y_eval=y_eval,
            seed=seed,
            quality_threshold=quality_threshold,
        )
        for family_name, configs in families.items()
    }

    met = {name: r for name, r in results.items() if r["status"] == "met_threshold"}
    overall_best = max(met.values(), key=lambda r: r["best"]["compression_ratio"]) if met else None

    return {
        "layer": layer_name,
        "layer_shape": list(layer_shape),
        "model_state": model_state_label,
        "seed": seed,
        "quality_threshold": quality_threshold,
        "families": results,
        "overall_best_family": overall_best["family"] if overall_best else None,
        "overall_best_ratio": overall_best["best"]["compression_ratio"] if overall_best else None,
    }
