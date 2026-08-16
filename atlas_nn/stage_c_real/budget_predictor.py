from __future__ import annotations

import json
from pathlib import Path

# Turns the project's central finding (Experiments 21-33: which
# compression method family wins, as a function of layer depth and
# whether the model was trained from scratch or derived from another
# model's weights) from a fact we've observed into a cheap PREDICTION
# that could replace an expensive full budget search on a new model.
#
# A full budget search evaluates all 31 configs (6 quantize + 7 svd +
# 5 prune + 4 vector_codebook + 9 atlas_block_dict) per layer per model
# state to find the best-at-quality-bar method. If depth + training
# origin alone can predict which FAMILY will win, a new model only
# needs that one family's configs evaluated -- a large, measurable
# compute reduction -- while recovering most of the achievable
# compression ratio.

FAMILY_CONFIG_COUNTS = {
    "quantize": 6,
    "svd": 7,
    "prune": 5,
    "vector_codebook": 4,
    "atlas_block_dict": 9,
}
TOTAL_CONFIGS = sum(FAMILY_CONFIG_COUNTS.values())

# Origin classification used throughout Experiments 21-30: models
# trained from a random initialization vs. models that started from
# another model's already-trained weights (distillation or fine-tuning).
FROM_SCRATCH_MODELS = {"gpt2", "gpt2-medium"}
DERIVED_MODELS = {"distilgpt2", "microsoft/DialoGPT-small", "lvwerra/gpt2-imdb"}

REAL_MODEL_FILES = {
    "distilgpt2": "results/atlas_nn_stage_c_real_budget_search.json",
    "gpt2": "results/atlas_nn_stage_c_real_budget_search_gpt2.json",
    "gpt2-medium": "results/atlas_nn_stage_c_real_budget_search_gpt2_medium.json",
    "microsoft/DialoGPT-small": "results/atlas_nn_stage_c_real_budget_search_dialogpt.json",
    "lvwerra/gpt2-imdb": "results/atlas_nn_stage_c_real_budget_search_gpt2imdb.json",
}

# The fine-tuning-trajectory checkpoints (Experiments 31-33) are not
# used to fit the predictor (their "origin" is not a clean binary
# category -- that is the whole point of Experiment 33) but are
# evaluated against it afterward as a stress test.
TRAJECTORY_FILES = {
    "gpt2_ft_steps20": "results/atlas_nn_stage_c_real_budget_search_ft20.json",
    "gpt2_ft_steps500": "results/atlas_nn_stage_c_real_budget_search_ft500.json",
    "gpt2_ftdiverse_steps20": "results/atlas_nn_stage_c_real_budget_search_ftdiverse20.json",
    "gpt2_ftdiverse_steps500": "results/atlas_nn_stage_c_real_budget_search_ftdiverse500.json",
    "gpt2_ftdiverse_steps5000": "results/atlas_nn_stage_c_real_budget_search_ftdiverse5000.json",
}


def load_searches(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["searches"]


def depth_label(layer_name: str, blocks: tuple[int, int]) -> str:
    block = int(layer_name.split(".")[2])
    return "early" if block == min(blocks) else "late"


def layer_type(layer_name: str) -> str:
    return layer_name.split(".", 3)[-1]


def infer_blocks(searches: list[dict]) -> tuple[int, int]:
    blocks = sorted({int(s["layer"].split(".")[2]) for s in searches})
    return (blocks[0], blocks[-1])


def build_pretrained_records(model_name: str, origin: str, path: str) -> list[dict]:
    searches = load_searches(path)
    blocks = infer_blocks(searches)
    records = []
    for s in searches:
        if s["model_state"] != "pretrained":
            continue
        records.append({
            "model": model_name,
            "origin": origin,
            "depth": depth_label(s["layer"], blocks),
            "layer_type": layer_type(s["layer"]),
            "layer": s["layer"],
            "true_best_family": s["overall_best_family"],
            "true_best_ratio": s["overall_best_ratio"],
            "original_bytes": s["families"]["quantize"]["best"]["original_bytes"],
            "families": s["families"],
        })
    return records


def family_ratio_if_met(families: dict, family_name: str) -> float | None:
    entry = families.get(family_name)
    if entry is None or entry["status"] != "met_threshold":
        return None
    return entry["best"]["compression_ratio"]


def family_compressed_bytes_if_met(families: dict, family_name: str) -> int | None:
    entry = families.get(family_name)
    if entry is None or entry["status"] != "met_threshold":
        return None
    return entry["best"]["compressed_bytes"]


def predict_family(records: list[dict], held_out_model: str, depth: str, ltype: str, origin: str) -> str:
    """Leave-one-model-out majority vote: among all pretrained records
    sharing the same origin category, depth, and layer type -- excluding
    the held-out model itself -- pick the most common winning family.
    Ties (and the from-scratch category's single-voter case) fall back
    to `quantize`, the one family verified safe on every random-init
    layer tested in this project."""
    votes: dict[str, int] = {}
    for r in records:
        if r["model"] == held_out_model or r["origin"] != origin:
            continue
        if r["depth"] != depth or r["layer_type"] != ltype:
            continue
        votes[r["true_best_family"]] = votes.get(r["true_best_family"], 0) + 1
    if not votes:
        return "quantize"
    best_count = max(votes.values())
    winners = sorted(f for f, c in votes.items() if c == best_count)
    if len(winners) > 1:
        return "quantize"
    return winners[0]


def evaluate_predictor() -> dict:
    all_records = []
    for model_name, path in REAL_MODEL_FILES.items():
        origin = "from_scratch" if model_name in FROM_SCRATCH_MODELS else "derived"
        all_records.extend(build_pretrained_records(model_name, origin, path))

    per_row = []
    for r in all_records:
        pred_family = predict_family(all_records, r["model"], r["depth"], r["layer_type"], r["origin"])
        achieved = family_ratio_if_met(r["families"], pred_family)
        achieved_bytes = family_compressed_bytes_if_met(r["families"], pred_family)
        if achieved is None:
            achieved = family_ratio_if_met(r["families"], "quantize") or 1.0
            achieved_bytes = family_compressed_bytes_if_met(r["families"], "quantize") or r["original_bytes"]
            fell_back = True
        else:
            fell_back = False
        true_bytes = family_compressed_bytes_if_met(r["families"], r["true_best_family"])
        per_row.append({
            "model": r["model"],
            "origin": r["origin"],
            "depth": r["depth"],
            "layer_type": r["layer_type"],
            "predicted_family": pred_family,
            "true_family": r["true_best_family"],
            "exact_match": pred_family == r["true_best_family"],
            "fell_back_to_quantize": fell_back,
            "predicted_ratio": achieved,
            "true_ratio": r["true_best_ratio"],
            "recovery_fraction": achieved / r["true_best_ratio"] if r["true_best_ratio"] else None,
            "compute_fraction": FAMILY_CONFIG_COUNTS[pred_family] / TOTAL_CONFIGS,
            "original_bytes": r["original_bytes"],
            "predicted_compressed_bytes": achieved_bytes,
            "true_compressed_bytes": true_bytes,
        })
    return {"per_row": per_row}


def evaluate_on_trajectory() -> list[dict]:
    """Stress test: apply the predictor (fit only on the 5 real models
    above) to the fine-tuning-trajectory checkpoints from Experiments
    31-33, using each checkpoint's TRUE depth-gradient category (as
    established by its own budget search) only for reporting -- the
    predictor itself only ever sees `origin='from_scratch'` (since these
    are all fine-tuned gpt2, and 'derived' is the only alternative this
    predictor knows) applied uniformly, to see where a origin-blind
    real-world deployment would go wrong."""
    all_records = []
    for model_name, path in REAL_MODEL_FILES.items():
        origin = "from_scratch" if model_name in FROM_SCRATCH_MODELS else "derived"
        all_records.extend(build_pretrained_records(model_name, origin, path))

    results = []
    for ckpt_name, path in TRAJECTORY_FILES.items():
        searches = load_searches(path)
        blocks = infer_blocks(searches)
        for s in searches:
            if s["model_state"] != "pretrained":
                continue
            depth = depth_label(s["layer"], blocks)
            ltype = layer_type(s["layer"])
            for assumed_origin in ("from_scratch", "derived"):
                pred_family = predict_family(all_records, "__trajectory__", depth, ltype, assumed_origin)
                achieved = family_ratio_if_met(s["families"], pred_family)
                if achieved is None:
                    achieved = family_ratio_if_met(s["families"], "quantize") or 1.0
                results.append({
                    "checkpoint": ckpt_name,
                    "layer": s["layer"],
                    "depth": depth,
                    "layer_type": ltype,
                    "assumed_origin": assumed_origin,
                    "predicted_family": pred_family,
                    "true_family": s["overall_best_family"],
                    "exact_match": pred_family == s["overall_best_family"],
                    "predicted_ratio": achieved,
                    "true_ratio": s["overall_best_ratio"],
                    "recovery_fraction": achieved / s["overall_best_ratio"] if s["overall_best_ratio"] else None,
                })
    return results
