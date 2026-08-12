"""One-off correctness check for atlas_nn.stage_c_real.parallel_budget_search:
runs a small subset (distilgpt2, 2 layers x 2 states) both the old
sequential way and the new parallel way, and asserts the numeric results
(overall_best_ratio, overall_best_family per layer/state) match exactly.
Not part of the pytest suite (too slow for CI) -- a manual gate before
trusting the parallel wrapper for a real multi-hour experiment.

MUST set BLAS thread-limiting env vars before numpy/torch are imported --
see atlas_nn/stage_c_real/parallel_budget_search.py's module docstring.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time  # noqa: E402

from atlas_nn.stage_b.budget_search import run_budget_search  # noqa: E402
from atlas_nn.stage_c_real.dataset import build_eval_batch  # noqa: E402
from atlas_nn.stage_c_real.model import (  # noqa: E402
    evaluate,
    get_weight,
    load_pretrained,
    load_random_init,
    load_snapshot,
    load_tokenizer,
    set_weight,
    snapshot,
)
from atlas_nn.stage_c_real.parallel_budget_search import run_budget_search_parallel  # noqa: E402

MODEL_NAME = "distilgpt2"
CHECK_LAYERS = ["transformer.h.0.attn.c_proj", "transformer.h.5.mlp.c_proj"]
CHECK_SEEDS = (11,)
QUALITY_THRESHOLD = 0.05


def run_sequential() -> list[dict]:
    tokenizer = load_tokenizer(MODEL_NAME)
    x_eval = build_eval_batch(tokenizer)
    results = []

    pretrained_model = load_pretrained(MODEL_NAME)
    for layer_name in CHECK_LAYERS:
        layer_shape = tuple(get_weight(pretrained_model, layer_name).shape)
        results.append(run_budget_search(
            model=pretrained_model, layer_name=layer_name, layer_shape=layer_shape,
            model_state_label="pretrained", x_eval=x_eval, y_eval=None, seed=0,
            get_weight=get_weight, set_weight=set_weight, evaluate=evaluate,
            snapshot=snapshot, load_snapshot=load_snapshot, quality_threshold=QUALITY_THRESHOLD,
        ))
    del pretrained_model

    for seed in CHECK_SEEDS:
        random_model = load_random_init(seed, MODEL_NAME)
        for layer_name in CHECK_LAYERS:
            layer_shape = tuple(get_weight(random_model, layer_name).shape)
            results.append(run_budget_search(
                model=random_model, layer_name=layer_name, layer_shape=layer_shape,
                model_state_label="random_init", x_eval=x_eval, y_eval=None, seed=seed,
                get_weight=get_weight, set_weight=set_weight, evaluate=evaluate,
                snapshot=snapshot, load_snapshot=load_snapshot, quality_threshold=QUALITY_THRESHOLD,
            ))
        del random_model

    return results


def key(search: dict) -> tuple:
    return (search["layer"], search["model_state"], search["seed"])


def main() -> None:
    print("running sequential (baseline)...", flush=True)
    t0 = time.time()
    seq_results = run_sequential()
    seq_seconds = time.time() - t0
    print(f"sequential done in {seq_seconds:.1f}s", flush=True)

    print("running parallel...", flush=True)
    t0 = time.time()
    par_results = run_budget_search_parallel(
        MODEL_NAME, CHECK_LAYERS, CHECK_SEEDS, QUALITY_THRESHOLD,
    )
    par_seconds = time.time() - t0
    print(f"parallel done in {par_seconds:.1f}s", flush=True)

    seq_by_key = {key(s): s for s in seq_results}
    par_by_key = {key(s): s for s in par_results}

    assert seq_by_key.keys() == par_by_key.keys(), (seq_by_key.keys(), par_by_key.keys())

    mismatches = []
    for k in seq_by_key:
        s, p = seq_by_key[k], par_by_key[k]
        if s["overall_best_ratio"] != p["overall_best_ratio"] or s["overall_best_family"] != p["overall_best_family"]:
            mismatches.append((k, s["overall_best_ratio"], s["overall_best_family"], p["overall_best_ratio"], p["overall_best_family"]))

    if mismatches:
        print("MISMATCHES FOUND:", flush=True)
        for m in mismatches:
            print(m, flush=True)
        raise SystemExit(1)

    print(f"MATCH: all {len(seq_by_key)} searches identical between sequential and parallel.", flush=True)
    print(f"speedup: {seq_seconds / par_seconds:.2f}x ({seq_seconds:.1f}s -> {par_seconds:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
