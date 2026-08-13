"""Experiment 29 (budget search): the decisive depth-gradient reading for
`microsoft/DialoGPT-small` -- same architecture/scale as `gpt2` (124M,
12 blocks), but fine-tuned from gpt2's own weights on dialogue data
rather than distilled or trained from scratch. Tests whether it matches
gpt2's early-block-dominant achievable-ratio pattern (supporting
"distillation specifically" as Experiment 21's explanation) or
distilgpt2's late-block-dominant one (weakening that story).

Uses the parallel, checkpointed wrapper (verified in
experiments/verify_parallel_budget_search.py) given this model's
124M-parameter scale.

MUST set BLAS thread-limiting env vars before numpy/torch are imported
anywhere in this process -- see parallel_budget_search.py's docstring.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from atlas_nn.stage_b.experiment import save_json  # noqa: E402
from atlas_nn.stage_c_real.parallel_budget_search import run_budget_search_parallel  # noqa: E402
from experiments.run_atlas_nn_stage_c_real_budget_search import (  # noqa: E402
    QUALITY_THRESHOLD,
    RANDOM_SEEDS,
    budget_search_layer_names,
)

MODEL_NAME = "microsoft/DialoGPT-small"
OUTPUT_PATH = "results/atlas_nn_stage_c_real_budget_search_dialogpt.json"


def main() -> None:
    layer_names = budget_search_layer_names(MODEL_NAME)
    print(f"model={MODEL_NAME} layers={layer_names} seeds={RANDOM_SEEDS}", flush=True)

    searches = run_budget_search_parallel(
        MODEL_NAME, layer_names, RANDOM_SEEDS, QUALITY_THRESHOLD,
    )

    for search in searches:
        best = search["overall_best_ratio"]
        best_family = search["overall_best_family"]
        best_ratio_str = f"{best:.2f}" if best is not None else "NONE (no config met threshold)"
        print(
            f"state={search['model_state']:<11} seed={search['seed']:>3} layer={search['layer']:<32} "
            f"best_family={str(best_family):<18} best_ratio={best_ratio_str}",
            flush=True,
        )

    save_json(
        {
            "experiment": "atlas_nn-stage_c_real_budget_search",
            "model_name": MODEL_NAME,
            "random_seeds": list(RANDOM_SEEDS),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "tested_layers": layer_names,
            "searches": searches,
            "ran_via": "parallel_budget_search",
        },
        OUTPUT_PATH,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
