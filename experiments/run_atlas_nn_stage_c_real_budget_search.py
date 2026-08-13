"""Follow-up to the Stage C (real) smoke test: turns the qualitative
tensor-vs-behavior gap into the same "achievable compression ratio at
matched behavioral quality" question Experiments 4/6/9 asked of the MLP
and the from-scratch Transformer -- now on a literal pretrained model
(distilgpt2), reusing atlas_nn.stage_b.budget_search.run_budget_search
unchanged (architecture-agnostic since the Experiment 8 refactor).

Layer scope: a full budget search (31 configs per layer -- see
atlas_nn.stage_b.budget_search.all_families) on all 12 of
atlas_nn.stage_c_real.model's tested layers, across 4 model states
(pretrained + 3 random-init seeds), is not practical on CPU in one session
-- the two slowest method families (vector-codebook k-means and the Atlas
block-dictionary's own k-means step) scale with element count and this
model's layers are far larger than Stage B/C-lite's. This script instead
covers a smaller, still depth-balanced subset: the first and last of
distilgpt2's 6 blocks (mirroring Stage C-lite's block-0-vs-block-1 depth
comparison), x 3 sublayer types (attention-out, FFN-in, FFN-out -- skipping
the fused qkv projection c_attn, whose 3x-wider shape has no clean analogue
in earlier experiments' per-type tables). This is a deliberate scope
reduction for compute reasons, not a hidden one.
"""
from __future__ import annotations

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.experiment import save_json
from atlas_nn.stage_c_real.dataset import build_eval_batch
from atlas_nn.stage_c_real.model import (
    evaluate,
    get_weight,
    load_pretrained,
    load_random_init,
    load_snapshot,
    load_tokenizer,
    set_weight,
    snapshot,
)

RANDOM_SEEDS = (11, 22, 33)
QUALITY_THRESHOLD = 0.05
BUDGET_SEARCH_SUFFIXES = ("attn.c_proj", "mlp.c_fc", "mlp.c_proj")
# distilgpt2 (6 blocks) was the original target; per-model overrides let a
# second model (e.g. gpt2, 12 blocks) pick its own first/last block pair
# instead of reusing distilgpt2's block indices verbatim.
BUDGET_SEARCH_BLOCKS_OVERRIDE = {"distilgpt2": (0, 5), "gpt2": (0, 11), "gpt2-medium": (0, 23)}


def budget_search_layer_names(model_name: str = "distilgpt2") -> list[str]:
    blocks = BUDGET_SEARCH_BLOCKS_OVERRIDE[model_name]
    return [
        f"transformer.h.{block}.{suffix}"
        for block in blocks
        for suffix in BUDGET_SEARCH_SUFFIXES
    ]


def run_state(model, state_label: str, seed: int, x_eval, all_searches: list, model_name: str) -> None:
    for layer_name in budget_search_layer_names(model_name):
        layer_shape = tuple(get_weight(model, layer_name).shape)
        search = run_budget_search(
            model=model,
            layer_name=layer_name,
            layer_shape=layer_shape,
            model_state_label=state_label,
            x_eval=x_eval,
            y_eval=None,
            seed=seed,
            get_weight=get_weight,
            set_weight=set_weight,
            evaluate=evaluate,
            snapshot=snapshot,
            load_snapshot=load_snapshot,
            quality_threshold=QUALITY_THRESHOLD,
        )
        all_searches.append(search)
        best = search["overall_best_ratio"]
        best_family = search["overall_best_family"]
        best_ratio_str = f"{best:.2f}" if best is not None else "NONE (no config met threshold)"
        print(
            f"state={state_label:<11} seed={seed:>3} layer={layer_name:<32} "
            f"best_family={str(best_family):<18} best_ratio={best_ratio_str}",
            flush=True,
        )


def main(model_name: str = "distilgpt2", output_path: str = "results/atlas_nn_stage_c_real_budget_search.json") -> None:
    tokenizer = load_tokenizer(model_name)
    x_eval = build_eval_batch(tokenizer)

    all_searches: list = []

    pretrained_model = load_pretrained(model_name)
    run_state(pretrained_model, "pretrained", seed=0, x_eval=x_eval, all_searches=all_searches, model_name=model_name)
    del pretrained_model

    for seed in RANDOM_SEEDS:
        random_model = load_random_init(seed, model_name)
        run_state(random_model, "random_init", seed=seed, x_eval=x_eval, all_searches=all_searches, model_name=model_name)
        del random_model

    save_json(
        {
            "experiment": "atlas_nn-stage_c_real_budget_search",
            "model_name": model_name,
            "random_seeds": list(RANDOM_SEEDS),
            "quality_threshold": QUALITY_THRESHOLD,
            "quality_metric": "relative_logit_error",
            "tested_layers": budget_search_layer_names(model_name),
            "searches": all_searches,
        },
        output_path,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
