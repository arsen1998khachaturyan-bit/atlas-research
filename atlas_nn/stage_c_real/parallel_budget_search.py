"""Multiprocess wrapper around atlas_nn.stage_b.budget_search.run_budget_search
for Stage C (real) models, where a single layer's full 31-config sweep can
take several minutes on CPU (Experiments 19, 21) -- the sequential scripts
used one of this container's 4 CPU cores at a time.

This module does NOT modify run_layer_experiment/run_budget_search
(the core, already-tested machinery every prior experiment in this project
depends on) -- it only parallelizes the *outer* loop over independent
model-state groups, each of which loads its own model instance and is
fully independent of every other group (no shared mutable state between
workers).

Grouping choice: one worker process per (state, seed) model instance
(pretrained + each random-init seed), each running all of that model's
layers sequentially, rather than one task per (layer, state) pair. This
matters in practice: loading gpt2-medium from local cache alone takes
~78s, so a flat per-layer task list (24 tasks for a 6-layer/4-state
sweep) would reload the model 24 times -- roughly half an hour wasted on
nothing but loading. Grouping by model instance loads each one exactly
once. With this project's usual 4 model states (1 pretrained + 3
random-init seeds) and this container's 4 CPU cores, this also happens to
map one-to-one onto full core utilization.

Checkpointing: this environment's container has been restarted twice
unannounced during long background runs in one session (once mid-way
through the gpt2-medium smoke test, once ~67% through the first attempt
at this module's own gpt2-medium budget search), each time losing all
in-memory progress -- but on-disk state survives the restart. Each
worker therefore writes its own partial results to `checkpoint_dir` after
every single layer's search completes, and `run_budget_search_parallel`
skips any (state, seed, layer) search whose checkpoint already exists on
a rerun. This turns "lose the whole run" into "lose at most the one
layer-search in flight when the container died."

Correctness-critical detail: the environment variables that cap BLAS
thread pools (OMP_NUM_THREADS etc.) must be set before numpy/torch are
first imported in this process, since OpenBLAS/MKL read them once at
library-init time -- setting them later (e.g. inside a multiprocessing
worker after fork, if numpy/torch were already imported pre-fork) would
not reliably take effect. Any script importing this module MUST set
those env vars first, before any other atlas_nn/torch/numpy import.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

QUALITY_THRESHOLD_DEFAULT = 0.05
DEFAULT_CHECKPOINT_DIR = "results/.budget_search_checkpoints"


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def _checkpoint_path(checkpoint_dir: str, model_name: str, state_label: str, seed: int, layer_name: str) -> Path:
    safe_model = model_name.replace("/", "_")
    safe_layer = layer_name.replace(".", "_")
    return Path(checkpoint_dir) / f"{safe_model}__{state_label}_{seed}__{safe_layer}.json"


def _run_one_group(group: tuple) -> list[dict]:
    model_name, state_label, seed, layer_names, quality_threshold, checkpoint_dir = group
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    already_done: dict[str, dict] = {}
    remaining_layers = []
    for layer_name in layer_names:
        cp = _checkpoint_path(checkpoint_dir, model_name, state_label, seed, layer_name)
        if cp.exists():
            already_done[layer_name] = json.loads(cp.read_text())
        else:
            remaining_layers.append(layer_name)

    if not remaining_layers:
        print(f"[worker state={state_label} seed={seed}] all layers already checkpointed, skipping", flush=True)
        return [already_done[layer_name] for layer_name in layer_names]

    from atlas_nn.stage_b.budget_search import run_budget_search
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

    model = load_pretrained(model_name) if state_label == "pretrained" else load_random_init(seed, model_name)
    x_eval = build_eval_batch(load_tokenizer(model_name))

    results_by_layer = dict(already_done)
    for layer_name in remaining_layers:
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
            quality_threshold=quality_threshold,
        )
        results_by_layer[layer_name] = search
        cp = _checkpoint_path(checkpoint_dir, model_name, state_label, seed, layer_name)
        cp.write_text(json.dumps(search, indent=2, default=str))
        print(
            f"[worker state={state_label} seed={seed}] layer={layer_name} "
            f"best_family={search['overall_best_family']} best_ratio={search['overall_best_ratio']} "
            f"(checkpointed)",
            flush=True,
        )

    return [results_by_layer[layer_name] for layer_name in layer_names]


def run_budget_search_parallel(
    model_name: str,
    layer_names: list[str],
    random_seeds: tuple[int, ...],
    quality_threshold: float = QUALITY_THRESHOLD_DEFAULT,
    n_workers: int | None = None,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> list[dict]:
    """Runs the same (layer x state) budget searches
    `run_atlas_nn_stage_c_real_budget_search.run_state` would run
    sequentially, but distributed across one worker process per model
    instance (pretrained + each random-init seed). No model object or
    torch tensor is ever pickled across the process boundary -- only the
    small, trivially-picklable `group` tuple of primitives -- avoiding any
    correctness risk from sharing mutable model state between workers.

    Resumable: any (state, seed, layer) search already checkpointed under
    `checkpoint_dir` from a previous, interrupted run is loaded from disk
    and skipped rather than recomputed.
    """
    groups = [(model_name, "pretrained", 0, layer_names, quality_threshold, checkpoint_dir)]
    groups += [
        (model_name, "random_init", seed, layer_names, quality_threshold, checkpoint_dir)
        for seed in random_seeds
    ]

    if n_workers is None:
        n_workers = len(groups)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        grouped_results = pool.map(_run_one_group, groups)

    return [search for group_results in grouped_results for search in group_results]
