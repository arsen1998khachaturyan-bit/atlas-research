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

Correctness-critical detail: the environment variables that cap BLAS
thread pools (OMP_NUM_THREADS etc.) must be set before numpy/torch are
first imported in this process, since OpenBLAS/MKL read them once at
library-init time -- setting them later (e.g. inside a multiprocessing
worker after fork, if numpy/torch were already imported pre-fork) would
not reliably take effect. Any script importing this module MUST set
those env vars first, before any other atlas_nn/torch/numpy import.
"""
from __future__ import annotations

import multiprocessing as mp

QUALITY_THRESHOLD_DEFAULT = 0.05


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


def _run_one_group(group: tuple) -> list[dict]:
    model_name, state_label, seed, layer_names, quality_threshold = group

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

    results = []
    for layer_name in layer_names:
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
        results.append(search)
        print(
            f"[worker state={state_label} seed={seed}] layer={layer_name} "
            f"best_family={search['overall_best_family']} best_ratio={search['overall_best_ratio']}",
            flush=True,
        )
    return results


def run_budget_search_parallel(
    model_name: str,
    layer_names: list[str],
    random_seeds: tuple[int, ...],
    quality_threshold: float = QUALITY_THRESHOLD_DEFAULT,
    n_workers: int | None = None,
) -> list[dict]:
    """Runs the same (layer x state) budget searches
    `run_atlas_nn_stage_c_real_budget_search.run_state` would run
    sequentially, but distributed across one worker process per model
    instance (pretrained + each random-init seed). No model object or
    torch tensor is ever pickled across the process boundary -- only the
    small, trivially-picklable `group` tuple of primitives -- avoiding any
    correctness risk from sharing mutable model state between workers.
    """
    groups = [(model_name, "pretrained", 0, layer_names, quality_threshold)]
    groups += [(model_name, "random_init", seed, layer_names, quality_threshold) for seed in random_seeds]

    if n_workers is None:
        n_workers = len(groups)

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        grouped_results = pool.map(_run_one_group, groups)

    return [search for group_results in grouped_results for search in group_results]
