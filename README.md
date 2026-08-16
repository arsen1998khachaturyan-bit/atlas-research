# Atlas Research Framework v1.0

Atlas is a research framework for query-efficient extraction of compact symbolic programs from black-box behavior.

> **Note:** this repository now hosts two independent research tracks. This
> README describes Track A (symbolic active learning, below). Track B is a
> separate, newer research effort into compact/generative representations of
> trained neural-network weights, living in `atlas_nn/`; see
> `docs/CURRENT_STATE.md` for how the two relate, `docs/RESEARCH_LOG.md` for
> Track B's experiment log, and `docs/BEST_RESULTS.md` /
> `docs/NEXT_RESEARCH_DECISION.md` for its verified results and next steps.
> Run Track B's benchmark with `python -m experiments.run_atlas_nn_stage_a`.

## Current evidence

Internal experiments have shown that active information-gain query selection can require substantially fewer membership queries than passive or random querying on synthetic symbolic hypothesis spaces while preserving exact behavioral fidelity.

These results are promising, but they do **not** yet establish superiority on external benchmarks or against published active-learning methods.

## What this repository provides

- a shared `Oracle` interface;
- symbolic program and DSL primitives;
- passive, random, and active learners;
- reproducible blind and multi-seed benchmarks;
- JSON result export;
- command-line experiment runner;
- unit tests;
- one-click Colab smoke test.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pytest
```

## Run experiments

Quick smoke benchmark:

```bash
python -m experiments.run_smoke
```

Blind validation:

```bash
python -m experiments.run_blind_validation
```

Multi-seed replication:

```bash
python -m experiments.run_multiseed_validation
```

Results are written to `results/`.

## Research claims policy

Atlas should only make claims supported by reproducible experiments.

A new method is accepted only if it:

1. uses the same frozen benchmark protocol as the baseline;
2. preserves behavioral fidelity;
3. improves at least one predefined metric;
4. does not rely on hidden task identity;
5. reproduces across multiple seeds.

## Next scientific milestone

Evaluate Atlas on external automata-learning and program-synthesis benchmarks, and compare it against published methods rather than only passive and random baselines.
