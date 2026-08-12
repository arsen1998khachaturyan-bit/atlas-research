"""Experiment 21: the gpt2 counterpart to Experiment 19's distilgpt2
budget search. Experiment 20 found that two different fixed compression
methods give contradictory depth readings on gpt2 -- this is the
budget-search methodology (each layer picks its own best method/
parameter) that Experiment 19 showed is the only trustworthy way to
measure the true achievable-ratio depth gradient, now applied to gpt2 to
see whether its depth pattern matches distilgpt2's (rising sharply toward
the last block, up to 307x at one layer) or looks different.

Reuses run_atlas_nn_stage_c_real_budget_search.main unchanged
(model-name parameterized since Experiment 20's generalization).
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_budget_search import main

if __name__ == "__main__":
    main(model_name="gpt2", output_path="results/atlas_nn_stage_c_real_budget_search_gpt2.json")
