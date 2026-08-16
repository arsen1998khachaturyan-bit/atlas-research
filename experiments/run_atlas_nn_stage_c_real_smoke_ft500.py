"""Experiment 31 (checkpoint 3 of 3): smoke test on gpt2 fine-tuned for
500 steps -- the heaviest-intensity point. See
run_atlas_nn_stage_c_real_smoke_ft20.py for the full rationale.
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

CHECKPOINT_DIR = "results/finetune_ckpts/gpt2_ft_steps500"

if __name__ == "__main__":
    main(model_name=CHECKPOINT_DIR, output_path="results/atlas_nn_stage_c_real_smoke_ft500.json")
