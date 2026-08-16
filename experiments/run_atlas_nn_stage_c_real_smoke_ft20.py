"""Experiment 31 (checkpoint 1 of 3): smoke test on gpt2 fine-tuned for
20 steps on the project's own sentiment corpus -- the lightest-intensity
point in the fine-tuning-depth sweep (see
experiments/run_finetune_intensity_checkpoints.py). Compares against the
100-step and 500-step checkpoints to check whether the depth-gradient
reversal's magnitude (Experiment 30's open lead) scales with how far
training moved the weights from gpt2's original values.

Reuses run_atlas_nn_stage_c_real_smoke.main unchanged -- it already
accepts any model_name transformers can load, including a local
directory path.
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

CHECKPOINT_DIR = "results/finetune_ckpts/gpt2_ft_steps20"

if __name__ == "__main__":
    main(model_name=CHECKPOINT_DIR, output_path="results/atlas_nn_stage_c_real_smoke_ft20.json")
