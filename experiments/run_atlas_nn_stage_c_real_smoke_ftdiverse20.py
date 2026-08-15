"""Experiment 32 (checkpoint 1 of 2): smoke test on gpt2 fine-tuned for
20 steps on the diverse ten-topic corpus (see
experiments/run_finetune_diverse_checkpoints.py). Compares against the
500-step checkpoint to test the corpus-diversity hypothesis Experiment
31 raised.
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

CHECKPOINT_DIR = "results/finetune_ckpts/gpt2_ftdiverse_steps20"

if __name__ == "__main__":
    main(model_name=CHECKPOINT_DIR, output_path="results/atlas_nn_stage_c_real_smoke_ftdiverse20.json")
