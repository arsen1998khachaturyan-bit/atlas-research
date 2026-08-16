"""Experiment 33: smoke test on gpt2 fine-tuned for 5000 steps on the
diverse ten-topic corpus -- 10x Experiment 32's maximum step count,
testing whether the late:early gain ratio continues falling, plateaus,
or reverses at larger fine-tuning scale (the non-monotonic hypothesis
Experiments 31-32 raised).
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

CHECKPOINT_DIR = "results/finetune_ckpts/gpt2_ftdiverse_steps5000"

if __name__ == "__main__":
    main(model_name=CHECKPOINT_DIR, output_path="results/atlas_nn_stage_c_real_smoke_ftdiverse5000.json")
