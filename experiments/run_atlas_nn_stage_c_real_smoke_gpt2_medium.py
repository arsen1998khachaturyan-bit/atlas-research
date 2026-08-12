"""Experiment 22: does gpt2-medium (355M, 24 blocks, same from-scratch
non-distilled OpenAI training recipe as gpt2) reproduce gpt2's
early-block-dominant depth pattern, or diverge again the way gpt2 diverged
from distilgpt2?

This is a step toward disentangling Experiment 21's two open hypotheses
for the depth-gradient reversal: (a) model scale/depth, or (b) distilgpt2's
knowledge-distillation training procedure specifically. If gpt2-medium
(much larger than gpt2, but trained the same non-distilled way) still
matches gpt2's pattern rather than distilgpt2's, that weakens the pure
scale/depth explanation and strengthens the training-procedure one --
scale changed a lot (124M -> 355M) while the pattern, if it holds, would
not have.

Reuses run_atlas_nn_stage_c_real_smoke.main unchanged (model-name
parameterized since Experiment 20).
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

if __name__ == "__main__":
    main(model_name="gpt2-medium", output_path="results/atlas_nn_stage_c_real_smoke_gpt2_medium.json")
