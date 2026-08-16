"""Experiment 29: a fourth pretrained model, chosen specifically to help
isolate Experiments 21-24/28's two competing explanations for the
depth-gradient reversal -- model scale, or distilgpt2's knowledge-
distillation training procedure.

`microsoft/DialoGPT-small` is the same architecture and scale as `gpt2`
(124M params, 12 blocks, 768-dim, 12 heads -- config-identical) but
trained differently: not distilled, and not pretrained from scratch --
it was initialized from gpt2's own weights and further trained
(fine-tuned) on dialogue data. This is a third training-procedure
category, neither of the first two: if it matches gpt2's early-block-
dominant pattern, that argues distillation specifically (not just "any
deviation from pure from-scratch training") drives distilgpt2's opposite
pattern. If it matches distilgpt2 instead, that weakens the
distillation-specific story.

Reuses run_atlas_nn_stage_c_real_smoke.main unchanged (model-name
parameterized since Experiment 20).
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

if __name__ == "__main__":
    main(model_name="microsoft/DialoGPT-small", output_path="results/atlas_nn_stage_c_real_smoke_dialogpt.json")
