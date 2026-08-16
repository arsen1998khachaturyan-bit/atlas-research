"""Experiment 20: does Experiment 18/19's finding (behavioral-robustness
effect + a depth gradient concentrated near the network's output) hold on
a second real pretrained model, or is it specific to distilgpt2?

`gpt2` (124M params, 12 blocks, same 768-dim/12-head config family as
distilgpt2 but twice the depth and not distilled) is the natural next
checkpoint to try: same architecture family, larger and undistilled, so
a different depth/magnitude profile would be genuinely informative rather
than just noise.

Reuses `run_atlas_nn_stage_c_real_smoke.main` unchanged (model-name
parameterized since this experiment), rather than duplicating the smoke
test loop.
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

if __name__ == "__main__":
    main(model_name="gpt2", output_path="results/atlas_nn_stage_c_real_smoke_gpt2.json")
