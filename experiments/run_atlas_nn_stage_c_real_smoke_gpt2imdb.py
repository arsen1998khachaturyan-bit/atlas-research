"""Experiment 30: a fifth pretrained model, chosen to test whether
Experiment 29's refined hypothesis ("started from a random init vs.
started from another model's already-trained weights", not distillation
specifically) is really about the training origin in general, or just
happens to hold for DialoGPT-small's specific dialogue fine-tuning task.

`lvwerra/gpt2-imdb` is config-identical to `gpt2` and `microsoft/
DialoGPT-small` (124M params, 12 blocks, 768-dim, 12 heads -- verified
via AutoConfig) but was produced by a third procedure again: ordinary
supervised fine-tuning of gpt2's own weights on IMDB movie-review text
(not dialogue, not distillation). If it matches distilgpt2/DialoGPT-
small's late-block-dominant pattern, that strengthens "any
derived-from-prior-weights training" as the driver. If it matches gpt2's
early-block-dominant pattern instead, DialoGPT-small's result would look
more like a one-off tied to its specific fine-tuning task, not a general
rule.

Reuses run_atlas_nn_stage_c_real_smoke.main unchanged (model-name
parameterized since Experiment 20).
"""
from __future__ import annotations

from experiments.run_atlas_nn_stage_c_real_smoke import main

if __name__ == "__main__":
    main(model_name="lvwerra/gpt2-imdb", output_path="results/atlas_nn_stage_c_real_smoke_gpt2imdb.json")
