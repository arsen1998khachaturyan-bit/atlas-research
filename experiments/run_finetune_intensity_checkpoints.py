"""Experiment 31: build the checkpoints needed to test whether the
depth-gradient reversal's magnitude tracks fine-tuning intensity
(Experiment 30's open lead) -- fine-tunes gpt2 on the project's own
self-authored sentiment corpus for three different step counts, saving
a full checkpoint at each. One continuous training run (steps saved at
20/100/500 out of a single 500-step run, not three separate runs) so
the total training cost is just 500 steps, not 620.

Reproduce with `python -m experiments.run_finetune_intensity_checkpoints`.
Downstream smoke tests / budget searches then point
atlas_nn.stage_c_real.model's model_name at each saved directory --
that infra already accepts local paths unmodified (AutoConfig/
AutoModelForCausalLM.from_pretrained works the same way on a directory
as on a HF Hub name).
"""
from __future__ import annotations

from atlas_nn.stage_c_real.finetune import finetune_checkpoints

BASE_MODEL = "gpt2"
OUTPUT_DIRS = {
    20: "results/finetune_ckpts/gpt2_ft_steps20",
    100: "results/finetune_ckpts/gpt2_ft_steps100",
    500: "results/finetune_ckpts/gpt2_ft_steps500",
}


def main() -> None:
    finetune_checkpoints(BASE_MODEL, OUTPUT_DIRS, seed=0)
    print("done", flush=True)


if __name__ == "__main__":
    main()
