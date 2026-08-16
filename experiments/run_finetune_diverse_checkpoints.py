"""Experiment 32: does fine-tuning corpus diversity (not duration) drive
the depth-gradient reversal's magnitude? Experiment 31 fine-tuned gpt2 on
a narrow, repetitive, single-domain corpus and found the late:early gain
ratio moving AWAY from the derived-models' pattern as steps increased
(0.29 -> 0.20 -> 0.07) -- refuting "duration alone" and raising corpus
diversity as the better-supported candidate instead. This experiment
holds duration fixed at the same two step counts (20, 500) for direct
comparability and swaps in a substantially more diverse self-authored
corpus (ten unrelated topics, six grammatical shapes, vs. one topic and
one shape in Experiment 31 -- see atlas_nn.stage_c_real.diverse_corpus).

If diversity is the real driver, the late:early ratio should rise
(toward or past 1) as steps increase this time, unlike Experiment 31.
If it still falls, diversity isn't the answer either and the mechanism
remains open.

Reproduce with `python -m experiments.run_finetune_diverse_checkpoints`.
"""
from __future__ import annotations

from atlas_nn.stage_c_real.diverse_corpus import generate_diverse_corpus
from atlas_nn.stage_c_real.finetune import finetune_checkpoints

BASE_MODEL = "gpt2"
OUTPUT_DIRS = {
    20: "results/finetune_ckpts/gpt2_ftdiverse_steps20",
    500: "results/finetune_ckpts/gpt2_ftdiverse_steps500",
}


def main() -> None:
    corpus = generate_diverse_corpus(4000, seed=0)
    finetune_checkpoints(BASE_MODEL, OUTPUT_DIRS, seed=0, corpus=corpus)
    print("done", flush=True)


if __name__ == "__main__":
    main()
