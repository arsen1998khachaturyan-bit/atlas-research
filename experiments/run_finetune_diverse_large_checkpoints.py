"""Experiment 33: does the late:early gain ratio reverse at larger
fine-tuning scale? Experiments 31-32 showed the ratio falling
monotonically through 500 steps regardless of corpus (narrow: 0.07,
diverse: 0.071) -- always in the *opposite* direction from
distilgpt2/DialoGPT-small/gpt2-imdb's late-dominant pattern. The
best-supported remaining hypothesis: 500 steps (~2,000 examples) may be
far too small a fine-tuning budget to reach the regime those models'
real training occupies, and the relationship may be non-monotonic (an
early dip, then eventually a much larger rise past parity at greater
scale).

This continues the diverse corpus from Experiment 32 (already
established diversity alone doesn't explain the difference at 500
steps, so there is no reason to also vary corpus here) to 5,000 steps --
10x Experiment 32's maximum -- as the first direct test of the
non-monotonic hypothesis. If the ratio keeps falling or plateaus, that
weakens the hypothesis further; if it turns and rises, that's the first
positive evidence for it.

Reproduce with `python -m experiments.run_finetune_diverse_large_checkpoints`.
"""
from __future__ import annotations

from atlas_nn.stage_c_real.diverse_corpus import generate_diverse_corpus
from atlas_nn.stage_c_real.finetune import finetune_checkpoints

BASE_MODEL = "gpt2"
OUTPUT_DIRS = {
    5000: "results/finetune_ckpts/gpt2_ftdiverse_steps5000",
}


def main() -> None:
    corpus = generate_diverse_corpus(4000, seed=0)
    finetune_checkpoints(BASE_MODEL, OUTPUT_DIRS, seed=0, corpus=corpus)
    print("done", flush=True)


if __name__ == "__main__":
    main()
