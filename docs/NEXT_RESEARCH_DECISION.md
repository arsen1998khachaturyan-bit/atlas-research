# Next Research Decision

Updated after Experiment 20 (gpt2 cross-model generalization check).
Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 18-19:** the behavioral-robustness effect reproduces on
real pretrained distilgpt2, and a full budget search found the largest
compression number in the project (307x at 2.3% behavioral error on one
late-block layer), with a clean depth gradient once each layer is allowed
to pick its own best compression method.

**Experiment 20 (this round) — the effect's magnitude generalizes to a
second, larger model; the depth-pattern shape story gets more
interesting, not simpler.** `gpt2` (124M, 12 blocks) shows the same
1.4x-8.0x gain range as distilgpt2 (1.2x-7.6x) at fixed compression
parameters, with tight seed-to-seed reproducibility. But two different
fixed methods (`svd_rank4`, `atlas_block_dict`) give two *contradictory*
depth pictures on gpt2 itself -- direct, independent confirmation of
Experiment 19's lesson that no single fixed-parameter method's depth
reading should be trusted; only a budget search (each layer picking its
own best method) gives a reliable depth gradient.

## 2. What failed / remains untested

- No budget search has been run on `gpt2` yet -- this is the direct,
  obvious next step (mirrors Experiment 19), and the only way to learn
  whether gpt2's *true* achievable-ratio depth gradient matches
  distilgpt2's (rising sharply toward the last block) or looks different.
- Only two models tested, both in the GPT-2 family. Whether the magnitude
  range (roughly 1-8x at fixed parameters, up to 300x+ at matched-quality
  budget search) holds for a differently-trained or differently-sized
  model (e.g. gpt2-medium, or a non-GPT-2 architecture) is untested.
- The mechanism questions (effective rank, LayerNorm, attention mixing --
  Experiments 10-17) have still not been checked against either real
  pretrained model.
- The MLP input-layer question remains the project's longest untouched
  open thread.

## 3. What worked

- Generalizing `atlas_nn/stage_c_real/model.py` to accept any GPT-2-family
  model by name (rather than writing a parallel, duplicated module for
  gpt2) made this cross-model check cheap to add and kept the interface
  consistent -- the same architecture-agnostic pattern this project has
  used since the Stage C-lite refactor, now proven across model *names*,
  not just model *classes*.
- Running the cheaper smoke-test methodology first (not a full budget
  search) on the second model was the right call: it answered the most
  important question (does the magnitude generalize) in ~75 minutes
  instead of ~3+ hours, and it surfaced Result 2 (the fixed-method depth
  disagreement) as a genuine new finding rather than something a full
  budget search would have quietly resolved without being noticed as a
  methodological point in its own right.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and now on a second independent real model.** The magnitude of the
core effect is not a distilgpt2 idiosyncrasy -- it reproduces cleanly on
a different-sized, undistilled model in the same family. This is the
kind of replication the mission's falsification discipline calls for
before treating a Stage C finding as more than a single-checkpoint
curiosity.

## 5. The single most informative next experiment

**Run the budget search on gpt2** (the same script pattern as Experiment
19, retargeted via the now-generalized `model_name` parameter) to learn
whether its true depth gradient matches distilgpt2's rising-toward-the-
end shape, or diverges the way the two fixed methods' readings already
disagreed with each other. This is the most direct way to find out
whether Experiment 19's headline 307x number and its depth story are a
distilgpt2 idiosyncrasy or a real property of GPT-2-family models more
broadly.

After that, in rough priority order: (a) a third, differently-sized
checkpoint (e.g. gpt2-medium) if the gpt2 budget search confirms the
pattern holds; (b) revisit the MLP input-layer question, now the
project's clearly longest-standing open thread; (c) fold Experiments
18-20 into the project-wide synthesis artifact, which currently stops at
Experiment 17.
