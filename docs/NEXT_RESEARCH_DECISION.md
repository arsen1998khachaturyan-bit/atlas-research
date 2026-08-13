# Next Research Decision

Updated after Experiment 23 (gpt2-medium budget search), which
substantially narrowed the scale-vs-training-procedure question raised by
Experiment 21's depth-gradient reversal. Covers Track B (`atlas_nn`)
only.

## 1. What we learned

**Experiments 18-21:** the behavioral-robustness effect reproduces on
real pretrained models; the achievable-ratio depth gradient reverses
between distilgpt2 (late-block-dominant) and gpt2 (early-block-dominant),
raising two untested explanations -- model scale/depth, or distilgpt2's
knowledge-distillation training procedure.

**Experiment 22:** gpt2-medium's fixed-method smoke test hinted its shape
resembles gpt2's, not distilgpt2's -- flagged explicitly as provisional
pending the actual budget search.

**Experiment 23 (this round) -- narrowed to one better-supported
hypothesis.** gpt2-medium's true depth gradient (32.9x block 0 vs. 9.3x
block 23, ~3.5:1) closely matches gpt2's own ratio (24.8x vs. 7.1x, also
~3.5:1) despite gpt2-medium having 2x gpt2's depth and 3x its parameter
count. Two very differently-scaled models sharing the same non-distilled
training recipe agree with each other; the one distilled model
(distilgpt2) disagrees with both. Training procedure -- specifically
distillation -- is now the better-supported explanation, though not
conclusively proven (would need a 4th model isolating one factor).

**Also learned (environment, not research):** this container restarted
unannounced twice during one overnight unattended session, killing two
separate multi-hour runs with zero partial results saved each time.
Built and verified per-layer checkpointing
(`atlas_nn/stage_c_real/parallel_budget_search.py`) in response --
future long runs in this environment should always checkpoint
incrementally.

## 2. What failed / remains untested

- The scale-vs-procedure question is narrowed, not resolved. A fourth
  model is needed: a second distilled model (confirms distillation
  specifically) or a non-distilled model at distilgpt2's exact scale
  (more cleanly isolates scale). Neither attempted yet.
- No mechanism work (effective rank, LayerNorm, attention mixing --
  Experiments 10-17) has been checked against any real pretrained model.
- Layer coverage remains a 6-of-many subset on the two larger models.
- The MLP input-layer question remains the project's longest-standing
  untouched thread, now several rounds without attention.
- The parallel/checkpointed infrastructure has only been exercised on
  Stage C (real) budget searches -- not yet applied to smoke tests or any
  other experiment family, though the same reload-cost and
  restart-vulnerability logic would apply there too.

## 3. What worked

- Building and verifying the parallel wrapper *before* trusting it for a
  real multi-hour run (matching sequential results exactly on a subset
  first) caught nothing wrong, but is exactly the discipline that would
  have caught a bug if one existed -- continuing this project's standing
  practice of internal-consistency checks before extending machinery.
- Responding to the second lost run by building checkpointing rather
  than just relaunching a third time blind -- a real engineering
  investment that will pay off on every future long run in this
  environment, not just this one.
- Reporting the "two non-distilled models agree, the distilled one
  doesn't" pattern as narrowing evidence rather than overclaiming
  resolution -- the honest scope (three models, not four; correlation
  between training procedure and pattern, not a controlled isolation of
  it) is stated plainly.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the mechanism picture for real pretrained models is now
sharper.** The core behavioral-robustness effect has now been confirmed
on three independent real pretrained checkpoints. Where the effect
concentrates in the network appears to depend on how the model was
trained (distilled vs. not) rather than simply how big it is -- a real,
falsifiable, substantially-narrowed finding that emerged from exactly the
kind of chained, skeptical follow-up this project's discipline calls for.

## 5. The single most informative next experiment

**A fourth model isolating training procedure from scale directly** --
either a second distilled model (to confirm distillation specifically
drives the late-block-dominant pattern) or a non-distilled model at
distilgpt2's exact scale (~82M, 6 blocks, to more cleanly isolate scale
from procedure). This is the decisive test the current three-model
evidence cannot provide on its own.

After that, in rough priority order: (a) revisit the MLP input-layer
question, now clearly overdue; (b) check mechanism questions (effective
rank, LayerNorm, attention) against a real pretrained model for the first
time; (c) extend the parallel/checkpointed infrastructure to Stage C
(real) smoke tests, given the same restart risk applies there; (d) fold
Experiments 18-23 into the project-wide synthesis artifact, which
currently stops at Experiment 17 and does not reflect any of the
real-pretrained-model work.
