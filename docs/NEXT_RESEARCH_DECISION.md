# Next Research Decision

Updated after Experiment 24 (effective rank vs. compression gain on real
models), which found a second independent measure splitting real models
by training procedure (distilled vs. not), matching Experiments 21/23's
depth-gradient split. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 18-23:** the behavioral-robustness effect reproduces on
three real pretrained models (distilgpt2, gpt2, gpt2-medium); the
achievable-ratio depth gradient reverses between distilgpt2 (distilled,
late-block-dominant) and both gpt2 and gpt2-medium (non-distilled,
early-block-dominant, closely matching each other despite a 3x scale
difference) -- narrowing Experiment 21's open question toward training
procedure over model scale as the explanation.

**Experiment 24 (this round) -- a second, independent confirmation of
the same split.** Effective-rank shrinkage correlates with compression
gain strongly on gpt2 (r=0.83, the strongest correlation found anywhere
in this project), moderately on gpt2-medium (r=0.49), and has the wrong
sign entirely on distilgpt2 (r=-0.29). Two unrelated analyses -- depth-
gradient shape and effective-rank correlation -- now split the same three
models the same way, by training procedure, not size.

## 2. What failed / remains untested

- The scale-vs-procedure question is narrowed twice now, still not
  conclusively resolved. A fourth model isolating one factor (a second
  distilled model, or a non-distilled model at distilgpt2's exact scale)
  remains the decisive test.
- *Why* distillation would produce this specific pattern (both the depth
  shape and the rank-correlation sign) is not explained by either
  measure -- only that it correlates with training procedure, twice.
- The MLP input-layer question remains the project's longest-standing
  untouched thread -- now several rounds without any new lead, since
  Experiment 16.
- LayerNorm/attention-mixing mechanism questions (Experiments 11, 14, 17)
  have still not been checked against any real pretrained model.
- Experiment 24's own power caveat (effective n≈6 per model, not 18) is
  real and unresolved -- more seeds would help but effective rank of a
  fixed random-init scheme may simply not vary much with seed at this
  scale regardless of how many are added; a cleaner test might vary
  something else (e.g. which specific dimensions of noise the init
  scheme uses) rather than just adding more seeds.

## 3. What worked

- Choosing a cheap, no-restart-risk experiment (effective rank needs no
  new compression sweeps, just one SVD per already-tested layer on
  already-downloaded models) after two consecutive multi-hour runs had
  each been interrupted by a container restart -- got a real, informative
  result without gambling more hours of compute against a third restart.
- The pooling trap flagged in Experiment 7 (pooling across populations
  with different underlying relationships hides real effects) recurred
  here almost exactly -- pooling all 3 models gives r=0.23, hiding gpt2's
  r=0.83 and distilgpt2's r=-0.29 entirely. Checking per-model before
  trusting a pooled number, now established practice, caught it again.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the training-procedure-vs-scale question is now supported by
two independent lines of evidence rather than one.** Depth-gradient shape
and effective-rank correlation both split the same three real models the
same way. Still short of proof (needs a model that isolates the factor
directly), but this is meaningfully stronger than either measure alone.

## 5. The single most informative next experiment

Still, as after Experiment 23: **a fourth model isolating training
procedure from scale directly** (a second distilled model, or a
non-distilled model at distilgpt2's exact scale) -- now even more
motivated, since two independent measures agree on the split rather than
just one.

Given this session has run long (two container restarts cost several
hours of recompute on top of the originally-planned work) and the user is
away, the next reasonable steps, roughly in order of value per hour of
compute:

**(a)** The MLP input-layer question -- cheap (small synthetic MLP, no
model downloads, no multi-hour compute), and the project's clearly most
overdue open thread.

**(b)** A fourth pretrained model, if network/compute budget allows --
the decisive test for the scale-vs-procedure question, but requires
another multi-hour budget search (now checkpointed, so lower-risk than
before, but still a real time investment).

**(c)** Fold Experiments 18-24 into the project-wide synthesis artifact
(the "Slack Hypothesis" retrospective), which currently stops at
Experiment 17 and does not reflect any of the real-pretrained-model work
-- this is the single most valuable, safest use of remaining
autonomous-session time: no compute risk, consolidates six experiments'
worth of new findings (including the project's strongest results) into
the deliverable the user will actually see first.
