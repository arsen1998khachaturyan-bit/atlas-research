# Next Research Decision

Updated after Experiment 29, which resolved the project's longest-running
open question (scale vs. training procedure as the driver of the
depth-gradient reversal) using a fourth real pretrained model, and
refined "distillation specifically" into a broader, better-supported
hypothesis. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 21-24, 28 (three converging measures, same three
checkpoints):** distilgpt2 (distilled) disagrees with gpt2 and
gpt2-medium (both trained from scratch) on depth-gradient shape,
final-matrix rank correlation, and delta-rank correlation. Scale was
ruled out as the driver (gpt2 vs. gpt2-medium agree despite a 3x/2x
scale-and-depth difference); "distillation specifically" vs. "training
procedure more broadly" remained open.

**Experiment 29 (this round) -- the decisive fourth model.**
`microsoft/DialoGPT-small` (124M, config-identical to gpt2, but
initialized from gpt2's own weights and fine-tuned on dialogue --
never distilled) shows distilgpt2's late-block-dominant depth pattern,
not gpt2's, and more extremely (28.6:1 late:early gain ratio vs.
distilgpt2's 17.3:1; block 0's mean gain is 0.59x, the only sub-1
block-mean seen on any real model). Since DialoGPT-small was never
distilled, "distillation specifically" is refuted as the mechanism. The
refined hypothesis: **training from a random initialization (gpt2,
gpt2-medium) vs. starting from another model's already-trained weights
(distilgpt2 via distillation, DialoGPT-small via fine-tuning)** is the
real dividing line. Also survived: a fourth container restart
mid-run, recovered cleanly via the checkpointing infrastructure built
after Experiment 23's restart, confirming it generalizes to new models
without modification.

## 2. What failed / remains untested

- *Why* starting from prior weights (rather than distillation per se)
  disrupts the depth-gradient shape and rank-based correlations is still
  not mechanistically explained -- four models and four measures now
  agree it does, none explain why.
- The from-scratch vs. derived-from-prior-weights split rests on **two**
  checkpoints per category, and the two "derived" checkpoints used
  mechanically different procedures (teacher distillation vs. ordinary
  fine-tuning) that happen to agree. A fifth model is needed to rule out
  coincidence at this sample size -- candidates: a second fine-tuned
  (non-distilled) derivative of gpt2, or a model fine-tuned on
  substantially more data to see if the effect strengthens with more
  post-initialization training.
- *Why* Stage C-lite's small from-scratch Transformer showed no
  delta-rank relationship while real non-distilled Transformers show a
  strong one (Experiment 27's open question) is still unexplained.
- The per-block breakdowns in Experiments 24/28 remain unreliable due to
  the seed-redundancy power caveat.

## 3. What worked

- The checkpointed parallel budget search survived a fourth unannounced
  container restart with zero wasted compute on completed layers --
  the infrastructure investment after Experiment 23 continues to pay for
  itself on every subsequent long-running experiment.
- Choosing a fourth model specifically to *discriminate* between two
  standing hypotheses, rather than to accumulate more of the same kind of
  evidence -- this is what let Experiment 29 actually resolve (refute a
  hypothesis, not just add a fourth data point).
- Catching and fixing a real float32 overflow bug (DialoGPT-small's
  logits) before it could contaminate results, via direct reproduction
  outside the pipeline rather than a guessed fix.
- Continuing to flag fixed-single-method smoke-test depth readings as
  provisional and requiring the budget search to confirm -- the smoke
  test's reading here (resembling gpt2) was in fact reversed by the
  budget search, the second time this project has seen a fixed-method
  reading mislead (after Experiments 19-21).

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes.** The behavioral-robustness / compression-gain effect itself has
now replicated on four independent real pretrained checkpoints with no
exceptions (every random-init budget search loses to `quantize`; every
pretrained search that clears the quality bar uses a structure-aware
method instead, 48/48 searches). Where the depth-gradient *shape*
reverses, that reversal is now mechanistically narrower and
better-explained than at any earlier point in the project.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** A fifth model testing the refined from-scratch-vs-derived
hypothesis directly -- e.g. a second fine-tuned-from-gpt2 model trained
on a different (non-dialogue) dataset, to check whether the effect is
about "started from prior weights" in general or something specific to
DialoGPT-small's dialogue fine-tuning task.

**(b)** Investigate why Stage C-lite's from-scratch Transformer diverges
from real non-distilled Transformers on delta-rank fraction specifically
(carried over from the prior decision, still untouched) -- candidates
include training duration/steps, real vs. synthetic data, or scale.

**(c)** Fold Experiment 29 (and the still-outstanding Experiments 18-28)
into the project-wide synthesis artifact, which currently stops at
Experiment 17 -- the safe, no-compute-risk option, worth doing regardless
of which research thread is picked up next.
