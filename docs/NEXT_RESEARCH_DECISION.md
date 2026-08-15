# Next Research Decision

Updated after Experiment 30, which confirmed the training-origin
hypothesis a third time on an independent derivation procedure and
surfaced a new, untested idea: that the depth-gradient reversal's
*magnitude* may track how much post-initialization training occurred,
not just whether any occurred. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiment 29:** DialoGPT-small (fine-tuned from gpt2, never
distilled) showed distilgpt2's late-block-dominant pattern, refuting
"distillation specifically" in favor of "trained from scratch vs.
started from another model's already-trained weights."

**Experiment 30 (this round) -- a third derivation procedure agrees on
direction, disagrees on magnitude.** `lvwerra/gpt2-imdb` (gpt2
fine-tuned on IMDB reviews -- a third distinct procedure, on a third
distinct domain) shows the same late-block-dominant direction (2.73:1
late:early) as distilgpt2 (17.3:1) and DialoGPT-small (28.6:1), not
gpt2/gpt2-medium's early-dominant direction (0.28-0.29:1). Five for five
models now agree on sign with zero exceptions. But gpt2-imdb's effect is
an order of magnitude weaker than the other two "derived" models -- the
most plausible reading is that fine-tuning *depth* (corpus size,
training procedure intensity) tracks the magnitude, with distillation
and heavy dialogue fine-tuning producing much larger shifts than light
IMDB fine-tuning.

## 2. What failed / remains untested

- The magnitude-tracks-training-depth idea is new and unverified -- it
  would need fine-tuning duration/data volume varied directly on a
  single base model (e.g. checkpoint gpt2 fine-tuned on IMDB at several
  different training-step counts) to test as a continuous relationship,
  rather than inferred from comparing three unrelated, differently-
  confounded checkpoints.
- *Why* starting from prior weights (of any kind) disrupts the
  depth-gradient shape is still not mechanistically explained -- five
  models and multiple measures now agree it does, none explain why.
- *Why* Stage C-lite's small from-scratch Transformer showed no
  delta-rank relationship while real non-distilled Transformers show a
  strong one (Experiment 27's open question) is still unexplained.
- The per-block breakdowns in Experiments 24/28 remain unreliable due to
  the seed-redundancy power caveat.

## 3. What worked

- The checkpointed parallel budget search completed Experiment 30 in a
  single pass, no container restart -- the infrastructure has now
  survived one restart directly (Experiment 23) and avoided further loss
  on every run since, including two more multi-hour ones.
- Choosing a fifth model on a third domain (not dialogue, not
  distillation) specifically to stress-test whether Experiment 29's
  result generalizes or was task-specific -- it could have refuted the
  hypothesis and didn't, which is worth more than a model chosen to
  merely add a data point.
- Noticing the magnitude gradient rather than only checking direction --
  a binary hypothesis would have been satisfied by "same sign" alone and
  would have missed the more interesting, more specific pattern in the
  data.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes.** The core behavioral-robustness / compression-gain effect has
now replicated on five independent real pretrained checkpoints with no
exceptions (every random-init budget search loses to `quantize`; every
pretrained search that clears the quality bar uses a structure-aware
method, 120/120 layer-state searches at full granularity). The
depth-gradient reversal's *direction* is now explained by training
origin with strong, repeated support; its *magnitude* may be explained
by training depth, a promising but untested lead.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** Test the magnitude-tracks-training-depth hypothesis directly:
fine-tune (or find pre-existing checkpoints of) the same base model at
several different training intensities and check whether the
late:early gain ratio increases monotonically with training depth. This
is the most specific, most falsifiable next test the project has had in
several rounds.

**(b)** Investigate why Stage C-lite's from-scratch Transformer diverges
from real non-distilled Transformers on delta-rank fraction specifically
(carried over from two prior decisions, still untouched).

**(c)** Fold Experiments 29-30 (and the still-outstanding 18-28) into
the project-wide synthesis artifact, which currently stops at
Experiment 17 -- the safe, no-compute-risk option, worth doing regardless
of which research thread is picked up next.
