# Next Research Decision

Updated after Experiment 33, which confirmed the non-monotonic
fine-tuning-intensity hypothesis: at 5,000 steps the late:early ratio
reverses past parity (4.04), matching the derived models' direction for
the first time in this project's controlled sweeps. Covers Track B
(`atlas_nn`) only.

## 1. What we learned

**Experiments 31-32:** neither fine-tuning duration alone (up to 500
steps) nor corpus diversity alone explains the depth-gradient reversal's
magnitude -- both converged to nearly the same ratio (~0.07) regardless
of corpus, always moving further from the derived models' late-dominant
pattern.

**Experiment 33 (this round) -- the reversal appears at larger scale.**
Extending the same diverse corpus to 5,000 steps (10x further) produced
a late:early ratio of 4.04 -- solidly late-dominant, matching
distilgpt2/DialoGPT-small/gpt2-imdb's direction for the first time. Full
trajectory: 0.29 (base) -> 0.285 (20 steps) -> 0.071 (500 steps) -> 4.04
(5,000 steps) -- clearly non-monotonic, a dip followed by a large
reversal. `transformer.h.0.attn.c_proj`'s achievable ratio, unmoved
(384x) across every checkpoint tested before this one, finally shifted
(to 5.28x) -- first sign of genuine widespread disruption at this
scale, not a late-block-only effect.

## 2. What failed / remains untested

- This is one 3-point trajectory (20/500/5,000 steps) on one base model
  and one corpus. The exact shape between 500 and 5,000 steps is
  unknown -- does the ratio cross 1.0 gradually or sharply? Does it
  keep rising past 4.04 with more steps, or plateau, or overshoot and
  fall back? A denser sweep (e.g. 1000/2000/3000/5000/10000) would
  answer this.
- Whether the reversal happens at a similar step count on the *narrow*
  corpus (only tested up to 500 steps in Experiment 31) is untested --
  if diversity truly doesn't matter (as Experiment 32 suggested at 500
  steps), the narrow corpus should reverse too, at a similar scale.
  Worth checking as a replication.
- Whether this generalizes to a different base model (not just gpt2) or
  a different downstream task is untested -- one model, one corpus
  family (even if now two variants), one training recipe.
- *Why* the reversal happens mechanistically (what changes in the
  weights between 500 and 5,000 steps that produces late-block
  compressibility) is completely open -- this experiment establishes
  *that* it happens, not *why*.
- *Why* Stage C-lite's small from-scratch Transformer diverges from real
  non-distilled Transformers on delta-rank fraction (Experiment 27) is
  still unexplained, and now newly relevant: delta-rank analysis on
  these fine-tuning-intensity checkpoints (cheap, reuses existing
  compression_gain data per Experiment 24/28's method) could show
  whether the same rank-based signature that explains real models'
  magnitude also explains this trajectory's reversal.

## 3. What worked

- Committing to the larger-scale test after two refutations, rather than
  concluding the magnitude mechanism was simply unexplainable --
  Experiment 33 is the payoff of treating "duration doesn't work in this
  range" as underspecified rather than as a dead end.
- Reusing Experiment 32's exact corpus and continuing to a much larger
  step count in one script made the trajectory directly comparable
  point-by-point.
- The checkpointed parallel budget search infrastructure, unchanged
  since Experiment 23, scaled to this sixth fine-tuning-derived
  checkpoint with zero modification.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the mechanism picture is now much more complete than at any
earlier point.** The core behavioral-robustness effect is unaffected and
has held on every real and self-fine-tuned checkpoint tested. The
depth-gradient reversal, previously explained only by "training origin"
(from-scratch vs. derived) as a binary property, now has a first
concrete account of *how* a model gets from one regime to the other: a
trajectory over training scale, not a fixed property of its starting
point. Two specific, cheap-relative-to-the-payoff hypotheses were
refuted before this one confirmed -- exactly the falsification discipline
this project has followed throughout.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** Run delta-rank analysis (Experiment 24/28's method, cheap --
reuses already-computed compression_gain, no new compression sweeps) on
the three diverse-corpus checkpoints (20/500/5,000 steps) to see if the
rank-based signature that explains real models' magnitude also tracks
this trajectory's dip-then-reversal shape. This is the cheapest next
step and could connect two previously-separate analysis threads.

**(b)** Fill in the trajectory between 500 and 5,000 steps (e.g. a
checkpoint at 1,500-2,000 steps) to see whether the ratio crosses 1.0
gradually or sharply -- narrows exactly where the transition happens.

**(c)** Replicate on the narrow corpus at 5,000 steps to check whether
the reversal is corpus-independent (as Experiment 32's 500-step result
would predict) or whether diversity matters after all at larger scale.
