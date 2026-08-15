# Next Research Decision

Updated after Experiment 31, which refuted its own predecessor's
magnitude-tracks-training-depth hypothesis via a direct, single-base-
model controlled test -- a real correction, not a confirmation. Covers
Track B (`atlas_nn`) only.

## 1. What we learned

**Experiment 30:** gpt2-imdb's late:early gain ratio (2.73:1) was much
weaker than distilgpt2's (17.3:1) or DialoGPT-small's (28.6:1),
suggesting fine-tuning duration/intensity might explain the magnitude
gradient among "derived-from-prior-weights" models.

**Experiment 31 (this round) -- direct test refutes it.** Fine-tuned
gpt2 itself (same base model, same architecture) on a fixed narrow
corpus for 20 vs. 500 steps. The late:early gain ratio moved the
*opposite* direction from predicted: 0.29 (base gpt2) -> 0.20 (20 steps)
-> 0.07 (500 steps) -- more training pushed *further* into gpt2's own
early-dominant pattern, not toward the derived models' late-dominant one.
The late block's mean gain fell monotonically (7.07x -> 5.07x -> 1.75x),
converging toward random-init parity. Training duration alone, holding
a narrow corpus fixed, is not the mechanism -- and can run backward.
Corpus diversity/naturalness is now the better-supported candidate
(untested directly): distilgpt2/DialoGPT-small/gpt2-imdb all used real,
diverse natural-language data; Experiment 31's corpus was small and
repetitive by construction.

## 2. What failed / remains untested

- The magnitude-tracks-training-depth hypothesis (Experiment 30) is now
  refuted as stated. Do not resurrect it without a corpus-diversity
  control.
- Corpus diversity/naturalness as the real driver of magnitude is a new
  hypothesis, itself completely untested -- would need fine-tuning gpt2
  on a real, diverse natural-language corpus (at comparable step counts
  to Experiment 31's sweep) to see if the late-block gain rises instead
  of falling.
- The five-model, zero-exception *sign* split (from-scratch vs.
  derived-from-prior-weights) from Experiments 21-30 is untouched by
  this result and remains the best-supported finding in this thread --
  only the magnitude explanation attached to it in Experiment 30 was
  wrong.
- *Why* Stage C-lite's small from-scratch Transformer diverges from real
  non-distilled Transformers on delta-rank fraction (Experiment 27) is
  still unexplained.
- The per-block breakdowns in Experiments 24/28 remain unreliable due to
  the seed-redundancy power caveat.

## 3. What worked

- Building the fine-tuning infrastructure cheaply (a ~70-line training
  loop, zero changes needed to any downstream smoke-test/budget-search
  code since `save_pretrained` output loads via the same `model_name`
  parameter as a HF Hub name) meant testing a real, falsifiable
  prediction cost about as much engineering effort as reusing an
  existing model.
- Running only the two extreme checkpoints (20, 500 steps) instead of
  all three saved roughly a third of the compute while still getting an
  unambiguous, monotonic-enough result to draw a conclusion; the third
  checkpoint (100 steps, infra already built) was correctly not run once
  the extremes disagreed with the hypothesis clearly.
- The checkpointed budget search survived a fifth container restart with
  zero lost compute (the finished ft20 result was already saved to disk
  before the restart hit).
- Stating the refutation as a visible, explicit correction to Experiment
  30's own claim (in `docs/BEST_RESULTS.md`, blockquoted in place) rather
  than quietly revising the earlier entry -- continuing this project's
  standing discipline for exactly this situation.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, for the core effect; the depth-gradient mechanism story is
narrower than it looked after Experiment 30.** The behavioral-robustness
/ compression-gain effect itself is unaffected by this result and has
now held on five real pretrained checkpoints with no exceptions. The
*explanation* for why derived-from-prior-weights models show late-block
dominance is back to "training origin matters, magnitude's cause is
unknown" rather than the more specific (and now wrong) "duration
explains magnitude" story Experiment 30 suggested.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** Test the corpus-diversity hypothesis directly: fine-tune gpt2 on
a real, diverse natural-language corpus (not the narrow template one) at
comparable step counts to Experiment 31's sweep, and check whether the
late:early ratio rises this time. This is the natural, most specific
next test given Experiment 31's result.

**(b)** Investigate why Stage C-lite's from-scratch Transformer diverges
from real non-distilled Transformers on delta-rank fraction specifically
(carried over from three prior decisions, still untouched).

**(c)** The steps=100 checkpoint's budget search is already built
(`run_atlas_nn_stage_c_real_budget_search_ft100.py`) but was held back;
running it would confirm whether the 20->500 trend is monotonic in the
middle or non-monotonic, though this is lower priority than testing the
corpus-diversity hypothesis directly since the sign of the effect is
already clear from the extremes.
