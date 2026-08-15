# Next Research Decision

Updated after Experiment 32, which refuted the corpus-diversity
hypothesis too, alongside Experiment 31's duration hypothesis --
two clean refutations in a row that narrow the search space for what
actually drives the depth-gradient reversal's magnitude. Covers Track B
(`atlas_nn`) only.

## 1. What we learned

**Experiment 31:** fine-tuning gpt2 on a narrow corpus for 20 vs. 500
steps pushed the late:early ratio further from the derived models'
pattern (0.29 -> 0.20 -> 0.07), refuting "duration alone."

**Experiment 32 (this round) -- "diversity alone" is refuted too, and
both corpora converge.** A substantially more diverse corpus (ten
topics, six grammatical shapes) showed a real difference at light
training (20 steps: diverse corpus exactly matched base gpt2, 0.285;
narrow corpus had already shifted to 0.20) -- but at 500 steps, both
corpora converged to essentially the same ratio (diverse: 0.071, narrow:
0.070), despite completely different text and very different training
loss trajectories (0.71 vs 1.42). Neither duration nor diversity, at the
20-500 step range tested, explains the magnitude. Both sweeps show the
same shape: fast initial divergence from base gpt2 that deepens with
more steps, always toward *more* early-dominance -- the opposite
direction from distilgpt2/DialoGPT-small/gpt2-imdb's late-dominant
pattern.

## 2. What failed / remains untested

- Both specific magnitude hypotheses proposed so far (duration in
  Experiment 30/31, diversity in Experiment 31/32) are refuted within
  the tested range (20-500 steps). A third, more speculative hypothesis
  is now the best-supported one: 500 steps may be far too small a
  fine-tuning budget compared to what distilgpt2 (full distillation),
  DialoGPT-small (large dialogue corpus), and gpt2-imdb actually
  underwent -- the true relationship between training amount and the
  late:early ratio may be non-monotonic (an early dip below base gpt2's
  own ratio, followed by a much larger eventual rise past parity into
  late-dominance), not monotonic in either direction as both experiments
  so far implicitly assumed.
- Testing the non-monotonic hypothesis would need checkpoints at
  10x-1000x more steps than Experiment 31/32's sweep (e.g. 5,000 or
  50,000 steps) -- a substantially larger compute commitment than either
  prior round, and worth scoping carefully before committing.
- The underlying five-model, zero-exception *sign* split (from-scratch
  vs. derived-from-prior-weights) from Experiments 21-30 remains
  untouched by both refutations and is still the best-supported finding
  in this thread.
- *Why* Stage C-lite's small from-scratch Transformer diverges from real
  non-distilled Transformers on delta-rank fraction (Experiment 27) is
  still unexplained.

## 3. What worked

- Reusing Experiment 31's exact step counts (20, 500) and training
  hyperparameters for Experiment 32's diverse-corpus sweep made the two
  results directly comparable without any normalization -- the
  convergence at 500 steps is a clean, unconfounded observation because
  of this.
- The `finetune_checkpoints()` refactor (optional `corpus` parameter,
  Experiment 32) required touching only the corpus-generation call site,
  not the training loop itself -- cheap to extend to a third corpus if
  the non-monotonic hypothesis is tested next.
- Two refutations in a row, both stated as visible corrections
  (`⚠`/`⚠⚠` blockquotes in `docs/BEST_RESULTS.md`) rather than silently
  revised, continuing this project's standing discipline.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, for the core effect; the depth-gradient magnitude mechanism is
now doubly narrowed rather than explained.** The behavioral-robustness /
compression-gain effect itself is unaffected by either refutation and
has held on five real pretrained checkpoints with no exceptions. Two
specific, falsifiable magnitude hypotheses have now been tested and
ruled out in the 20-500 step range, which is real progress (a smaller
remaining hypothesis space) even though neither refutation directly
explains the derived models' magnitude.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** Scope and potentially run a much larger fine-tuning-intensity
sweep (thousands to tens of thousands of steps) to test the
non-monotonic hypothesis directly -- this is the most specific
remaining lead from this thread, but is a substantially bigger compute
commitment than Experiments 31/32 and should be sized carefully
(consider a coarser step-count grid, e.g. 500/5,000/50,000, rather than
a dense sweep) before committing.

**(b)** Investigate why Stage C-lite's from-scratch Transformer diverges
from real non-distilled Transformers on delta-rank fraction specifically
(carried over from four prior decisions, still untouched).

**(c)** Fold Experiments 31-32 into the project-wide synthesis artifact
(already includes 18-31; would need one more update for 32) -- safe,
no-compute-risk, worth doing regardless of which thread is picked up
next.
