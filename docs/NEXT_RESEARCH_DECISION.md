# Next Research Decision

Updated after Experiment 34, which turned the project's central
scientific finding into a small practical tool (a compression-family
predictor) and measured its real value and failure modes. Covers Track
B (`atlas_nn`) only.

## 1. What we learned

**Experiment 33:** the depth-gradient reversal's magnitude follows a
non-monotonic trajectory over fine-tuning scale, confirmed by catching
gpt2's own ratio cross from 0.07 to 4.04 between 500 and 5,000 steps.

**Experiment 34 (this round) -- from finding to tool.** A leave-one-
model-out predictor built from depth + training-origin alone (no new
compression sweeps, pure analysis of existing budget-search results)
recovers 79.6% of optimal aggregate compression using only 23% of the
31-config search space -- roughly a 4x search-cost reduction with most
of the benefit retained. Failure modes (large SVD/codebook outliers on
specific layers the predictor can't see) are identified and quantified,
not hidden. A stress test on the Experiment 31-33 fine-tuning trajectory
found the checkpoint that crossed into derived-like behavior (5,000
steps) is better predicted assuming "derived" origin than "from-scratch"
-- a small, secondary confirmation of Experiment 33's finding from an
independent angle.

## 2. What failed / remains untested

- The predictor's worst misses are concentrated on layers with unusually
  large SVD/vector-codebook outliers (e.g. DialoGPT-small's late
  `mlp.c_fc`, 1% recovery) -- a second-stage feature (e.g. flagging
  attn.c_proj specifically, which produced the largest outlier ratios
  project-wide: 384x/512x/96x) might close this gap without a full
  search; untested.
- The predictor has never been evaluated on a model it wasn't at least
  indirectly fit from (all 5 real models were both training data and
  test data via leave-one-out). A genuinely held-out sixth real model
  would be the real test of generalization.
- The two-tier deployment idea this experiment motivates (run the
  predicted family first, fall back to full search only if the quality
  bar isn't comfortably met) is not implemented -- only the underlying
  data supporting it (families' pass/fail status per model) is analyzed.
- The trajectory stress test's origin-assumption comparison (94.7% vs.
  88.0% at 5,000 steps) is 6 rows and secondary -- not independently
  verified evidence, just a consistency check.
- The magnitude-mechanism questions from Experiments 30-33 (why the
  reversal happens, whether it replicates on the narrow corpus, the
  exact shape between 500 and 5,000 steps) remain open, unaffected by
  this experiment.
- *Why* Stage C-lite's small from-scratch Transformer diverges from real
  non-distilled Transformers on delta-rank fraction (Experiment 27) is
  still unexplained.

## 3. What worked

- Reframing "the finding is scientific, not a product" honestly, then
  finding the cheapest legitimate path to something with direct
  practical value -- reusing already-computed budget-search JSONs meant
  this required zero new compute, only analysis, and produced a real,
  quantified number (79.6%/23%) in under an hour.
- Reporting the byte-weighted aggregate metric instead of an unweighted
  per-row average, which would have been misleadingly close to 100%
  despite several severe misses -- catching this before it became an
  overclaim.
- Explicitly quantifying where the predictor fails (not just reporting
  the headline recovery number) -- consistent with this project's
  standing discipline of reporting negative results as prominently as
  positive ones.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and it now has one small piece of direct practical value
attached.** The core phenomenon (trained networks tolerate compression
error better than random-init ones, with depth and training origin
predicting where and how much) has held on every real and self-
fine-tuned checkpoint tested. Experiment 34 shows this isn't purely
descriptive -- it can measurably cut the cost of applying compression to
a new model, though it is not, and does not claim to be, a novel
compression algorithm or a result tested at production scale.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** Add a layer-specific second-stage feature (e.g. flag
`attn.c_proj` specifically, the source of the largest-magnitude outlier
ratios project-wide) to the predictor and re-run the same
leave-one-model-out validation -- cheap (still pure analysis, no new
compute) and directly targets the identified failure mode.

**(b)** Test the corpus-diversity/scale trajectory questions still open
from Experiment 33 (narrow-corpus replication, denser step-count
sampling between 500 and 5,000).

**(c)** Investigate why Stage C-lite's from-scratch Transformer diverges
from real non-distilled Transformers on delta-rank fraction (carried
over from five prior decisions, still untouched).
