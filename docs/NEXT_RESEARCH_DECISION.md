# Next Research Decision

Updated after Experiment 28, which found the strongest single correlation
in the project (gpt2, r=-0.90) and a third independent measure of the
distilled-vs-non-distilled split running through Experiments 21-24.
Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 25-27:** delta-rank fraction is the strongest MLP mechanism
correlation (r=-0.75) but does not transfer to the small, from-scratch
Stage C-lite Transformer (r=-0.06, confirmed at 8 seeds).

**Experiment 28 (this round) -- transfers strongly to real, non-distilled
Transformers.** gpt2: r=-0.90 (strongest correlation in the project).
gpt2-medium: r=-0.60. distilgpt2: r=+0.19 (no relationship). This is the
third independent analysis (after Experiments 21/23's depth-gradient
shape and Experiment 24's final-matrix rank correlation) to split the
same three real models the same way -- non-distilled models resemble
each other, the distilled one doesn't, regardless of scale. It also
narrows Experiment 27's open question: the delta-rank mechanism isn't
"an MLP-only thing" -- it works strongly on real, well-trained
Transformers, just not on Stage C-lite's toy-scale from-scratch version,
and not on distilled models.

## 2. What failed / remains untested

- *Why* distillation disrupts both the depth-gradient shape and two
  independent rank-based correlations is still not established -- three
  measures now agree it does, none explain why.
- *Why* Stage C-lite's from-scratch Transformer showed no delta-rank
  relationship while real non-distilled Transformers show a strong one
  is also unexplained -- candidate factors (scale, training duration,
  real vs. synthetic data, optimizer schedule) are all confounded
  together in the comparison and not disentangled.
- A fourth model isolating scale from training procedure directly
  (Experiments 21-24's original open question) is still not run.
- The per-block breakdowns in both Experiment 24 and 28 are unreliable
  due to the seed-redundancy power caveat -- a genuinely different kind
  of seed variation (e.g. varying the random-init scheme itself, not
  just its seed) might get real per-block/per-layer power where simply
  adding more seeds cannot.

## 3. What worked

- Extending the cheap, no-restart-risk pattern from Experiment 24 (reuse
  already-computed compression_gain, one more SVD per layer) to a second
  metric -- kept this round of investigation fast and safe while still
  producing the strongest single result in the project.
- Catching the per-block correlations' implausible perfection (r=-0.9999
  fit through effectively 3 tripled points) before it could be
  mis-reported as a finding -- continuing the discipline Experiment 24
  established for exactly this failure mode.
- Explicitly connecting this result back to Experiment 27's unresolved
  question rather than treating it as a standalone finding -- it directly
  answers part of what Experiment 27 left open.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, more strongly than at any earlier point.** gpt2's r=-0.90 is the
single strongest piece of quantitative evidence in the entire project
for a mechanistic account of the compression-gain phenomenon. Combined
with three independent measures now agreeing on the distilled-vs-
non-distilled split, the mechanism picture for real pretrained models is
sharper than for either from-scratch architecture on its own.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** A fourth pretrained model isolating scale from training
procedure directly (Experiments 21-24's original open question, now
with three converging lines of evidence motivating it more than ever) --
a second distilled model, or a non-distilled model at distilgpt2's exact
scale.

**(b)** Investigate why Stage C-lite's from-scratch Transformer diverges
from real non-distilled Transformers on delta-rank fraction specifically
-- candidates include training duration/steps, real vs. synthetic data,
or scale; disentangling these would need new from-scratch Transformer
runs at varying duration/data-realism, a nontrivial new experiment
design.

**(c)** Fold Experiments 18-28 into the project-wide synthesis artifact,
which currently stops at Experiment 17 and does not reflect the real-
pretrained-model or delta-rank work -- the safe, no-compute-risk option,
worth doing regardless of which research thread is picked up next.
