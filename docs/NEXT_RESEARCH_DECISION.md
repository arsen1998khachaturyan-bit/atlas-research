# Next Research Decision

Updated after the effective-rank capacity metric analysis (Experiment 7),
which turned the capacity-sweep's qualitative finding into a quantitative,
testable correlation on the layer where it matters most. Covers Track B
(`atlas_nn`) only — Track A (the pre-existing symbolic active-learning
framework) is unaffected and out of scope for this document.

## 1. What we learned

**Stage A:** a purpose-built structural method finds affine-block
structure baselines miss, doesn't fake compression on random data, and
loses to SVD/zlib outside its target structure type. Narrow, verified.

**Experiment 4 (easy task) → 5 (harder task, no gain) → 6 (capacity
sweep, gain reappears with more capacity):** established, across three
linked experiments, that post-training compression headroom tracks *spare
network capacity relative to task difficulty* rather than "training" as a
blanket effect — a predicted reversal that was then observed, the
strongest evidentiary structure in the project so far.

**Experiment 7 (this round) — quantifying it:** effective rank (a spectral
capacity-usage metric, computed independently of any compression method)
predicts the *magnitude* of the compression gain, not just its sign — but
only on the hidden→hidden layer (r=0.67 Pearson, 0.62 Spearman, n=16). The
input layer's correlation is weak/inconsistent (consistent with its
already-known distinct, unexplained behavior), and the output layer's
correlation isn't a fair test (its rank range is nearly degenerate — only
2 possible values). Pooling all layers together erases the signal (masks a
real, layer-specific relationship as noise) — a methodological lesson in
its own right: this kind of cross-layer analysis needs to be done
per-layer, not pooled, given how different each layer's scale and
mechanism can be.

**Also found (again) while verifying this:** a training-success filter
that looked adequate (15-point accuracy margin over random-init) still let
a partially-failed run through, because it didn't check against that
condition's own achievable ceiling. Fixed, but this is the second time a
generic accuracy check needed strengthening — treat this as a standing
risk for any future experiment involving trained-vs-random comparisons,
not a one-off bug.

## 2. What failed

- Block-dictionary clustering vs. SVD/zlib outside its target structure
  (Stage A).
- `vector_codebook` never met the 5% behavioral-error bar (Experiment 4).
- The general "training creates compressibility" claim, as a blanket
  statement (Experiment 5) — survives only in the capacity-relative form.
- Reusing one training-hyperparameter set across network widths
  (Experiment 6) — broke at width 256.
- Pooling the effective-rank correlation across all three layers
  (Experiment 7) — the real per-layer signal (r≈0.67 on layer 2) is
  invisible in the pooled number (Spearman ≈0).
- A relative-only accuracy-margin filter for detecting failed training
  runs (Experiment 7) — needed a per-condition ceiling check added.

## 3. What worked

- Chaining falsifiable predictions across experiments (4→5→6) produced far
  stronger evidence than any single result could have.
- Checking training success explicitly, twice now, caught two different
  kinds of confound (total failure in Experiment 6, partial failure in
  Experiment 7) that would otherwise have quietly inflated the headline
  numbers.
- Computing the capacity metric independently of the compression search
  itself (pure linear algebra on the weight tensor, no compression method
  involved) means the correlation, where it holds, isn't circular — it's a
  genuinely independent quantitative check on the qualitative story.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, with the most specific and best-evidenced form of the claim to
date.** The current state of support: trained-network weights can be
represented with substantially less independent information while
preserving behavior, in a way that (a) is not a general property of
training but specifically of *unused capacity relative to task
difficulty*, confirmed via a predicted reversal (Experiments 4–6), and
(b) is now partially quantifiable via an independent spectral metric on
one architecturally identifiable layer type (hidden→hidden, Experiment 7).
This is a materially stronger, more mechanistic claim than "training
helps compression" — it says roughly *where* and *why*, on the small
synthetic setups tested so far.

**What's still missing:** why the input and output layers don't follow the
same rank-based story (two separate open questions, not yet investigated);
whether the r≈0.67 correlation is itself stable across more/different
conditions (n=16 is not large); and, as always, whether any of this
transfers beyond small synthetic MLPs.

## 5. The single most informative next experiment

Two reasonable next moves; recommending the first given it's cheap and
directly follows from this round's cleanest open question.

**(a) Investigate why the input layer doesn't follow the capacity story.**
Now flagged in three consecutive experiments (4, 6, 7) as consistently
different from the hidden layer, never explained. A concrete, cheap next
step: check whether the input layer's behavior is about *information
preservation* rather than *capacity* — e.g., does its compression
tolerance correlate with how much of the input signal is actually
task-relevant (2 of 32 dims for 2-XOR, 3 of 8 for 3-parity) rather than
with its effective rank? This is testable by sweeping the noise-dimension
ratio (fraction of task-irrelevant input features) independently of
network width, reusing the same infrastructure.

**(b) Move to Stage C** (a small pretrained transformer, real task) armed
with a much sharper hypothesis than before: expect compression gains
concentrated in layers with spare capacity relative to what the task
needs, measurable via effective rank, and expect the earliest/latest
layers to behave differently from the rest — a specific, falsifiable
prediction rather than a blind search.

Recommendation: **(a) first.** It's the same cheap CPU/synthetic
infrastructure, closes a named open question from three experiments in a
row rather than leaving it hanging, and — if it resolves cleanly — would
mean Stage C starts with an explanation for *all three* layer types
(input, hidden, output) instead of two out of three. If the user prefers
to move to Stage C now instead (the capacity story is already well-evidenced
enough to justify it), that's a reasonable call too — this is a
judgment call about depth-first vs. moving-on, not a blocked dependency.

No architectural blockers for either — (a) needs no new dependencies;
(b) would need a dataset/checkpoint source decision from the user when
reached.
