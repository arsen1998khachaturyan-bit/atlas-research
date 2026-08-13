# Next Research Decision

Updated after Experiment 27's 8-seed addendum, which confirmed (rather
than resolved away) a non-replication of the project's strongest
mechanism finding on the Transformer. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 25-26:** delta-rank fraction (effective rank of the
training update) is the strongest mechanism correlation found in this
project (r=-0.75, MLP hidden layer, 6-condition capacity sweep).

**Experiment 27 + addendum (this round) -- confirmed non-replication on
the Transformer, checked at two seed counts.** At 3 seeds: pooled
r=-0.15, block 0 wrong direction, block 1 weak-right-direction. At 8
seeds (matching the project's later standard, and the same seed-count
increase that resolved Experiment 10/11's earlier ambiguity toward a
real positive result): pooled r=-0.06 (weaker), block 1 flips to the
*wrong* direction. More statistical power sharpened the non-relationship
rather than revealing a hidden one. The sublayer-type pattern is stable
across both seed counts and points opposite to the MLP's prediction:
`out_proj` has both the most-concentrated updates and the lowest gain.

## 2. What failed / remains untested

- *Why* delta-rank fraction (and final-matrix effective rank before it,
  Experiment 10) predicts the MLP well and the Transformer poorly is
  still unexplained -- two independent rank-based metrics have now hit
  the same architecture wall, which narrows the space of explanations
  (something about residual connections/LayerNorm's error-absorbing
  role, per the standing hypothesis from Experiments 11/13/14/17) but
  doesn't test it directly.
- Delta-rank fraction has not been checked on any real pretrained model.
- The scale-vs-training-procedure question from Experiments 21-24 remains
  open, separately from this thread.
- The MLP input-layer mystery itself (why the input layer's update stays
  diffuse regardless of capacity) still has no causal explanation, only
  the correlational evidence from Experiments 25-26.

## 3. What worked

- Not stopping at the first non-replication and not assuming a second
  seed-count check would automatically resolve it favorably (as
  Experiment 11's addendum did) -- running the check with a genuinely
  open mind about which direction it would go, and reporting the actual
  (sharpened-negative) result rather than the one that would have made a
  tidier narrative.
- Extending both the delta-rank script AND its upstream compression-gain
  source (the budget search) to 8 seeds together, rather than only
  adding seeds to the metric being tested while leaving its comparison
  data at lower power -- kept the two sides of the correlation on equal
  footing.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes on the core claim; the mechanism-hunting thread has now clearly
mapped its own boundary.** Two independent rank-based metrics (final
matrix, training update) each explain the MLP's compression-gain pattern
well and the Transformer's poorly, at good statistical power on both
architectures. This is a stable, well-established boundary of what these
specific metrics can explain -- not evidence against the underlying
compression-gain phenomenon, which remains robust across every
architecture and every real model tested throughout this project.

## 5. The single most informative next experiment

No single option clearly dominates; in rough priority order:

**(a)** A real pretrained-model check of delta-rank fraction, now that
the Stage C (real) infrastructure exists -- would add a third data point
(after MLP and Stage C-lite Transformer) on whether this metric is
architecture-general or MLP-specific, and real models are the highest-
value target given the project's recent emphasis there.

**(b)** A fourth pretrained model isolating scale from training
procedure (Experiments 21-24's still-open question) -- unrelated thread,
similar priority.

**(c)** Fold Experiments 18-27 into the project-wide synthesis artifact,
which currently stops at Experiment 17 and does not reflect any of the
real-pretrained-model or delta-rank work -- the safest, no-compute-risk
option, worth doing regardless of which research thread gets picked up
next.
