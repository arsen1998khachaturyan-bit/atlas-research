# Next Research Decision

Updated after Experiment 27 (delta-rank on the Stage C-lite Transformer),
a clean non-replication that echoes Experiment 10's earlier finding for
the final-matrix version of this same mechanism-hunting approach. Covers
Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 25-26:** delta-rank fraction (effective rank of the
training update, not the final matrix) is the strongest mechanism
correlation found in this project (r=-0.75, MLP hidden layer, across a
6-condition capacity sweep).

**Experiment 27 (this round) -- doesn't transfer to the Transformer.**
Pooled r=-0.15, block 0 r=+0.16 (wrong direction), block 1 r=-0.15
(right direction, weak). A sublayer-type breakdown, if anything, points
opposite to the MLP's result. This closely echoes Experiment 10's
non-replication of the *final*-matrix effective-rank finding on this
same architecture -- both versions of rank-based mechanism-hunting
(final matrix, training update) hit the same wall on the Transformer.

## 2. What failed / remains untested

- Only 3 seeds used for Experiment 27 (Experiment 9's original power).
  Experiment 11's addendum showed raising 3->8 seeds materially changed
  a weak/inconsistent correlation's picture (there, revealing a real
  positive relationship that 3 seeds had obscured as slightly negative).
  Not yet tried here -- the most direct, cheap next step before treating
  this non-replication as final.
- Why delta-rank fraction (and final-matrix effective rank before it)
  works for the MLP but not the Transformer remains unexplained. A
  plausible, untested story: residual connections and LayerNorm (already
  implicated in Experiments 11/13/14/17 as an error-absorbing pathway
  independent of any single sublayer's own weight structure) may make
  rank-based metrics generally less informative on this architecture,
  regardless of which specific rank-based quantity is measured.
- Delta-rank fraction has not been checked on any real pretrained model.
- The scale-vs-training-procedure question from Experiments 21-24 remains
  open, separately from this thread.

## 3. What worked

- Treating "does this transfer to the Transformer" as a real, gate-worthy
  question rather than assuming a strong MLP result would generalize --
  exactly the discipline that caught the Experiment 7->10 non-replication
  the first time this exact pattern occurred, now applied consistently
  to a second, related metric.
- Flagging the classifier head's r=0.95 as a numerical artifact (constant
  input to a correlation, degenerate 2-dimensional max rank) before it
  could be mistaken for a real finding -- the same discipline Experiment
  7 established for this exact layer type.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes on the core claim; the delta-rank mechanism is now known to be
MLP-specific, not general.** The central behavioral-robustness effect
remains well-supported across every architecture and every real model
tested. The *specific* rank-based explanations for *why* (both final-
matrix and training-update versions) are now known to work well for the
MLP and poorly for the Transformer -- a real, useful narrowing of what
"the mechanism" actually is, not evidence against the phenomenon itself.

## 5. The single most informative next experiment

**Rerun Experiment 27 at 8 seeds**, mirroring exactly how Experiment 11's
addendum resolved a similar weak/ambiguous 3-seed correlation on this
same Transformer. Cheap (Stage C-lite training is fast) and the most
direct way to know whether this non-replication is real or another
small-sample artifact before treating it as settled.

After that, in rough priority order: (a) a fourth pretrained model
isolating scale from training procedure (Experiments 21-24's still-open
question); (b) a real pretrained-model check of delta-rank fraction,
now that the Stage C (real) infrastructure exists; (c) fold Experiments
18-27 into the project-wide synthesis artifact, which currently stops at
Experiment 17.
