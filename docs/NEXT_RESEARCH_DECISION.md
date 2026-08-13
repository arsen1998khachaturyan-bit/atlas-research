# Next Research Decision

Updated after Experiment 26 (delta-rank capacity-sweep cross-check),
which turned Experiment 25's single-condition finding into the strongest
mechanism correlation found anywhere in this project. Covers Track B
(`atlas_nn`) only.

## 1. What we learned

**Experiment 25:** the training UPDATE's rank (not the final matrix's)
cleanly separates the input layer (diffuse, 87.7% of available rank) from
the hidden layer (concentrated, 34.9%), on one task/width, 8 seeds, zero
overlap.

**Experiment 26 (this round) -- confirmed and strengthened across a full
capacity-sweep grid.** Within the hidden layer, delta-rank fraction
correlates with compression-gain magnitude at r=-0.75 (Pearson),
r=-0.70 (Spearman) across 6 conditions (2 tasks x 3 widths) -- stronger
than Experiment 7's original final-matrix effective-rank finding
(r=0.67), the first cross-check in this project to produce a *larger*
effect than the result it verified. A visible pattern within the hard
`parity3` task alone: delta-rank fraction falls 0.60->0.39->0.19 as width
rises 16->64->256, while gain rises ~0.7x->1.0x->2.1x -- giving
Experiments 5-6's capacity/slack story a mechanistic complement (spare
capacity -> more concentrated training update -> more compressible).
The input layer still shows no within-layer relationship (r=0.18) --
its update stays diffuse regardless of task or width.

## 2. What failed / remains untested

- *Why* the input layer's update stays diffuse regardless of available
  capacity, while the hidden layer's concentrates when capacity allows,
  is still not established -- correlational evidence now spans two
  experiments and 6+ conditions, but no causal mechanism has been tested.
- Not yet checked on the Stage C-lite Transformer or any real pretrained
  model -- the natural next step, mirroring how Experiments 8-9 extended
  the original capacity-sweep finding beyond the MLP.
- The output layer's degenerate delta-rank-fraction (exactly 0.500,
  every seed, both experiments) has never been given a fair test --
  would need a task/architecture where the output layer isn't stuck at
  a 2-dimensional max rank.
- Experiments 21-24's scale-vs-training-procedure question (on real
  pretrained models) is still open, separately from this thread.

## 3. What worked

- Running the exact cross-check Experiment 25 itself proposed (does the
  finding predict magnitude, not just direction, across the established
  capacity-sweep grid) rather than treating a clean single-condition
  result as sufficient on its own -- this is precisely the discipline
  that turned Experiment 7's original finding into something trustworthy,
  now applied to a new metric with an even stronger result.
- Catching a sign-interpretation subtlety before writing it up: a
  negative correlation between delta-rank *fraction* (low = concentrated)
  and gain is the *same direction* as a positive correlation between
  rank *shrinkage* (high = more shrinkage) and gain in Experiment 7 --
  worth being explicit about this when comparing the two metrics'
  reported correlation signs going forward.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the mechanism picture for the MLP thread is now the most
precise it has been.** Delta-rank fraction is currently the single
strongest quantitative predictor of compression-gain magnitude found
anywhere in this project (r=-0.75, hidden layer, 6 conditions) --
stronger than the original effective-rank finding it was built to
extend. It also unifies two previously separate findings (the
capacity/slack story and the rank-based mechanism story) into one
picture: spare capacity changes how training updates weights, not just
how compressible the result ends up being.

## 5. The single most informative next experiment

**Check whether delta-rank fraction transfers to the Stage C-lite
Transformer**, the same generalization step Experiments 8-9 took for the
original capacity-sweep finding. This would test whether "spare capacity
concentrates the training update" is a general property of trained
networks or specific to the MLP, and would give the Transformer
mechanism-hunting thread (Experiments 10-17, which found effective rank
of the *final* matrix a weaker predictor there) a genuinely new angle to
try.

After that, in rough priority order: (a) a real pretrained-model check
of delta-rank fraction (would need access to a model's own random-init
counterpart, straightforward given the Stage C (real) infrastructure
already built); (b) a fourth pretrained model isolating scale from
training procedure (Experiments 21-24's still-open question); (c) fold
Experiments 18-26 into the project-wide synthesis artifact, which
currently stops at Experiment 17.
