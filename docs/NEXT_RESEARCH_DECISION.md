# Next Research Decision

Updated after Experiment 25 (weight-delta rank analysis), the first
candidate explanation for the MLP input-layer mystery with a large,
cleanly-separated effect size. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 18-24:** the behavioral-robustness effect and its depth
gradient generalize to three real pretrained models; the depth-gradient
reversal (Experiment 21) is now better explained by training procedure
(distillation) than model scale, confirmed by two independent measures
(Experiments 21/23's depth shape, Experiment 24's effective-rank
correlation sign).

**Experiment 25 (this round) -- back to the project's longest-standing
open question, with a genuinely new tool.** Rather than another property
of the *final* trained matrix (three of which were already ruled out:
effective rank, noise fraction, raw-vs-processed input), measured the
effective rank of the training *update* itself (`W_trained - W_random`).
Result: the input layer's update uses 87.7% of its available rank
(diffuse, near full-rank); the hidden layer's uses only 34.9%
(concentrated, low-rank) -- zero overlap across all 8 seeds. Movement
*magnitude* doesn't separate the layers the same way; it's specifically
the *structure* of the change.

## 2. What failed / remains untested

- *Why* the input layer's update would be diffuse while the hidden
  layer's is concentrated -- Experiment 25 offers one unverified,
  untested story (the input layer must preserve every task-relevant
  coordinate somewhat independently; internal layers can route through
  fewer channels since downstream layers adapt around them) but does not
  test it.
- Whether delta-rank fraction predicts compression-gain *magnitude*
  (not just direction) the way Experiment 7's final-matrix effective
  rank did, across the harder-task/wider-network conditions Experiments
  5-6 used to strengthen that original finding -- direct, cheap, and
  the most natural immediate follow-up.
- Whether this generalizes to the Transformer (where the input-layer
  mystery has not been separately tested at all) or to any real
  pretrained model.
- The scale-vs-training-procedure question from Experiments 21-24 is
  still not conclusively resolved -- would need a fourth pretrained
  model isolating one factor.

## 3. What worked

- Taking `docs/NEXT_RESEARCH_DECISION.md`'s own prior note seriously --
  "would need a different kind of tool ... not another ablation of the
  same shape" -- rather than running a fourth variation of the same
  final-matrix-property test that had already failed three times. The
  new angle (rank of the *update*, not the *result*) found in one
  cheap experiment what three prior experiments' worth of final-matrix
  probing did not.
- Reusing the exact original Stage B setup (architecture, task, seed
  count) so the new result connects directly to the already-established
  compression-gain pattern rather than needing its own fresh baseline.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the project's longest-standing open question now has its
first real lead.** Not resolved -- a large, clean correlation between
delta-rank fraction and compression-gain direction is not yet a tested,
causal mechanism -- but this is the first candidate explanation in four
attempts with an effect size large enough to be worth building on rather
than another one-off negative result.

## 5. The single most informative next experiment

**Test whether delta-rank fraction predicts compression-gain magnitude,
not just direction**, across Experiment 6's capacity-sweep conditions
(different tasks, different widths) -- the same cross-check that turned
Experiment 7's final-matrix effective-rank finding from "one observation"
into "a replicated, quantitative relationship." Cheap (reuses the 2-XOR/
3-parity capacity-sweep infrastructure already built), and the natural
next step to determine whether Experiment 25's finding is a real
mechanism or a single-condition coincidence.

After that, in rough priority order: (a) check whether the same
delta-rank pattern holds on the Stage C-lite Transformer's input-facing
layer, extending the finding across architectures the way Experiments
10-17 did for the mechanism-hunting thread; (b) a fourth pretrained model
isolating scale from training procedure (Experiments 21-24's open
question); (c) fold Experiments 18-25 into the project-wide synthesis
artifact.
