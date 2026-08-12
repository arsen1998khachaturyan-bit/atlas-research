# Next Research Decision

Updated after Experiment 16 (raw-vs-projected input test), which ruled
out a third specific hypothesis for the MLP input-layer question. Covers
Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–9:** the core phenomenon (trained networks tolerate
weight-compression error far better than tensor error predicts, scaling
with depth) is well-established across two architectures, three tasks.

**Experiments 10–14:** four architectural factors individually tested on
the Transformer mechanism — residual connections ruled out; effective
rank partially explains the MLP; LayerNorm drives rank-decoupling
(generalizes across architectures) and a secondary magnitude effect;
attention is the dominant magnitude driver. That thread reached a
well-resolved natural stopping point.

**Experiments 12, 15:** the MLP input layer's flat-to-negative
post-training compressibility does not track task-irrelevant input noise
fraction — confirmed cleanly at good power after fixing a data-size
confound.

**Experiment 16 (this round) — a third hypothesis ruled out:** does the
input layer's behavior come from literally seeing *raw, untransformed*
task input, unlike every other layer (which receives an already-processed
upstream representation)? Tested by prepending a frozen, never-trained
orthogonal projection before the trainable stack, so the first trainable
layer sees a fixed transform of the input instead. Result: no positive
gain (mean 0.94 raw vs. 0.80 projected) — if anything slightly worse
projected, the opposite of the hypothesis's prediction. Ruled out.

**Where the input-layer question stands after six experiments (4, 6, 7,
12, 15, 16):** three specific hypotheses tested and ruled out (effective
rank, noise fraction, raw-vs-processed input). The behavior itself — no
gain, sometimes a penalty, unique among all layer types in every
architecture tested — remains real, reproducible, and unexplained by
every mechanism tried so far.

## 2. What failed / remains untested

- Effective rank, noise fraction, and raw-vs-processed input — all three
  now ruled out as explanations for the input layer's behavior.
- Whether an input-facing layer in the Transformer shows an analogous
  pattern — not yet checked (option (a) from the prior round, still
  open).
- The attention-ablation parameter-count confound from Experiment 14
  (`NoMixingAttention` removes both cross-token mixing and ~3/4 of
  attention's parameters simultaneously) — a `MatchedParamNoMixingAttention`
  variant has been implemented (same parameter count as real attention,
  skips only the softmax mixing step) but not yet run at the time of this
  writing.
- A genuine pretrained-model test — still blocked by network policy.

## 3. What worked

- Testing the input-layer question with a third, structurally different
  hypothesis (preprocessing/rawness, distinct from both rank and noise
  fraction) rather than re-testing variations on the same idea kept the
  search genuinely exploratory rather than repetitive.
- Reusing the exact same budget-search methodology (Experiment 4/6/12/15)
  for a third time made this result directly comparable to the prior two
  null results without any new analysis machinery.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, unaffected — the input-layer question remains a mechanism detail,
not a threat to the core phenomenon.** Three consecutive, well-powered
null results on the input layer is itself informative: whatever explains
its behavior is not any of the "obvious" candidates (rank, noise ratio,
preprocessing), suggesting either a more subtle information-theoretic
property, an optimization-dynamics explanation (e.g. something about how
gradients specifically reach the first layer), or something not yet
hypothesized. This doesn't weaken the core phenomenon's evidence at all —
it narrows an open side-question.

## 5. The single most informative next experiment

Three specific hypotheses for the input layer have now failed. Rather
than continue guessing hypotheses, the two most promising open threads
are:

**(a)** Finish and run the `MatchedParamNoMixingAttention` check
(Experiment 17, implemented, not yet executed) — disentangles whether
Experiment 14's attention-magnitude finding was really about cross-token
mixing or just about attention's extra parameters. This is close to
completion and should be finished before opening new threads.

**(b)** Check whether an input-facing layer in the Transformer shows a
similar pattern to the MLP's input layer — would clarify whether "input
layers are different" is architecture-general (worth continued
investment) or MLP-specific (in which case further guessing at MLP-only
hypotheses has diminishing value).

**(c)** Given three clean negatives on the input-layer question and no
remaining strong hypothesis, it's reasonable to deprioritize further
*hypothesis-guessing* there specifically (distinct from (b), which tests
generality rather than guessing a new mechanism) and record it as a
standing open question rather than continuing an unguided search.

**Standing recommendation, unchanged:** a real pretrained checkpoint
remains the single highest-value possible addition to this research line
if network policy allows it at some point.
