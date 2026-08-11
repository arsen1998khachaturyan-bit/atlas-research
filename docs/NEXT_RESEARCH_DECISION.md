# Next Research Decision

Updated after Experiment 11 (residual/LayerNorm ablation), which
separated two previously-conflated questions about the Transformer's
capacity-slack mechanism and found the rank-correlation part of
Experiment 10's hypothesis to be statistically unreliable at the sample
sizes available. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–9 (MLP + Transformer, phenomenon and quantitative
form):** trained networks are far more behaviorally robust to
weight-compression error than tensor error predicts; this scales with
depth; achievable compression ratio at matched quality increases
correspondingly. Reproduced across two architectures, three tasks.

**Experiment 10:** effective rank does not explain the Transformer's
gain the way it explained the MLP's — a genuine, if noisy, divergence.

**Experiment 11 (this round) — refining, not resolving, Experiment 10's
mechanism question:**
- The depth gradient (later layers gain more) is **architecture-
  independent within this Transformer** — it survives with residual
  connections and LayerNorm both removed. Whatever drives it is more
  fundamental than either feature.
- **LayerNorm, not residual connections, plausibly drives the overall
  gain magnitude** — removing it roughly halves-to-thirds the block-1
  gain in both available comparisons; removing residual connections
  alone barely moves it, contrary to the original hypothesis's emphasis.
- **The specific rank-shrinkage correlation from Experiment 10 (r=−0.54)
  turned out not to be reproducible** even under a same-condition rerun
  (a different implementation gave +0.33) — downgraded from "moderate
  negative correlation" to "no reliably measurable correlation at this
  sample size." The broader qualitative claim (no MLP-like positive
  relationship) still stands; the specific number does not.

## 2. What failed / remains untested

- The residual-connection half of Experiment 10's hypothesis — refuted
  by direction (removing residuals didn't shrink the gain).
- Precise correlation estimation between rank shrinkage and gain at
  Transformer scale — not achievable with n≈9 per block; would need
  either many more seeds or a design that produces more data points per
  condition (e.g. more layers, a deeper stack) to be trustworthy.
- Why the depth gradient exists at all, if not residual/LayerNorm — still
  completely open. Two features tested and neither explains it.
- The MLP's own still-unexplained input-layer behavior (open since
  Experiment 4) — untouched this round, now the longest-standing open
  question in the project.

## 3. What worked

- Splitting a bundled hypothesis ("residual/LayerNorm") into two
  independently toggleable factors caught a real asymmetry (LayerNorm
  matters for magnitude, residual connections don't) that a single
  on/off ablation would have missed entirely.
- Treating a same-condition rerun as an implicit reproducibility check
  caught Experiment 10's correlation number as fragile before it could
  be treated as settled — exactly the kind of self-correction the
  project's falsification discipline is meant to produce.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes for the phenomenon (now on especially solid ground — reproduced
across architectures, tasks, and now shown robust to two specific
architectural ablations); still open for any specific mechanism.**
Three rounds of mechanism-hunting (Experiments 7, 10, 11) have
established what does *not* fully explain the effect (rank alone;
residual connections) more clearly than what does (LayerNorm is a
plausible partial magnitude driver; the depth gradient itself remains
unexplained by anything tested). This is consistent with a real,
robust phenomenon whose mechanism is more subtle than any single
architectural feature — worth stating plainly rather than forcing a
premature mechanistic story onto noisy small-sample correlations.

## 5. The single most informative next experiment

**Pivot back to the MLP's input-layer mystery** (queued since Experiment
4, never investigated): does its flat-to-negative post-training
compressibility relate to how much of the input is task-relevant (2 of 32
dims for 2-XOR) rather than to capacity or any architectural feature?

Why pivot here now, rather than continuing to chase the Transformer's
depth-gradient mechanism: the last three attempts to explain a mechanism
on the Transformer (Experiments 7/10 rank correlation, Experiment 11
residual/LayerNorm) have each been limited by small sample size and
implementation-sensitivity (n=9–21, correlation signs flipping between
reruns) — that ceiling won't lift without either many more seeds or a
larger model, both bigger investments than are currently justified by
what's still an open, not-yet-sharpened question. The MLP input-layer
question, by contrast, is answerable cheaply and with much better
statistical power (the MLP infrastructure supports larger seed counts and
faster training than the Transformer), on a question that's been open the
longest and is more tractable to test directly: sweep the fraction of
task-irrelevant ("noise") input dimensions independently of network width,
holding the informative-dimension count fixed, and see whether the input
layer's compressibility tracks that ratio.

**Concretely:** for the 2-XOR task, vary `n_features` (e.g. 4, 8, 16, 32,
64 — informative dims fixed at 2) and measure the input layer's
best-ratio-at-threshold (Experiment 4/6's methodology) for both random-init
and trained weights. If the input layer's compressibility (or its
training-induced change) tracks the noise-dimension fraction, that
explains its distinct behavior as an information-preservation constraint
specific to being the first layer to see raw features, rather than a
capacity effect. If it doesn't track, the mystery deepens further and a
different hypothesis is needed.

**Standing recommendation, unchanged:** a real pretrained checkpoint
remains the single most valuable possible upgrade to this whole research
line, if network policy allows it at some point.
