# Next Research Decision

Updated after Stage B (trained-vs-random-init MLP experiments). Covers
Track B (`atlas_nn`) only — Track A (the pre-existing symbolic active-learning
framework) is unaffected and out of scope for this document.

## 1. What we learned

**Stage A (synthetic matrices):**
- A real gap exists between "structure a standard baseline already handles"
  and "structure nothing standard handles." Five of six Stage-A synthetic
  matrix types are dominated by an obvious baseline (SVD for low-rank, zlib
  for exact repetition, plain quantization as the lossy floor for pure
  noise). Only the affine-per-block-transform case had no good baseline —
  and the first Atlas structural method (block dictionary + affine
  transform + residual) does find and exploit exactly that structure,
  reconstructing it exactly where every baseline fails.
- The same method correctly does *not* fake compression on random data —
  now a standing automated test.

**Stage B (trained vs. random-init MLP), new this round:**
- Training a small MLP on a synthetic XOR task (not linearly separable, so
  training is necessary) produces a large, consistent, multiseed effect:
  **the trained network's output is far less sensitive to weight-level
  compression error than the raw tensor error would predict** (3–11×
  smaller relative output error than relative weight error, across every
  method tested), while the random-init network shows weight error and
  output error tracking almost exactly 1:1. This is real evidence that
  *training* — not just architecture — creates additional exploitable
  redundancy, which is the mission's central Stage B question.
- One method (SVD) also showed a genuine tensor-level compressibility gain
  on the 64×64 hidden layer (relative reconstruction error roughly halved,
  0.89 → 0.41, at the same rank-4 budget, post-training).
- The Atlas structural method's own tensor-level error was flat between
  random-init and trained — it isn't (yet) sensitive to whatever training
  changed at the tensor level, even though the *behavioral consequence* of
  its error dropped sharply post-training, same as every other method.
- Plain accuracy was too coarse a metric on this task (noisy ±0–5 points,
  200–300-sample eval set) to see the trained/random distinction;
  `relative_logit_error` was the metric that actually revealed it. Worth
  carrying forward as the primary behavioral metric in Stage C.
- Small layers (the 2×64 output layer, 512 bytes) revealed a real,
  unglamorous failure mode: method overhead (a codebook or dictionary) can
  exceed the tensor's own size, making "compression" a net expansion.
  Structural methods need an overhead check before being applied
  per-layer.

## 2. What failed

- Block-dictionary clustering does not help on globally low-rank structure
  or low-rank-plus-noise synthetic data — SVD is the right tool there.
- k-means dictionary fitting at small dictionary sizes was unreliable in
  1/3 seeds in Stage A — a local-optimum failure mode worth fixing before
  trusting small dictionaries.
- The current structural method is strictly worse than doing nothing
  (zlib) on exact block repetition in Stage A, despite reconstructing
  exactly — a format/encoding problem (fixed bit-width, no entropy coding).
- In Stage B, the Atlas structural method did not show the "training helps"
  tensor-level signal that SVD showed — it may simply not be the right lens
  for whatever structure this particular training run added.

## 3. What worked

- Byte-honest accounting (every storage component summed explicitly)
  surfaced real problems immediately: zlib beating Atlas on Stage-A
  repeats, and method overhead exceeding tensor size on Stage-B's small
  layer. Neither would have been visible from a ratio number reported in
  isolation.
- Measuring *both* tensor error and behavioral/output error, as the mission
  requires (section 3), was what actually produced Stage B's main finding —
  tensor error alone would have shown almost nothing (flat or mixed
  results across methods); the tensor-vs-behavior *gap* was the real
  signal.
- The falsification test from Stage A (structural method vs. random matrix)
  continues to pass and is now joined by Stage B's own regression tests
  (training genuinely improves eval accuracy, weight substitution correctly
  restores the model afterward, etc.) — all in the standing `pytest` suite.

## 4. Does the evidence currently support the Atlas hypothesis?

**More than before, and now with real (if narrow) evidence from an actual
trained network, not just synthetic matrices.** Stage A showed a purpose-
built method can find structure standard baselines miss, on data
constructed to contain it. Stage B now shows that *training a real network*
does create something exploitable — most clearly, a large gap between how
much a weight perturbation changes the tensor versus how much it changes
the network's actual output. That gap is exactly the kind of "hidden
structure" the mission's core question asks about, measured on a real
trained model rather than assumed.

**What Stage B does not yet show:** that this translates into a *specific,
better Atlas storage representation* — the structural method's own
compression numbers didn't improve post-training, only its *downstream
behavioral cost* did (which is still valuable — it means a lossier
representation could be tolerated post-training — but it's a different
claim than "the tensor itself is more compressible with this method").
Nor does it show anything yet about a real dataset, a larger network, or
whether this behavioral-robustness effect holds at a scale where it would
matter practically (mission Stage C: pretrained transformers, real tasks).

## 5. The single most informative next experiment

**Directly exploit the tensor-vs-behavior gap found in Stage B: re-run the
Stage-A method panel on the trained-model weights, but choose each method's
compression *budget* using the behavioral (relative_logit_error) tolerance
instead of a fixed bit-width, and see how much further the ratio can go
before behavioral quality actually degrades.** Concretely: for the trained
64×64 layer, find the most aggressive rank/bits/dict_size for each method
that keeps `relative_logit_error` under a small threshold (e.g. 0.05), and
compare the resulting compression ratios directly against Stage A's
fixed-budget numbers.

Why this one, specifically:
- It turns Stage B's main finding (tensor error and behavioral error
  diverge after training) into an actionable optimization rather than
  leaving it as an observation — this is the natural next step implied by
  the mission's "storage × memory × speed × quality" tradeoff framing
  (section 8), and it's cheap (reuses all existing Stage B infrastructure,
  just sweeps a parameter instead of adding new code).
- It's a better use of the next research cycle than immediately jumping to
  Stage C (pretrained transformers): the mission explicitly says not to
  scale up until a method demonstrates a meaningful advantage at the
  current stage, and right now the "advantage" (behavioral robustness) is
  measured but not yet converted into an actual bigger compression ratio at
  matched quality — that conversion is the missing piece, and it's testable
  in minutes on CPU with what's already built.
- A secondary, smaller follow-up worth doing alongside it: verify the
  tensor-vs-behavior gap isn't an artifact of this specific tiny/easy XOR
  task (e.g. re-run Stage B with a harder synthetic task, or a
  higher-dimensional hidden layer) before trusting it as a general
  property of "training," rather than a property of this one experiment.

No architectural blockers remain for this next step — `torch` is now an
installed, declared optional dependency (`pip install -e ".[stage_b]"`);
Stage C (pretrained transformers) would still need a real dataset/checkpoint
decision, which is deliberately deferred until the parameter-sweep
experiment above justifies scaling up.
