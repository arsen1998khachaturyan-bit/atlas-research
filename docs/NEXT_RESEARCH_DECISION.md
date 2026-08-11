# Next Research Decision

Updated after the behavior-budgeted compression sweep (Experiment 4).
Covers Track B (`atlas_nn`) only — Track A (the pre-existing symbolic
active-learning framework) is unaffected and out of scope for this
document.

## 1. What we learned

**Stage A (synthetic matrices):** a purpose-built method (block dictionary
+ affine transform + residual) finds and exploits structure standard
baselines miss (affine-transformed shared blocks), and correctly does not
fake compression on random data. It loses to SVD on low-rank data and to
plain zlib on exact block repetition — narrow, verified, not general.

**Stage B (trained vs. random-init MLP):** training a small MLP on a
synthetic XOR task makes network *behavior* far more robust to weight
compression error than the raw tensor error predicts (3–11× smaller output
error than weight error, vs. ~1:1 for random-init, across every method).

**Experiment 4 (this round) — turning that into an actionable budget:**
at a fixed 5% behavioral-error bar, the best achievable compression ratio
is **1.5–2× higher post-training** on the two deeper layers (hidden→hidden:
5.32× → 8.00–10.61×; hidden→output: 4.92× → 7.11–9.14×), reproduced across
3 seeds. The standout case is SVD on the hidden layer: near-full rank
needed pre-training (ratio 0.50) vs. rank 4 sufficient post-training (ratio
8.00) — a direct, measured 16× swing caused by training alone. **The input
layer showed zero improvement** — same ratio, same winning method, in both
states, in all 3 seeds. So the effect is real, multiseed-verified, and
specifically depth-dependent, not a uniform property of "trained weights."

## 2. What failed

- Block-dictionary clustering does not help on globally low-rank or
  low-rank-plus-noise synthetic data (Stage A) — SVD is the right tool
  there.
- The structural method is strictly worse than zlib on exact block
  repetition (Stage A) — a fixed-width-encoding limitation.
- `vector_codebook` never met the 5% behavioral-error bar in Experiment 4,
  in either model state, at any tested `k` — consistently too lossy for
  this task, training or no training.
- The input layer (layer 0) shows no post-training compressibility gain at
  all — whatever training changes, it doesn't touch this layer's
  compressibility budget in this experiment.
- `magnitude_prune`'s sparse-COO encoding has enough per-nonzero overhead
  that even sparsity levels which now meet the quality bar post-training
  still produce ratio <1 (net expansion) at this matrix scale — the
  underlying finding (more sparsity survives post-training) is real, but
  the current encoding can't cash it in as an actual size win yet.

## 3. What worked

- Byte-honest accounting caught two real problems immediately: zlib beating
  Atlas on Stage-A repeats, and encoding overhead making "compression"
  net-expand tiny layers and low-sparsity pruning.
- Measuring behavioral error, not just tensor error, was what actually
  produced every interesting Stage B / Experiment 4 finding — tensor error
  alone (Stage A style) would have shown little or nothing about the
  trained/random distinction.
- Converting the Stage B *observation* (behavior is more robust
  post-training) into a budget *search* (how much further can ratio go at
  matched quality) turned a descriptive finding into a concrete, comparable
  number (1.5–2×), which is a much stronger form of evidence and directly
  actionable for anyone deciding how aggressively to compress a trained
  layer.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, more concretely now.** The mission's central question — can trained
weights be represented with substantially less independent information
than dense storage, while preserving behavior — now has a real, positive,
multiseed-verified, quantified answer for this (small, synthetic-task)
model: yes, by roughly 1.5–2× beyond what the same architecture supports
before training, concentrated in specific layers (not uniformly). This is
still far from the mission's "major research result" bar (dramatic storage
*and* memory reduction *and* direct inference without materializing full
weights) — it is squarely in "interesting, narrow, and honestly scoped"
territory, exactly where Stage A/B work should be at this point in the
ladder.

**What's still missing before scaling up (mission's explicit gate before
Stage C):** confirmation that this isn't an artifact of one tiny synthetic
task and one architecture — i.e., does the "deeper layers gain more
post-training compressibility, first layer gains none" pattern hold on a
different/harder task, or a bigger network? That's a cheap, same-scale
check, and the mission explicitly says not to scale to real pretrained
models until a method shows a meaningful, *robust* advantage at the current
stage.

## 5. The single most informative next experiment

**Robustness check of the depth-dependent pattern: repeat Experiment 4 on
a harder synthetic task and/or a deeper network (e.g. 4–5 hidden layers
instead of 2), and check whether (a) the ~1.5–2× post-training compression
gain reproduces, and (b) the "first layer doesn't benefit" pattern holds,
or whether it's actually "the *input-facing* layer never benefits" vs.
"the *first* layer never benefits" — a deeper network would distinguish
these two explanations.**

Why this one, specifically:
- Everything found so far rests on one 3-layer MLP and one XOR task. Before
  treating "deeper layers compress more after training" as a property of
  training in general (rather than of this one setup), the mission's
  falsification discipline (section 11) calls for testing it on a second,
  different setup at the same cheap scale — not for jumping straight to
  Stage C on a single data point.
- It's still CPU-cheap and reuses 100% of existing infrastructure
  (`atlas_nn.stage_b.budget_search`, `atlas_nn.stage_b.model` just needs a
  configurable depth) — no new dependencies or scale-up risk.
- If the pattern *does* reproduce, that's a much stronger basis for the
  Stage C investment (a real pretrained transformer, real dataset) than a
  single-task result would be; if it does *not* reproduce, that's equally
  valuable — it would mean the effect is task/architecture-specific and
  redirect effort toward characterizing *when* it appears rather than
  assuming it generalizes.

No architectural blockers remain — this reuses the existing `stage_b`
optional dependency (`torch`, CPU) and infrastructure; only
`atlas_nn/stage_b/model.py::build_mlp` needs a `hidden_layers` parameter
and a second dataset generator (e.g. a harder nonlinear task) needs to be
added.
