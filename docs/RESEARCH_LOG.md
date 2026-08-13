# Research Log — Track B (`atlas_nn`)

Full per-row data:
- Stage A: `results/atlas_nn_stage_a.json` (180 rows = 6 synthetic matrices × 10 methods × 3 seeds `(101, 202, 303)`, shape 64×64).
- Stage B: `results/atlas_nn_stage_b.json` (126 rows = 3 Linear layers × 2 model states (random-init / trained) × 7 methods × 3 seeds `(11, 22, 33)`).

Every row carries git commit + timestamp; every JSON is reproducible with
the corresponding `experiments/run_atlas_nn_stage_*.py` script.

---

## Experiment 1 — Baseline sweep on controlled synthetic matrices

**Hypothesis.** Standard baselines (quantization, low-rank/SVD, magnitude
pruning, vector-codebook VQ, lossless zlib) will each do well on the type of
structure they're suited for and poorly elsewhere, giving a real reference
frame before evaluating anything called "Atlas."

**Method.** `experiments/run_atlas_nn_stage_a.py`. Six matrix types
(`random_gaussian`, `low_rank(rank=4)`, `block_repeated(4 unique 8×8
blocks)`, `block_transformed(4 unique blocks, random per-tile scale/sign/
shift)`, `structured_plus_noise(rank=4, noise_std=0.1)`,
`hierarchical_blocks`), each 64×64 float32, seeds 101/202/303. Baselines:
8-bit/4-bit uniform quantization, rank-4/rank-8 SVD, 50%/90% magnitude
pruning, vector codebook (len-8 vectors, k=16), zlib. All byte costs are
literal component sums (`component_bytes`), not estimates.

**Result (mean over 3 seeds; ratio = original/compressed bytes, rel_l2 =
relative L2 reconstruction error):**

| matrix | best baseline (ratio @ rel_l2) | why |
|---|---|---|
| random_gaussian | zlib 1.08 @ 0.0 (lossless, no compression) OR quantize_4bit 6.4 @ 0.088 | noise is incompressible; only lossy quantization buys ratio, at a real cost |
| low_rank | svd_rank4 **8.00 @ 0.0000** | exact — matrix genuinely has rank 4 |
| block_repeated | zlib **10.15 @ 0.0000** | exact byte-level repeats are DEFLATE's best case |
| block_transformed | *none* clear — best lossy is quantize_8bit 3.99 @ 0.0095; zlib only 1.08 (no exact byte repeats since every tile has a different continuous scale/shift) |
| structured_plus_noise | svd_rank4 **8.00 @ 0.046** | still dominated by the rank-4 signal |
| hierarchical_blocks | zlib **14.15 @ 0.0000** | same reason as block_repeated, more redundancy |

**Interpretation.** Every synthetic category has an obvious, strong,
"boring" baseline that already handles it well — except `block_transformed`,
where structure exists (four shared prototypes under per-tile affine
transforms) but none of the standard baselines can see it, because it's
neither exactly low-rank, exactly repeated, nor bytewise-identical. That's
exactly the gap the mission's "shared prototype + transformation + residual"
idea (section 5) targets, and exactly the condition under which a
purpose-built structural method would be worth having.

**Next experiment.** Implement the block-dictionary + affine-transform +
residual method and test it against this same baseline table on the same
six matrices.

---

## Experiment 2 — First Atlas structural method: block dictionary + affine transform + residual

**Hypothesis.** A method that (a) clusters blocks into a small dictionary,
(b) fits a per-block scalar affine map `a·prototype + b` (trying both signs
of the prototype), and (c) stores a quantized residual, will reconstruct
`block_transformed` and other block-structured matrices far better than any
Stage-1 baseline at a comparable bit budget, while doing no better than
those baselines (and no better than pure noise floor) on `random_gaussian`.

**Method.** `atlas_nn.structural.block_dictionary_transform`, block_shape
(8,8), dict_size ∈ {8, 16}, residual_bits=4, same seeds/matrices as
Experiment 1.

**Result (mean over 3 seeds, dict_size=16, residual_bits=4 ⇒ fixed ratio
2.27 for every matrix by construction — the format doesn't adapt its size to
content):**

| matrix | rel_l2 | reads as |
|---|---|---|
| random_gaussian | 0.0721 | no structure found (expected — see falsification note below) |
| low_rank | 0.0706 | no better than random; block-dict doesn't see global low rank |
| block_repeated | **0.0000** | exact |
| block_transformed | **0.0000** | exact — this is the targeted case |
| structured_plus_noise | 0.0709 | no better than random; same reason as low_rank |
| hierarchical_blocks | **0.0000** | exact |

**Where it wins vs. the Experiment-1 baseline table:**
- `block_transformed`: Atlas gets ratio 2.27 at **0.0000** error; the best
  baseline (8-bit quantization) got ratio 3.99 but at 0.0095 error, and zlib
  (which would be "free" exactness) only reached ratio 1.08. At matched
  near-zero error, Atlas is the only method that compresses this matrix at
  all. This clears the mission's "Interesting" bar (section 15: "2× smaller
  than a strong comparable baseline at similar behavioral quality") relative
  to zlib, the only other near-zero-error method on this matrix.

**Where it loses, badly:**
- `block_repeated` / `hierarchical_blocks`: zlib reaches ratio 10–14 at
  **the same** zero error, because these matrices have byte-identical
  repeated blocks and Atlas's fixed per-block bit budget (dictionary index +
  sign + a + b + quantized residual) doesn't shrink even when the residual
  is exactly zero — it never goes below a hard floor of ~2.27–3.18×. Plain
  lossless compression is strictly better here. This is a real limitation of
  the current encoding, not of the underlying idea: the fix is to add
  entropy coding on top (e.g., only pay for a residual when it's non-zero,
  Huffman/range-code the indices) rather than a fixed-width format.
- `low_rank` / `structured_plus_noise`: SVD wins by a wide margin (ratio 8
  at ≤0.046 error vs. Atlas's ratio 2.27 at 0.07). Block-local dictionary
  matching cannot see a low-rank pattern that spans the whole matrix. Not a
  bug — a scope mismatch between method and structure type, exactly why the
  mission asks for baselines across *all* structure families before any
  claim.

**Falsification check (mission section 11.7, encoded as
`tests/test_atlas_nn_structural.py::test_structural_method_does_not_fake_compression_on_random_matrix`).**
At the identical bit budget, `random_gaussian` gives rel_l2 = 0.072, more
than 5× the error on `block_transformed` (0.0000) and far above the 0.03
threshold the test enforces. The method does not hallucinate structure on
pure noise. This test is part of the regular suite (`pytest`), not a one-off
manual check.

**Novelty discipline (mission section 14).** This method is best described
as **vector quantization / clustering (k-means codebook) generalized with a
per-cluster affine correction**, closely related to product quantization and
to "Deep Compression"-style weight-clustering codebooks; the residual
quantization is standard scalar quantization. Nothing here is claimed as
novel — it's an assembly of known techniques, evaluated honestly, per
mission section 14's requirement to identify the known family before any
novelty discussion.

**Interpretation.** The core mechanism works exactly as designed: it finds
and exploits affine block structure that literally nothing else in the
baseline set can see, and it correctly fails to find structure that isn't
there. But its current fixed-width encoding is dominated by trivial lossless
compression whenever the structure is exact repetition rather than
continuous affine variation, and it's irrelevant for global low-rank
structure. So: real, narrow, verified win — not yet a general-purpose
result.

**Next experiment.** Move to Stage B: does *training* a real network
produce structure that compression methods (baselines or Atlas) can
exploit, compared to the same architecture at random initialization? This
is the mission's actual central question — Stage A was scaffolding for it.

---

## Experiment 3 — Stage B: does training create compressible structure?

**Hypothesis.** Weight tensors from a *trained* small MLP will be more
compressible (lower reconstruction error at matched bit budget, and/or less
behaviorally sensitive to compression error) than the same tensors at random
initialization — because training is expected to concentrate the useful
signal into a lower-effective-dimensional subspace of the weight space.

**Method.** `experiments/run_atlas_nn_stage_b.py`. 3-layer MLP
(32→64→64→2, `atlas_nn.stage_b.model.build_mlp`), trained on a synthetic
XOR-of-two-coordinates task embedded in 32-dim Gaussian noise
(`atlas_nn.stage_b.dataset.make_xor_dataset` — not linearly separable, so
training is required to solve it; reaches 100% train accuracy, 87–93% held-
out accuracy across seeds 11/22/33 in 500 epochs of Adam). For each of the
3 Linear layers' weight matrices, in both the **random-init** state
(snapshotted before training) and the **trained** state (same weights,
after training), every Stage-A method is applied and the reconstructed
weight is substituted back into the live model to measure **both** tensor
error (relative L2) **and** behavioral error (accuracy drop,
relative L2 error of output logits vs. the uncompressed model) on a held-out
eval set — directly implementing the mission's requirement to measure
behavioral preservation, not just reconstruction MSE (section 3).

**Result 1 — tensor-level compressibility (mean over 3 seeds).** The
clearest signal is on the 64×64 hidden→hidden layer:

| method | random-init rel_l2 | trained rel_l2 |
|---|---|---|
| svd_rank4 | 0.891 | **0.414** |
| vector_codebook (k=16) | 0.722 | 0.583 |
| atlas_block_dict16_res4bit | 0.063 | 0.067 |
| quantize_4bit | 0.063 | 0.071 |

SVD is the one method that shows real tensor-level improvement from
training (relative error roughly halves at the same rank-4 budget) — training
visibly pushes this layer's weight matrix toward lower effective rank, a
known deep-learning phenomenon (small-task/low-intrinsic-dimensionality
training concentrating weight matrices onto a low-rank subspace), reproduced
here as a real, multiseed, measured effect rather than assumed from theory.
Quantization and the block-dictionary method show **no** improvement from
training (error is essentially unchanged or slightly worse) — those methods
aren't sensitive to the kind of structure training adds here.

**Result 2 — behavioral robustness (the stronger, more consistent signal).**
Comparing relative L2 error of the *tensor* vs. relative L2 error of the
*output logits* after substitution, at the 64×64 layer:

| method | random-init: rel_l2 → rel_logit | trained: rel_l2 → rel_logit |
|---|---|---|
| svd_rank4 | 0.891 → 0.928 (≈1:1) | 0.414 → **0.039** (11× smaller) |
| atlas_block_dict16_res4bit | 0.063 → 0.061 (≈1:1) | 0.067 → **0.015** (4.5× smaller) |
| quantize_4bit | 0.063 → 0.069 (≈1:1) | 0.071 → **0.017** (4× smaller) |
| prune_50pct | 0.354 → 0.356 (≈1:1) | 0.275 → **0.084** (3.3× smaller) |

For the **random-init** network, weight error and output error track each
other almost exactly (ratio ≈ 1) for every method — expected, since an
untrained network has no learned redundancy to absorb a perturbation. For
the **trained** network, every single method shows output error dropping to
a fraction of the weight error — the same absolute amount of weight
corruption does far less damage to what the network actually computes.
Accuracy drop (the coarser metric) stayed small and noisy for both states
(±0–5 points on a 200–300-sample eval set) and did **not** show this pattern
clearly — `relative_logit_error` was the metric that actually revealed the
trained/random distinction, which is itself a useful methodological
finding: accuracy alone was not sensitive enough on this task.

**Result 3 — small-layer overhead can make "compression" expand the
tensor.** The output layer (2×64, 512 bytes) is small enough that method
overhead (a k=16 codebook alone is 16×8×4 = 512 bytes; the Atlas
dictionary is 16×64×4 = 4096 bytes) exceeds the tensor itself:
`atlas_block_dict16_res4bit` gives ratio **0.21** (the "compressed" form is
~5× *larger* than the original tensor) on this layer in both states; SVD,
vector-codebook, and zlib all hover at ratio ≈0.97–0.98 (net-neutral to
slightly expanding). Only plain quantization (no learned overhead) reliably
compresses this layer (ratio 3.76–6.4). Real, unglamorous, worth keeping in
mind for Stage C: per-layer structural methods should not be applied to
small layers without checking the overhead first.

**Interpretation.** This is genuine, if modest, positive evidence for the
mission's central Stage B question: **training does create additional
structure**, most clearly visible as increased *behavioral* robustness to
weight-level compression error (a large, consistent, multiseed effect across
every method tested) and, for at least one layer/method combination
(SVD on the 64×64 layer), as directly measurable tensor-level
compressibility. It does not yet show that any *specific* Atlas method
becomes dramatically more effective post-training — the block-dictionary
method's own reconstruction error was essentially flat between states, even
though the *behavioral consequence* of that same error dropped sharply. That
gap (tensor error unchanged, functional error much smaller) is itself the
finding, and is consistent with "trained networks live in flatter/more
redundant regions of weight space" rather than "trained weight tensors
literally contain more of the specific shared-prototype structure Atlas
section 5 targets."

**Next experiment.** Turn the tensor-vs-behavior gap into an actionable
compression-budget search: instead of a fixed bit-width, find the most
aggressive setting of each method that still keeps behavioral error under a
threshold, and see whether the achievable ratio is actually higher
post-training.

---

## Experiment 4 — Behavior-budgeted compression: how much further does the ratio go post-training?

**Hypothesis.** If trained-network behavior really is more robust to weight
error (Experiment 3), then at a *fixed behavioral quality bar* (rather than
a fixed bit-width), the best achievable compression ratio should be higher
for trained weights than for the same layer at random initialization.

**Method.** `experiments/run_atlas_nn_stage_b_budget_search.py` /
`atlas_nn.stage_b.budget_search`. For every layer × model state × seed
(11/22/33), each method family (quantization, SVD, magnitude pruning,
vector-codebook, Atlas block-dictionary) is swept over its own parameter
grid; for each family, the highest compression ratio among configurations
with `relative_logit_error ≤ 0.05` is kept (all raw sweep rows saved too).
Results: `results/atlas_nn_stage_b_budget_search.json` (18 layer×state×seed
searches × ~31 configs each = 558 measured rows).

**Result (best ratio meeting the 5% behavioral-error bar, mean over 3
seeds):**

| layer | shape | random-init best ratio | trained best ratio | change |
|---|---|---|---|---|
| 0 (input→hidden) | 64×32 | 5.31 (quantize_6bit) | 5.31 (quantize_6bit) | **none** |
| 2 (hidden→hidden) | 64×64 | 5.32 (quantize_6bit) | 8.00–10.61 (svd_rank4 / quantize_3bit) | **1.5–2×** |
| 4 (hidden→output) | 2×64 | 4.92 (quantize_6bit) | 7.11–9.14 (quantize_3bit/4bit) | **1.4–1.9×** |

Per-family detail on layer 2 (mean ratio among configs meeting the
threshold, out of 3/3 seeds meeting it unless noted):

| family | random-init | trained |
|---|---|---|
| quantize | 5.32 | **9.73** |
| svd | 0.50 (needs near-full rank just to stay accurate — barely better than storing the matrix directly) | **8.00** (rank 4 suffices) |
| prune | never meets the bar at any tested sparsity ≥30% | **0.71** (now meets it, though still net-expanding at this encoding's overhead — see caveat) |
| vector_codebook | never meets the bar | never meets the bar |
| atlas_block_dict | 3.19 | **4.54** |

**Interpretation.** This directly confirms and quantifies Experiment 3's
finding as an *actionable* result, not just an observation: at matched
behavioral quality, the deeper two layers (2 and 4) tolerate substantially
more aggressive compression after training — roughly 1.5–2× the ratio,
consistently across 3 seeds. The clearest individual case is SVD on layer
2: before training, keeping behavioral error under 5% requires close to
full rank (ratio 0.50, i.e. no real compression); after training, rank 4
suffices (ratio 8.00) — a direct, measured demonstration that training
concentrates this layer's useful signal into a low-rank subspace, going
beyond Experiment 3's rel_l2 number (which only showed the *ratio* of
tensor-to-behavioral error) to show the *achievable compression* itself
increased.

**The layer-0 null result matters too.** The first (input-facing) layer
shows *zero* improvement — same best ratio (5.31) whether trained or not,
and the same winning method (6-bit quantization) in both states. This
layer must preserve enough information about the two informative input
coordinates (out of 32, mostly noise) to solve the task at all, and
training doesn't create slack there the way it does in later layers. This
is a real negative result, not a gap in the experiment, and narrows the
"training creates structure" claim usefully: it's not uniform across
depth.

**Caveat on the `prune` family's ratio <1 result.** A ratio below 1.0 means
the "compressed" form is larger than the original tensor — here, that's a
property of the sparse-COO encoding's fixed per-nonzero overhead (int32
index + float32 value, i.e. 2× the cost of a dense element), not of the
underlying pruning idea. The finding that *some* sparsity now clears the
quality bar post-training (vs. none before) is real; the encoding just
isn't efficient enough to turn that into an actual size win at this matrix
scale. Flagged rather than hidden, per mission section 11.

**Falsification angle.** `vector_codebook` never met the quality bar in
either state, at any tested `k` — a real, consistent failure, not
cherry-picked away. Not every method benefits from training's added
robustness; this one appears too lossy for this task/threshold regardless.

**Next experiment.** Check whether the depth-dependent post-training
compression pattern is a general property of training, or an artifact of
this one easy task/architecture, before investing further.

---

## Experiment 5 — Robustness check: does the pattern survive a harder task or a deeper network?

**Hypothesis.** Experiment 4's finding (post-training compression headroom
rises ~1.5–2× on deeper layers, not on the input layer) should reproduce
qualitatively on (a) the same architecture trained on a harder synthetic
task, and (b) a deeper architecture trained on the original task — if it's
a real property of training rather than an artifact of one easy setup.

**Method.** `experiments/run_atlas_nn_stage_b_robustness_check.py`, same
behavior-budgeted search as Experiment 4 (5% relative-logit-error bar),
3 seeds, two conditions:
- **harder_task_parity3**: identical 3-Linear-layer architecture, trained
  on 3-way parity (XOR of 3 coordinate signs, `n_features=8`) instead of
  2-XOR — a genuinely harder task (a plain 2-hidden-layer/64-unit MLP
  trained on 3-parity with 32 noise dims failed to generalize at all,
  ~55% held-out accuracy despite 100% train accuracy; reducing to 8
  features made it learnable, 94–97% held-out accuracy).
- **deeper_net_xor2**: 6-Linear-layer architecture (5 hidden layers, same
  64-unit width), trained on the original 2-XOR task.

**Result A (harder_task_parity3): the pattern does not reproduce.** Across
all 3 layers and 3 seeds, trained and random-init best-ratio-at-threshold
were **identical** in 7 of 9 layer×seed combinations (e.g. layer 0: 5.22
for both states, seeds 11 and 22). Where they differed, the direction was
inconsistent: seed 11/22 showed a modest gain on the output layer only
(4.92 → 7.11); seed 33 showed random-init **beating** trained on layers 0
and 2 (7.76/7.97 vs. 5.22/5.32) — the opposite of Experiment 4's direction.
There is no reliable post-training compression gain on this harder task, on
any layer, in this data.

**Result B (deeper_net_xor2): the comparison is invalid as designed, for a
different reason.** The raw numbers looked dramatic — random-init ratios of
up to 64.0×, often exceeding the trained network's ratio — but investigating
why revealed the random-init 5-hidden-layer network is **degenerate**:
accuracy ~50% (chance) and, critically, `unique_argmax=1` — it predicts the
*same class for every single evaluation example*, with output logit std
0.04–0.12 (vs. 22–34 once trained). This is a vanishing-signal collapse from
stacking 5 unnormalized ReLU layers with plain PyTorch default init, not a
compression result. `relative_logit_error` against an already-constant,
meaningless output is measuring "how much does the perturbation change a
degenerate function," which is uninformative — any large ratio computed
against this baseline should not be trusted. **This condition's numbers are
reported for transparency but explicitly excluded from any comparison
claim.** See the caution now documented in `atlas_nn.stage_b.model.build_mlp`'s
docstring.

**Interpretation.** This substantially revises confidence in Experiment 4's
finding. The most coherent explanation available: Experiment 4's original
2-XOR task is so easy relative to the network's capacity (100% train
accuracy reached quickly, wide margin) that training leaves a lot of unused
representational slack, which shows up as extra post-training compression
headroom. On the harder 3-parity task, the network has to use more of its
capacity to solve the problem correctly, leaving less slack — and the
compression-gain effect disappears. This is a **plausible hypothesis**, not
a verified one; distinguishing "training creates general compressibility"
from "an easy task leaves more unused capacity, which looks like
compressibility" is exactly the kind of confound the mission's
falsification discipline (section 11) exists to catch, and this experiment
caught it.

**Falsification value.** This is precisely why Experiment 4's claim was
scoped narrowly in `docs/BEST_RESULTS.md` ("one small architecture, one
synthetic XOR task... not yet checked... at other thresholds, a real
dataset, or larger networks") rather than generalized. That scoping is now
justified by direct evidence, not just caution.

**Next experiment.** Test the slack/capacity hypothesis directly: hold
depth fixed and sweep network width for both tasks, to see whether the
compression gain shrinks with less spare capacity and reappears with more.

---

## Experiment 6 — Capacity sweep: does the compression gain track spare capacity?

**Hypothesis.** If Experiment 4's gain reflects unused representational
capacity left after an easy task, rather than "training" as a general
effect, then (a) shrinking network width on the easy task should shrink or
remove the gain, and (b) growing network width on the *hard* task
(3-parity, which showed no gain at matched width in Experiment 5) should
bring the gain back.

**Method.** `experiments/run_atlas_nn_stage_b_capacity_sweep.py`. Same
behavior-budgeted search (5% relative-logit-error bar), depth fixed at 2
hidden layers (to avoid Experiment 5's depth-related init degeneracy),
width ∈ {16, 64, 256}, both tasks (2-XOR, 3-parity), 3 seeds each — 6
conditions × 3 layers × 2 states × 3 seeds. Results:
`results/atlas_nn_stage_b_capacity_sweep.json`.

**Result — the hidden→hidden layer (layer 2), best ratio at the 5% bar
(all 3 seeds shown):**

| condition | random-init | trained | reads as |
|---|---|---|---|
| xor2_h16 (undersized, easy) | 7.5, 5.1, 5.1 | 5.1, 5.1, 5.1 | **gain gone** |
| xor2_h64 (baseline, easy) | 5.3, 5.3, 5.3 | 10.6, 10.6, 8.0 | gain present (matches Experiment 4) |
| xor2_h256 (oversized, easy) | 5.3, 8.0, 5.3 | 8.0, **128.0**, 64.0 | confounded — see caveat below |
| parity3_h16 (undersized, hard) | 7.5, 7.5, 5.1 | 5.1, 3.9, 5.1 | **gain absent/negative** |
| parity3_h64 (baseline, hard) | 5.3, 5.3, 8.0 | 5.3, 5.3, 5.3 | gain absent (matches Experiment 5) |
| parity3_h256 (oversized, hard) | 5.3, 5.3, 5.3 | **10.0, 8.0, 16.0** | **gain reappears** |

**The key confirmation: `parity3_h256`.** All 3 seeds trained successfully
here (100% train accuracy, 93–97% held-out accuracy — no confound). At
matched width (h64), 3-parity showed no compression gain at all
(Experiment 5). Given enough spare capacity (h256), the gain reappears on
the exact same task, in the exact same layer, at a similar-or-larger
magnitude (1.9×–3.0×) to what the easy task showed at its matched width.
This is direct, multiseed, unconfounded support for the slack/capacity
hypothesis: **it's not that training on an easy task creates
compressibility — it's that training leaves unused capacity compressible,
and how much capacity is "unused" depends on task difficulty relative to
network size, not on task identity.**

**Consistent with the hypothesis: `xor2_h16` and `parity3_h16`.** Shrinking
the easy task's network to a tight fit (h16) removed its previously-solid
gain — trained and random-init converge to the same ratio, or trained is
even *worse* (e.g. layer 0: random 7.76 vs. trained 3.94). A tightly-fit
network has no slack regardless of how easy the task was; consistent with
the hypothesis, and a genuinely new, useful data point.

**Confound found in `xor2_h256`, reported rather than used.** 2 of 3 seeds
at this width **failed to train properly** with the same hyperparameters
used everywhere else (500 epochs, lr=2e-2): seed 22 reached 49% held-out
accuracy (chance) and seed 33 reached 68% (partial). Only seed 11 is a
valid comparison (87% held-out accuracy, in line with h64's typical
90–91%), and it shows a gain of similar magnitude to h64 (layer 2: 5.3→8.0),
not dramatically larger. The eye-catching 128× and 85× numbers from seeds
22/33 are an artifact of the same underlying pathology as Experiment 5's
degenerate deep network: compressing a network that never learned a
meaningful function is not evidence of anything. **These numbers are kept
in the results file for transparency but explicitly excluded from any
claim.** Practical lesson for future stages: fixed training hyperparameters
do not automatically transfer across network sizes — this needs to be
checked, not assumed, especially before Stage C.

**A second, unplanned finding: the input layer (layer 0) is consistently
flat-to-negative after training, across every condition.** Not just "no
gain" (Experiment 4's original observation) — in most conditions here,
trained best-ratio is equal to or *lower* than random-init's (e.g.
`parity3_h256` layer 0: random 7.94 in all 3 seeds, trained 5.31 in all 3
seeds — a consistent, reproducible *penalty*, not just an absence of gain).
This held regardless of task or width, suggesting it's a distinct,
consistent phenomenon from the capacity story above, not yet explained.
Flagged as an open question rather than investigated further this round.

**Interpretation.** The slack/capacity hypothesis now has real, targeted,
falsifiable, multiseed support — specifically confirmed by bringing the
compression-gain effect back on a task that had previously shown no gain,
purely by adding network capacity, holding everything else fixed. This is
stronger evidence than Experiment 4 alone provided, precisely because it
predicted and then found a *reversal*, not just a repeated observation.
The `xor2_h256` training-failure confound and the layer-0 penalty are both
honestly reported rather than smoothed over, per mission section 11.

**Next experiment.** Turn "spare capacity" into a number: compute a
spectral capacity-usage metric per layer and check whether it predicts the
*magnitude* of the compression gain already measured, not just its sign.

---

## Experiment 7 — Quantifying capacity: does effective rank predict the compression-gain magnitude?

**Hypothesis.** If Experiment 6's "spare capacity" story is right, a
layer's effective rank (a spectral proxy for how many independent
directions its weight matrix actually uses) should shrink more after
training exactly where the compression gain was larger — turning a
qualitative story into a quantitative, continuous relationship.

**Method.** `atlas_nn.stage_b.capacity_metrics` implements three spectral
capacity metrics (Shannon/Roy–Vetterli effective rank, stable rank,
energy-95% rank; unit-tested against rank-1, random, and increasing-rank
synthetic matrices). `experiments/analyze_stage_b_capacity_metric.py`
deterministically reproduces all 6 capacity-sweep conditions × 3 seeds
(same seeds as Experiment 6, so the same trained models), extracts each
Linear layer's weight matrix in both states, computes effective rank, and
correlates rank shrinkage against `log(compression_gain)` where
`compression_gain = trained_best_ratio / random_init_best_ratio` (from
Experiment 6's results). `experiments/summarize_stage_b_capacity_metric.py`
post-processes the result with a stricter training-success filter (see
caveat below) without re-running any training.

**Result — a real effect, but layer-specific, not uniform.** Pooling all
3 layers together gives a weak, unreliable signal (Pearson r=0.47, but
Spearman r≈0.00 — the two disagree because layers have very different
absolute rank scales, so pooling mixes populations rather than measuring
one relationship). Breaking it down by layer is what actually shows
something (n=16 per layer after the stricter filter, correlating absolute
effective-rank shrinkage — random-init effective rank minus trained
effective rank — against log compression-gain):

| layer | Pearson r | Spearman r | reads as |
|---|---|---|---|
| 0 (input) | 0.13 | 0.37 | weak, inconsistent — matches Experiment 6's separate finding that layer 0 follows a different, unexplained pattern |
| 2 (hidden→hidden) | **0.67** | **0.62** | moderate-strong, positive — the layer where Experiment 6's effect was found and confirmed is exactly the layer where the rank-shrinkage metric predicts it |
| 4 (output, 2 units) | −0.56 (abs. shrink) / 0.61 (trained rank) | −0.51 | unstable and sign-flips between related metrics — `max_rank` is only 2 for this layer, so "effective rank" is nearly a binary 1-vs-2 variable and this correlation shouldn't be trusted as a real test of the mechanism |

**A methodological catch, found and fixed during this analysis.**
`analyze_stage_b_capacity_metric.py`'s original filter for "did this run's
training succeed" (trained accuracy at least 15 points above its own
random-init accuracy) let `xor2_h256` seed 33 through as "succeeded" —
but that run only reached 68% held-out accuracy, well below the ~87%
ceiling for that condition (the same partial-training artifact class
Experiment 6 already flagged for seed 22 at the same width, just less
severe). A relative-gain-only filter isn't sufficient to catch a partial
failure when the random-init baseline is already near chance. Fixed with
an added per-condition "near ceiling" check
(`experiments/summarize_stage_b_capacity_metric.py`) rather than silently
leaving the weaker filter in place; the numbers above are from the
corrected analysis. This is the second time in two experiments that a
generic accuracy check needed strengthening after specifically looking for
it — worth treating as a standing risk for any future run, not a one-off.

**Interpretation.** The effective-rank metric is a real, if partial,
quantitative confirmation of the capacity-slack mechanism — specifically
on the layer where the mechanism was hypothesized to operate (hidden→hidden,
where weight structure has room to vary continuously), with a correlation
strong enough to be useful (r≈0.6–0.7, not just directionally positive).
It does **not** generalize cleanly to the input layer (already known to be
governed by something else) or the tiny output layer (not a fair test —
degenerate rank range). This is a more honest and more useful outcome than
either "the metric perfectly explains everything" or "the metric shows
nothing" would have been: it identifies specifically where the proposed
mechanism is well-described by rank and where it manifestly isn't.

**Next experiment.** Move beyond the MLP/XOR setting: does the
behavioral-robustness effect transfer to a different architecture (attention,
not just feedforward) on a real task (real text, not synthetic feature
vectors)?

---

## Experiment 8 — Stage C-lite: does the behavioral-robustness effect transfer to a Transformer on a real task?

**Context.** Mission Stage C calls for "manageable open pretrained models."
This session's network policy blocks `huggingface.co` (403, confirmed via
the egress proxy status endpoint: `"gateway answered 403 to CONNECT
(policy denial or upstream failure)"`), so a literal pretrained-checkpoint
Stage C is not reachable here. Per the user's choice, this experiment
substitutes a genuinely different architecture (a small Transformer, not
another MLP) trained from scratch on a real task (self-authored English
sentiment sentences, not synthetic feature vectors) — see
`atlas_nn/stage_c_lite/`.

**Hypothesis.** If the Stage B behavioral-robustness finding (Experiment 3:
trained-network output is far less sensitive to weight-compression error
than tensor error predicts) reflects something general about trained
networks rather than an MLP/XOR-specific artifact, it should reproduce on
this different architecture and task too.

**Method.** `experiments/run_atlas_nn_stage_c_lite_smoke.py`. A 2-block
Transformer classifier (`atlas_nn.stage_c_lite.model`, d_model=64,
4 heads, from-scratch init, plain `torch.nn` — no pretrained weights or
external model download involved) trained on a self-authored,
template-generated English sentiment dataset (200 sentences, 150 train /
50 held-out, ~94-word vocabulary; not linearly trivial — random-init
held-out accuracy 56–64%, trained accuracy 84–94% across 3 seeds). The
Stage-A baseline panel + Atlas structural method, at the same fixed
parameters used in Stage B's original smoke test, applied to all 7 Linear
layers (attention output projections and FFN layers in both Transformer
blocks, plus the classification head), in both random-init and trained
states.

**Result — tensor error barely changes with training; behavioral error
collapses, and does so far more in the later block than the earlier one.**
Mean over 3 seeds, `svd_rank4` (the clearest case, consistent with Stage B):

| layer | tensor rel_l2 (random → trained) | behavioral rel_logit (random → trained) | gain |
|---|---|---|---|
| block 0 attn out_proj | 0.894 → 0.821 (flat) | 0.261 → 0.053 | 4.9× |
| block 0 linear1 (FFN) | 0.917 → 0.896 (flat) | 0.229 → 0.065 | 3.5× |
| block 0 linear2 (FFN) | 0.916 → 0.888 (flat) | 0.217 → 0.046 | 4.7× |
| block 1 attn out_proj | 0.894 → 0.834 (flat) | 0.204 → 0.009 | **22.7×** |
| block 1 linear1 (FFN) | 0.917 → 0.888 (flat) | 0.188 → 0.008 | **23.5×** |
| block 1 linear2 (FFN) | 0.916 → 0.892 (flat) | 0.204 → 0.013 | **15.7×** |

**This depth pattern held in every one of the 3 seeds individually, not
just in the mean** (per-seed gain multipliers: block 0 ranged 1.9×–12.9×,
block 1 ranged 10.3×–42.0× — the two ranges do not overlap in 2 of 3 seeds,
and even where they're close, block 1 is always higher within the same
seed). This is a materially stronger, more graded version of the MLP
finding: instead of a binary "hidden layer gains, input layer doesn't," a
real 2-block Transformer shows the gain *increasing with depth across two
full blocks*, on a task and architecture with no relationship to the
synthetic XOR/parity setup Experiments 3–7 were built on.

**Same encoding-overhead and tiny-output-layer caveats reproduce too.**
The classification head (2×64, 512 bytes) shows the identical pathology
found in Stage B: `atlas_block_dict16_res4bit` gives ratio 0.21 (net
expansion), `svd_rank4`/`vector_codebook`/`zlib` all hover at ratio
≈0.97–0.98 — overhead exceeding a small tensor's own size is not
architecture-specific, it recurs exactly as before.

**Interpretation.** This is the strongest transfer evidence in the project
so far: the core Stage B finding (trained networks are behaviorally far
more robust to weight-compression error than tensor error predicts, and
this scales with depth) reproduces, per-seed, on an attention-based
architecture trained on real (if simple) English text — not a variation on
the same MLP/synthetic-task setup everything else was built on. It does
not by itself prove this generalizes to large-scale pretrained models
(mission's literal Stage C, still blocked by network policy here), but it
substantially raises confidence that the mechanism is architecture-general
rather than an MLP-specific curiosity.

**What this experiment did not test.** Unlike Experiment 4/6, this used
fixed compression parameters (mirroring Stage B's Experiment 3), not a
behavior-budgeted search for the maximum achievable ratio at matched
quality — so there is no direct "Nx compression ratio" headline number
here, only the tensor-vs-behavior error gap. Running the budget-search
machinery (already architecture-agnostic after this experiment's
refactor) on this model is a natural, low-cost follow-up.

**Next experiment.** Turn Experiment 8's qualitative tensor-vs-behavior gap
into the same quantitative "achievable ratio at matched quality" number
Experiments 4/6 produced for the MLP, and check whether it deepens with
Transformer block depth the way the raw robustness gain did.

---

## Experiment 9 — Stage C-lite budget search: does achievable compression ratio deepen with Transformer block depth?

**Hypothesis.** Given Experiment 8's finding that behavioral robustness to
weight-compression error grows sharply with depth (block 1 gains
15.7–22.7× vs. block 0's 3.5–4.9×, at fixed compression parameters), the
*achievable compression ratio* at a fixed 5% behavioral-error bar
(Experiment 4/6's methodology) should also be higher for block 1 layers
than block 0 layers, and higher for trained than random-init weights
throughout.

**Method.** `experiments/run_atlas_nn_stage_c_lite_budget_search.py`.
`atlas_nn.stage_b.budget_search.run_budget_search` (unchanged, reused
as-is per the Experiment 8 refactor) applied to all 7 Linear layers of the
Stage C-lite Transformer, 3 seeds, 5% relative-logit-error bar.

**Result — confirmed, with a real quantitative depth gradient (mean
best-ratio-at-threshold over 3 seeds):**

| layer | random-init ratio | trained ratio | gain |
|---|---|---|---|
| block 0 attn out_proj | 9.7 | 14.2 | 1.5× |
| block 0 linear1 (FFN) | 11.5 | 25.9 | 2.3× |
| block 0 linear2 (FFN) | 9.8 | 30.2 | 3.1× |
| block 1 attn out_proj | 10.6 | 32.0 | 3.0× |
| block 1 linear1 (FFN) | 14.2 | 42.7 | 3.0× |
| block 1 linear2 (FFN) | 9.8 | 42.7 | **4.4×** |
| classifier (head) | 4.9 | 10.4 | 2.1× |

Block 1's mean gain (≈3.5×) is clearly and consistently higher than block
0's (≈2.3×) across all three layer types (attention-output, FFN-in,
FFN-out) — the same depth-gradient direction Experiment 8 found via
behavioral error alone, now expressed as an actual achievable-compression-
ratio number: trained block-1 layers reach ~32–43× compression at the 5%
quality bar, more than 3× what the same layers support before training,
and more than what block 0 supports even after training (14–30×).

**Why the numeric gap is smaller than Experiment 8's raw robustness-gain
numbers (up to 22.7×).** `run_budget_search` only evaluates a fixed,
discrete parameter grid per method family (e.g. SVD ranks
1/2/4/8/16/32/64, not a continuous sweep), so the *achievable ratio* moves
in coarse jumps — a real error reduction that doesn't cross the next
discrete grid point doesn't show up as a higher ratio. The direction and
relative ordering between layers is unaffected by this, but the exact
multiplier is a coarser, conservative estimate of the underlying effect
size compared to Experiment 8's continuous relative-logit-error numbers.

**SVD becomes the dominant method for trained block-1 layers** (winning in
8 of 9 trained-block-1 cases across seeds, at or near the largest tested
rank) — consistent with Experiment 7's MLP finding that SVD is the method
most sensitive to whatever training changes about a layer's effective
rank, now observed on a second architecture.

**Interpretation.** This closes the gap Experiment 8 flagged: the
Transformer's behavioral-robustness effect is not just a tensor-vs-
behavior-error curiosity, it translates into materially more achievable
compression at matched quality, and the amount of that translation
increases with depth — on a real task, a real (if small) attention-based
architecture, reproducing the same qualitative pattern found across
Experiments 4–7 on an entirely different architecture and task family.

**Next experiment.** Test whether Experiment 7's effective-rank
capacity-metric finding (rank shrinkage predicts compression-gain
magnitude, r≈0.67 on the MLP's hidden layer) transfers to the Transformer
the way the qualitative and quantitative compression findings already did.

---

## Experiment 10 — Does effective rank predict the Transformer's compression gain? (It does not — a genuine divergence from the MLP)

**Hypothesis.** Given that Experiments 8–9 showed the Transformer
reproduces (and sharpens) the MLP's behavioral-robustness and
achievable-compression-ratio findings, effective-rank shrinkage should
similarly predict the *magnitude* of the compression gain here, the way
it did (r≈0.67, Experiment 7) for the MLP's hidden layer.

**Method.** `experiments/analyze_stage_c_lite_capacity_metric.py`,
same 3 seeds as Experiments 8–9, deterministically reproduced. Effective
rank (`atlas_nn.stage_b.capacity_metrics`) computed per layer in both
states, correlated against `log(compression_gain)` from Experiment 9,
both overall and split by Transformer block.

**Result — no positive correlation; if anything, weakly negative, and the
underlying rank shrinkage itself is far smaller than the MLP's.**

| scope | n | Pearson r | Spearman r |
|---|---|---|---|
| overall | 21 | −0.25 | −0.26 |
| block 0 | 9 | −0.33 | −0.03 |
| block 1 | 9 | **−0.54** | **−0.53** |
| classifier head | 3 | 0.48 | −0.50 (n too small to weigh) |

This is the **opposite sign** from Experiment 7's MLP finding (r≈+0.67 on
the hidden layer). Block 1 — the layer type showing the *largest*
compression gains in Experiment 9 — shows the *clearest* negative
correlation between rank shrinkage and gain, of any subgroup tested.

**A second, more basic divergence, visible before even computing a
correlation: the rank shrinkage itself is tiny here.** Random-init
effective rank for `linear1`/`linear2` (max_rank 64) sits around 59–60
(92–93% of max) and only drops to ~58–59 after training (still ~91–92%) —
a 1–2% relative shrinkage. The attention `out_proj` layers shrink slightly
more (51–52 → 49–50, ~3–4%). Compare this to Experiment 7's MLP hidden
layer, where training shrank effective rank by up to 44% at width 256.
**Yet the Transformer's compression and behavioral-robustness gains
(Experiments 8–9) are as large or larger than the MLP's**, despite far
less movement in this particular spectral metric.

**Interpretation.** Effective rank of the raw weight matrix is not a
universal explanation for the capacity-slack effect — it worked
(moderately, r≈0.67, one layer) for the MLP, and does not work at all for
the Transformer, despite the underlying compression/behavioral-robustness
phenomenon itself transferring cleanly (Experiments 8–9). This suggests
the mechanism Experiment 7 partially captured for the MLP's hidden layer
is not "training reduces effective rank, and that's why compression gets
easier" as a general law — something else must explain the Transformer's
robustness gain. A plausible (unverified) candidate: residual connections
and layer normalization, present in the Transformer but not the MLP, may
provide a downstream-mixing/error-absorbing pathway that makes a given
sublayer's output tolerant to perturbation *without* that sublayer's own
weight matrix needing to become lower-rank — i.e. the robustness may live
in the architecture's connectivity, not in any single layer's spectral
structure. This is a **hypothesis, not a verified finding** — untested
this round.

**Caveat on statistical power.** n=9 per block with several *exactly
repeated* compression_gain values across seeds (e.g. block 1 `out_proj`:
gain=3.016 in all 3 seeds) — an artifact of `run_budget_search`'s coarse,
discrete parameter grid landing on the same grid point repeatedly, not
independent measurements. The correlation coefficients above should be
read as suggestive, not as precisely estimated effect sizes; the clearer
and more load-bearing finding here is the qualitative one (no positive
relationship, unlike the MLP), not the exact r values.

**Falsification value.** This is a direct, honest non-replication of
Experiment 7's mechanism on a second architecture, even though the
higher-level phenomenon (Experiments 3, 8) and its quantitative form
(Experiments 4/6, 9) both replicated. Reported as found, not smoothed into
the existing "effective rank explains it" narrative — per mission section
11, a negative result here is exactly as useful as a positive one, and
arguably more informative: it narrows what "capacity" actually means
across architectures rather than letting one convenient metric stand in
for it everywhere.

**Next experiment.** Test the residual-connection/LayerNorm hypothesis
directly: does removing either restore a rank-based relationship, or
change the magnitude/depth-gradient of the robustness gain?

---

## Experiment 11 — Residual/LayerNorm ablation: depth gradient is architecture-independent; LayerNorm drives magnitude; rank correlation is too noisy to trust

**Hypothesis.** If residual connections and/or LayerNorm explain the
Transformer's robustness gain (Experiment 10's working hypothesis),
removing them should shrink the gain and/or restore a positive
rank-shrinkage correlation like the MLP's.

**Method.** `experiments/run_atlas_nn_stage_c_lite_residual_ablation.py`.
A hand-rolled encoder block (`atlas_nn.stage_c_lite.model.
AblationEncoderBlock`, needed since `nn.TransformerEncoderLayer` hardcodes
both features) with independent `use_residual`/`use_layernorm` toggles,
trained in all 4 combinations, 3 seeds each. For each layer: behavioral
robustness gain (relative-logit-error ratio, random-init vs. trained, at
fixed `svd_rank4` — a continuous metric, unlike Experiment 9's discrete
budget-search ratio) and effective-rank shrinkage, correlated as in
Experiment 10, both pooled and split by block.

**Result 1 — the block1 > block0 depth gradient survives every
condition, including with both features removed.** Mean robustness gain,
block0 → block1:

| condition | block 0 | block 1 | ratio |
|---|---|---|---|
| residual + layernorm (baseline) | 4.5 | 27.1 | 6.0× |
| no residual (layernorm only) | 5.8 | 33.1 | 5.7× |
| no layernorm (residual only) | 3.9 | 13.9 | 3.6× |
| neither | 0.6 | 9.9 | 17× (block 0 nearly gone) |

The depth gradient itself is not caused by residual connections or
LayerNorm — it persists, in the same direction, in all four
configurations. Whatever makes later layers gain more from training than
earlier ones is a more fundamental property of depth/position in this
architecture, not an artifact of either ablated feature.

**Result 2 — LayerNorm (not residual connections) drives the overall
magnitude of the gain.** Comparing the two conditions *with* LayerNorm
(4.5→27.1 and 5.8→33.1) against the two *without* (3.9→13.9 and
0.6→9.9): removing LayerNorm roughly halves-to-thirds the block-1 gain in
every comparison; removing residual connections alone barely changes it
(27.1 → 33.1, if anything slightly higher without residuals). This
inverts the original hypothesis's emphasis — LayerNorm, not residual
connections, looks like the more likely magnitude driver.

**Result 3 — the rank-shrinkage correlation is too unstable across
conditions (and reruns) to support a mechanistic claim.** Block-1 Pearson
r: baseline +0.33, no-residual −0.73, no-layernorm +0.63,
neither +0.23 (all n=6–9). Note the baseline here (+0.33) does not even
match Experiment 10's original measurement of the *same conceptual
condition* (−0.54) — different underlying implementation (hand-rolled
block with dropout=0 vs. `nn.TransformerEncoderLayer`'s default dropout,
and a continuous vs. discretized gain metric), same small sample size.
**The sign and magnitude of this correlation are not reproducible enough,
at n≈9, to trust as evidence for or against any specific mechanism** —
this is itself the most important methodological finding of this
experiment, not any single r value.

**Interpretation.** Ablation cleanly separated two previously-conflated
questions. (1) The depth gradient is real, architecture-independent
(within this Transformer's variants), and not explained by either ablated
feature — an open question, not resolved by this experiment. (2)
LayerNorm plausibly explains *why the gains are as large as they are* — a
specific, falsifiable, moderately-supported claim. (3) The Experiment 10
correlation finding should be **downgraded from "moderate negative
correlation" to "no reliable correlation detectable at this sample
size"** — the original claim ("effective rank does not predict the
Transformer's gain the way it predicted the MLP's") still stands, but
the specific r=−0.54 should not be treated as a stable, precisely
estimated effect; see `docs/BEST_RESULTS.md` for the corrected framing.

**Falsification value.** This experiment did what it was designed to do
even though it didn't confirm the original hypothesis: it discriminated
between two candidate mechanisms (residual vs. LayerNorm) rather than
treating "residual/LayerNorm" as one bundled explanation, and it caught
its own predecessor's result (Experiment 10's specific correlation
number) as statistically fragile before that number could be over-relied
upon in future planning.

---

### Experiment 11 addendum — raised to 8 seeds (n=49–56 per condition): the pattern replicates and sharpens

**Why.** Finding 2 above (LayerNorm drives gain magnitude) rested on just
2 comparison pairs (n=6–9 per block per condition) — real signal, too
thin to trust on its own. Re-ran the identical script
(`experiments/run_atlas_nn_stage_c_lite_residual_ablation.py`) with SEEDS
raised from `(11, 22, 33)` to 8 seeds, no other changes.

**Result — both findings replicate closely, and the correlation picture
sharpens into something more interpretable (pooled across all 7 layers
per condition, n=49–56):**

| condition | mean robustness gain | rank-shrinkage correlation (Pearson) |
|---|---|---|
| residual + layernorm (baseline) | 16.3 (was 13.8 at 3 seeds) | 0.23 (was 0.18) |
| no residual | 18.4 (was 17.0) | 0.29 (was 0.30) |
| no layernorm | 8.1 (was 7.9) | **0.55** (was 0.51) |
| neither | 4.1 (was 4.6) | 0.37 (was 0.37) |

Every mean-gain number changed by less than 15% between the 3-seed and
8-seed runs — a strong reproducibility signal on its own. The magnitude
finding holds exactly as before: conditions *with* LayerNorm (16.3, 18.4)
show roughly 2–4× the gain of conditions *without* it (8.1, 4.1);
removing residual connections alone still doesn't reduce gain (if
anything, 16.3→18.4 is a slight increase).

**The correlation picture is now cleaner and consistently positive in
all 4 conditions** (0.23 to 0.55) — unlike the original n=9 measurement
in the entry above (which found a *negative* r=−0.54 on block 1 alone).
With better power, **all four conditions actually show a positive
rank-shrinkage relationship, and removing LayerNorm consistently raises
it** (0.23→0.55 holding residual fixed; 0.29→0.37 the other way) — a
real, replicated refinement: LayerNorm doesn't just increase gain
magnitude, it also *decouples* the gain from the layer's own weight-matrix
rank structure, consistent with LayerNorm providing a normalization
pathway that makes robustness less dependent on raw spectral properties.

**Per-block breakdown remains noisier** (n=21–24 per block, still smaller
than the pooled n): block 1's own Pearson r ranges from −0.22
(`no_residual`) to +0.64 (`no_layernorm`) — the *direction* "no_layernorm
has the highest correlation" holds, but individual block-level numbers
still shouldn't be over-read. One seed (33) failed to train under
`no_residual` specifically (held-out accuracy did not clear the
success threshold) — a reminder that removing residual connections does
carry a real, if occasional, optimization-difficulty cost, consistent
with the general risk found back in Experiment 5's deep-MLP degeneracy.

**Depth gradient re-confirmed at much better power too:** block-1 mean
gain exceeds block-0 in all 4 conditions by 4.0×–9.4× (previously
3.6×–17× at n=6–9) — same conclusion, tighter estimate.

**Revised interpretation.** Experiment 10's original r=−0.54 is now
better understood as a small-sample artifact rather than a stable
negative effect — the corrected picture (this addendum) is that
rank-shrinkage correlates *positively* with the Transformer's gain in
every condition tested, just more weakly when LayerNorm is present
(0.23–0.29) than when it's removed (0.37–0.55), and always weaker than
the MLP's r≈0.67. This is a more coherent, better-evidenced story than
either the original Experiment 10 (apparent non-relationship or negative
relationship) or the first pass of Experiment 11 (too noisy to trust)
supported — see `docs/BEST_RESULTS.md` for the corrected entries.

**Next experiment.** Pivot to the MLP's longest-standing open question
(input-layer behavior, open since Experiment 4), where better statistical
power is available: does its flat-to-negative post-training
compressibility track the fraction of task-irrelevant input dimensions?

---

## Experiment 12 — MLP input-layer noise-dimension sweep: the noise-fraction hypothesis is not supported

**Hypothesis.** The input layer's distinct, unexplained behavior (no
post-training compressibility gain, sometimes a penalty — Experiments 4,
6, 7) reflects an information-preservation constraint tied to how much of
its input is task-relevant, rather than a capacity/slack effect. If so,
its post-training compressibility should improve as the fraction of
task-irrelevant ("noise") input dimensions increases.

**Method.** `experiments/run_atlas_nn_stage_b_input_layer_noise_sweep.py`.
2-XOR task (informative dims fixed at 2), `n_features` ∈ {4, 8, 16, 32,
64} (noise fraction 50%–97%), network width fixed at 64, **5 seeds per
condition** (more than the usual 3, since this question has been open
longest and deserved better power). Input layer's best-ratio-at-5%-
behavioral-error, both states, at each noise level.

**Result — no supporting trend across the tested range; input layer stays
flat-to-negative throughout.**

| n_features | noise fraction | random-init ratio | trained ratio | gain |
|---|---|---|---|---|
| 4 | 50% | 7.05 | 7.53 | 1.07 |
| 8 | 75% | 6.24 | 6.24 | 1.00 |
| 16 | 87.5% | 7.88 | 5.28 | **0.67** |
| 32 | 93.8% | 5.31 | 5.31 | 1.00 |
| 64 | 96.9% | 6.91 | 4.79 | **0.69** |

No monotonic (or any clean) trend with noise fraction — gain hovers at
1.0 or below across the entire 50%–97% range tested, including two
conditions (16 and 64 features) with a real penalty. **This does not
support the noise-fraction hypothesis as stated.**

**A follow-up check at the zero-noise extreme (n_features=2, no noise
dimensions at all) shows a different, if inconsistent, picture:** mean
gain ≈1.14× (random-init ratio mean 6.23 → trained mean 7.08), with 4 of 5
seeds showing a positive direction and one reversed. This hints that the
*very* low end of the noise-fraction range might behave differently from
the 50–97% range tested in the main sweep — but the effect is small,
inconsistent across seeds, and the input dimensionality itself is
unusually tiny at this setting (a 64×2 weight matrix, an edge case not
comparable in shape to the rest of the sweep). Not confirmed; flagged as
a loose end, not a finding.

**Interpretation.** The clean, monotonic "more noise → more input-layer
slack" story is not supported by this data — the flat-to-negative pattern
found in prior experiments is robust across most of the noise-fraction
range, not something that resolves as noise increases. The input layer's
distinct behavior remains unexplained. This is a real negative result,
not a failed experiment: it rules out a specific, plausible, previously
untested hypothesis with reasonable statistical power (5 seeds × 5
conditions = 25 measurements), narrowing the space of remaining
explanations (e.g. something about being the very first layer to receive
un-normalized raw input, rather than about the input's informativeness
per se) for whoever picks this question up next.

**Confound worth flagging.** `n_features` was swept with training data
size held fixed (2000 samples), so higher `n_features` conditions are
simultaneously "more noise dimensions" *and* "a harder generalization
problem with the same amount of data" (trained held-out accuracy dropped
from 98.4% at n_features=4 to 73.6% at n_features=64). A cleaner version
of this experiment would scale training data with `n_features` to hold
task difficulty constant while varying only the noise fraction — not done
here, noted for anyone extending this.

**Next experiment.** Cross-architecture check of Experiment 11's
strengthened LayerNorm finding: does adding LayerNorm to the MLP raise
its rank-shrinkage correlation the way removing LayerNorm raised the
Transformer's (in reverse)?

---

## Experiment 13 — LayerNorm on the MLP: the rank-decoupling effect generalizes, but the magnitude effect *reverses*

**Hypothesis.** If LayerNorm is the general operative factor behind
Experiment 11's Transformer findings, adding it to the MLP (which
otherwise has none) should mirror those findings: larger post-training
gain, and a *weaker* rank-shrinkage correlation, with LayerNorm than
without.

**Method.** `experiments/run_atlas_nn_stage_b_layernorm_ablation.py`.
`atlas_nn.stage_b.model.build_mlp(use_layernorm=...)` inserts
`nn.LayerNorm` after each hidden block's ReLU. Same 2-XOR task, same
width, 8 seeds per condition from the start (matching Experiment 11's
final power) — `layernorm_off` (the original Stage B architecture) vs.
`layernorm_on`.

**Result — mixed, cleanly split into a generalizing part and a
reversing part (pooled over all 3 layers, n=24 per condition):**

| condition | mean robustness gain | rank-shrinkage correlation (Pearson) |
|---|---|---|
| layernorm_off (baseline) | 9.7 | 0.59 |
| layernorm_on | 2.9 | 0.16 |

**The decoupling direction generalizes:** adding LayerNorm to the MLP
weakens its rank-shrinkage correlation (0.59→0.16), the same direction
Experiment 11 found on the Transformer (LayerNorm present → 0.23–0.29,
LayerNorm absent → 0.37–0.55 — presence of LayerNorm consistently
associated with a *weaker* correlation in both architectures).

**The magnitude direction reverses:** on the Transformer, LayerNorm
roughly **doubled-to-quadrupled** the gain. On the MLP, LayerNorm
**shrinks** the gain by roughly 3.3× (9.7→2.9) — the opposite sign.
LayerNorm amplifies the effect in one architecture and suppresses it in
the other.

**Per-layer breakdown shows the effect is concentrated exactly where the
MLP's capacity-slack mechanism already lives — the hidden layer:**

| layer | layernorm_off gain | layernorm_on gain | layernorm_off rank shrinkage | layernorm_on rank shrinkage |
|---|---|---|---|---|
| input | 2.3 | 2.1 | 0.06 | 0.04 |
| hidden | **25.3** | **5.1** | 0.36 | 0.19 |
| output | 1.5 | 1.5 | 0.26 | 0.20 |

Input and output layers are essentially unaffected by adding LayerNorm
(consistent with the input layer's already-established inertness to
every manipulation tried since Experiment 4, and the output layer's
known small-matrix/overhead-dominated behavior). The entire
magnitude-suppression and rank-decoupling effect is concentrated in the
hidden layer — exactly the layer where the MLP's original capacity-slack
finding (Experiments 4, 6) was strongest.

**Interpretation.** LayerNorm has (at least) two separable effects, and
only one generalizes across architectures tested so far: it consistently
weakens the relationship between a layer's own rank shrinkage and its
compression/robustness gain (present in both MLP and Transformer,
localized to whichever layer already carries the capacity-slack effect
in each architecture) — but its effect on the *size* of that gain is
architecture-dependent, amplifying on the Transformer and suppressing on
the MLP. This rules out "LayerNorm generically helps" as a complete
explanation and points toward an interaction between LayerNorm and
something Transformer-specific (attention is the obvious remaining
candidate, not yet tested in isolation) for the magnitude effect
specifically, while the decoupling effect looks like a more general
property of normalization itself.

**Falsification value.** A clean split like this — one part of a
hypothesis replicating, one part reversing — is more informative than
either a full confirmation or a full failure would have been: it forces
the mechanism question to be more specific ("LayerNorm decouples rank
from gain, generally; something else about the Transformer determines
whether LayerNorm's net effect on magnitude is positive or negative")
rather than leaving "LayerNorm explains it" as an intact, oversimplified
claim.

**Next experiment.** Test attention directly: does removing it (keeping
LayerNorm and the FFN sublayer) collapse the gain toward MLP-like levels,
confirming attention as the ingredient LayerNorm needs to amplify rather
than suppress the effect?

---

## Experiment 14 — Attention ablation: attention is the dominant magnitude driver, and partly explains the depth gradient too

**Hypothesis.** If attention is the missing ingredient behind
Experiment 13's magnitude-sign reversal (LayerNorm amplifies on the
Transformer, suppresses on the MLP), removing attention from the
Transformer (while keeping LayerNorm) should collapse the gain toward
the MLP-with-LayerNorm level (~2.9, Experiment 13) rather than staying
near the Transformer's usual ~16.

**Method.** `experiments/run_atlas_nn_stage_c_lite_attention_ablation.py`.
Added `NoMixingAttention` (`atlas_nn.stage_c_lite.model`) — a drop-in
replacement for `nn.MultiheadAttention` with the same `out_proj` naming
but no cross-token mixing (a plain per-token linear projection instead of
attention-weighted averaging). 4 conditions crossing attention on/off ×
LayerNorm on/off (residual connections held on throughout — Experiment 11
found they don't matter), 8 seeds each.

**Result — confirmed, and attention turns out to matter *more* than
LayerNorm for magnitude (pooled over all 7 layers, n=56 per condition,
all 8/8 seeds trained successfully in every condition):**

| condition | mean robustness gain | rank-shrinkage correlation |
|---|---|---|
| attention + layernorm (baseline) | 16.3 | 0.23 |
| **no attention** (LayerNorm only) | **3.4** | 0.02 |
| no layernorm (attention only) | 8.1 | 0.55 |
| neither | 5.4 | 0.33 |

Removing attention alone crashes the gain by **79%** (16.3→3.4) — a
bigger drop than removing LayerNorm alone causes (16.3→8.1, 50%).
**3.4 is close to Experiment 13's MLP-with-LayerNorm result (2.9)** —
i.e. a Transformer with attention removed behaves, in gain-magnitude
terms, almost like the MLP that has no attention at all. This is
convergent evidence, not just a directional match: two different routes
to "no attention" (literally not having any, vs. having it and disabling
it) land in the same place.

**The depth gradient also shrinks substantially without attention** —
new information beyond the original hypothesis. Block-1-vs-block-0 gain
ratio: 6.9× with attention present (baseline), dropping to **2.4×**
without attention, vs. only a modest drop to 4.0× when LayerNorm alone is
removed. Attention contributes to *both* the magnitude effect and (unlike
residual connections, which Experiment 11 found irrelevant to it) part of
the depth gradient itself — plausibly because attention is what lets
later layers accumulate more context-mixed information from earlier ones
to begin with.

**The rank-decoupling (correlation) story stays LayerNorm-led.**
Correlation is low whenever LayerNorm is present (0.23, 0.02) and higher
whenever it's absent (0.55, 0.33), regardless of attention — attention's
own removal, if anything, pushes correlation *lower* still (0.23→0.02)
rather than restoring it. LayerNorm remains the primary decoupling
factor; attention's role there is secondary at most.

**Interpretation.** The mechanism now resolves into two largely separable
factors: **attention primarily drives gain magnitude** (and contributes
to the depth gradient), **LayerNorm primarily drives rank-decoupling**
(and contributes a smaller, independent boost to magnitude). Both
findings converge with Experiment 13's MLP result rather than
contradicting it — the MLP has no attention, so its LayerNorm-alone
behavior (suppression, weak decoupling-dominant) is exactly what this
experiment's `no_attention` condition reproduces on the Transformer
architecture. This is the most complete and internally consistent
mechanistic picture the project has produced.

**Falsification value.** The specific, falsifiable prediction (attention
removal → MLP-like magnitude) was confirmed quantitatively, not just
directionally — the two independent measurements (real MLP without
attention, Transformer with attention artificially removed) landing at
nearly the same number (2.9 vs 3.4) is a stronger form of confirmation
than either alone would provide.

**Next experiment.** Return to the project's longest-standing open
question (input layer, since Experiment 4) and re-test the
noise-fraction hypothesis (Experiment 12) with the data-size confound it
flagged actually fixed.

---

## Experiment 15 — Controlled input-layer noise sweep: the null result holds, even stronger, and the earlier hint disappears

**Hypothesis.** Experiment 12 found no relationship between input-layer
compressibility gain and noise fraction, but training data size was held
fixed while `n_features` varied, so higher-noise conditions were
confounded with harder generalization problems (held-out accuracy
73.6%–98.4% across the sweep). If that confound was masking a real
noise-fraction effect, controlling for it should reveal one.

**Method.** `experiments/run_atlas_nn_stage_b_input_layer_noise_sweep_v2.py`.
Same 2-XOR sweep as Experiment 12 (`n_features` now 2/4/8/16/32/64,
adding the zero-noise point to the main sweep instead of as an
addendum), but `N_TRAIN` raised from 2000 to 8000 — verified beforehand
to tighten the held-out accuracy band to 91.0%–99.6% across the same
range. 8 seeds (up from 5).

**Result — the null result holds, is if anything cleaner, and the
previous weak positive hint at zero noise is gone.**

| n_features | noise fraction | mean gain | held-out accuracy (seed 11) |
|---|---|---|---|
| 2 | 0% | **1.00** | 99.6% |
| 4 | 50% | 0.87 | 99.6% |
| 8 | 75% | 0.86 | 98.4% |
| 16 | 87.5% | 0.73 | 97.2% |
| 32 | 93.8% | 0.94 | 94.8% |
| 64 | 96.9% | 0.80 | 91.0% |

No trend across the full 0%–97% noise-fraction range — gain sits at
0.73–1.00 throughout, i.e. no reliable post-training compression gain on
the input layer at *any* noise level, including now-cleanly-measured zero
noise. Experiment 12's addendum had shown a weak, inconsistent positive
hint at `n_features=2` (≈1.14×, 4/5 seeds positive, only 5 seeds, at the
old `N_TRAIN=2000`) — **that hint is gone at 8 seeds and controlled data
size** (exactly 1.00, flat). The most likely explanation: it was sampling
noise from an underpowered measurement, not a real effect.

**Interpretation.** This is a stronger, more definitive version of
Experiment 12's negative result, not merely a repeat: it rules out both
the noise-fraction hypothesis *and* the specific loose end (the zero-noise
hint) that Experiment 12 left open. The input layer's flat-to-negative
post-training compressibility remains completely unexplained after four
experiments now (Experiments 4, 6, 7, 12, 15) — noise fraction, effective
rank, and (implicitly, since it was never differentially affected in any
MLP experiment) network width have all been checked and found wanting.
Whatever governs the input layer's distinct behavior has not yet been
identified among the factors this project has tried.

**Falsification value.** Deliberately re-testing a prior null result
under better-controlled conditions, rather than treating the first
negative result as final, is exactly the kind of check the mission's
falsification discipline calls for — and here it strengthened rather than
overturned the original conclusion, which is itself useful information:
the original result was not an artifact of the confound.

**Next experiment.** With rank (Exp 7) and noise fraction (Exp 12, 15)
both ruled out, test a third candidate: is the input layer's behavior
about literally seeing *raw, untransformed* task input, unlike every
other layer, which receives an already-processed upstream representation?

---

## Experiment 16 — Raw vs. projected input: also ruled out

**Hypothesis.** `atlas_nn.stage_b.model.build_mlp(use_frozen_input_
projection=True)` prepends a frozen, orthogonally-initialized, never-
trained linear projection before the trainable stack, so the first
*trainable* layer sees a fixed transform of the raw input instead of the
raw input directly — same task, same layer shape, only "rawness" changes.
If the input layer's flat-to-negative compressibility is about literally
seeing raw features, the first trainable layer should behave more
hidden-layer-like (a real positive gain) once it's one step removed from
raw input.

**Method.** `experiments/run_atlas_nn_stage_b_input_layer_raw_vs_projected.py`.
2-XOR task, 8 seeds, `raw_input` (baseline) vs. `projected_input`
(frozen orthogonal projection in front). Same Experiment 4/6/12/15
budget-search methodology, measuring the first *trainable* layer's gain
in both conditions.

**Result — ruled out, cleanly, in the wrong direction if anything:**

| condition | mean random-init ratio | mean trained ratio | gain |
|---|---|---|---|
| raw_input (baseline) | 5.63 | 5.31 | 0.94 |
| projected_input | 6.62 | 5.31 | **0.80** |

No positive gain in either condition — both flat-to-negative, matching
every prior measurement of the input layer. If anything, the projected
condition is slightly *worse* (0.80 vs 0.94), the opposite of what the
raw-input hypothesis predicted. The frozen projection itself, being
untrained by construction, trivially shows no change between states
(excluded from analysis, as planned).

**Interpretation.** Three specific hypotheses now tested and ruled out
for the input layer's distinct behavior: effective rank (Experiment 7,
weak/inconsistent), noise fraction (Experiments 12, 15, cleanly ruled
out), and raw-vs-processed input (this experiment, cleanly ruled out).
The behavior itself remains real and reproducible — it just isn't
explained by any capacity-, information-, or preprocessing-based story
tried so far.

**Next experiment.** Finish disentangling Experiment 14's attention
ablation: was the magnitude-drop from removing attention really about
losing cross-token mixing, or just about losing ~3/4 of attention's
parameters?

---

## Experiment 17 — Attention's magnitude effect is about mixing, not parameter count

**Hypothesis.** Experiment 14's `NoMixingAttention` removed both
cross-token mixing and most of attention's parameters simultaneously
(only `out_proj` remained; `in_proj_weight`, ~3/4 of the total, was
dropped). If parameter count — not mixing — was the real driver of the
79% gain drop found there, restoring parameter count while still skipping
mixing should bring the gain back toward `full_attention`'s level. If
mixing is what matters, restoring parameters without mixing should stay
near `NoMixingAttention`'s level.

**Method.** `experiments/run_atlas_nn_stage_c_lite_attention_param_disentangle.py`.
Added `MatchedParamNoMixingAttention` (`atlas_nn.stage_c_lite.model`) —
keeps `in_proj_weight`/`in_proj_bias` as raw Parameters (same storage
convention as real `nn.MultiheadAttention`, invisible to
`linear_layer_names`, so the tracked-layer set is unchanged), giving it
the exact same total parameter count as real attention (verified: 16,640
params either way), but skips the softmax cross-token mixing step
entirely — the value projection goes straight to `out_proj`, per token.
3 conditions, LayerNorm and residual held on, 8 seeds.

**Result — parameter count restored, mixing still absent, gain stays
low: mixing is the operative factor, decisively.**

| condition | mean robustness gain | rank-shrinkage correlation |
|---|---|---|
| full_attention (baseline) | 16.3 | 0.23 |
| no_mixing_no_param_match (= Experiment 14's `no_attention`) | 3.4 | 0.02 |
| **no_mixing_matched_params** (new) | **4.4** | −0.05 |

`no_mixing_matched_params` (4.4) lands close to the original
parameter-poor no-mixing condition (3.4) — both far below full attention
(16.3) — despite having the identical parameter count as real attention.
Restoring the missing ~12,000 parameters (`in_proj_weight`/`bias`)
changed the mean gain by only 0.9 (3.4→4.4, within seed-to-seed noise
territory) while leaving out the actual mixing step left the gain at
roughly a quarter of full attention's. The `full_attention` and
`no_mixing_no_param_match` numbers here match Experiment 14's exactly
(16.3 and 3.4), confirming this script correctly reproduces that
experiment before trusting the new condition.

**Interpretation.** This resolves the ambiguity Experiment 14 left open:
attention's magnitude effect is about cross-token information mixing
specifically, not about having more parameters in that sublayer. A
same-sized but non-mixing sublayer behaves like no attention at all, not
like attention. This sharpens the project's mechanism picture to its most
precise form yet: **cross-token mixing (attention specifically) drives
gain magnitude and part of the depth gradient; LayerNorm drives
rank-decoupling and a smaller, independent magnitude contribution.**

**Falsification value.** This was a genuine two-way test, not a foregone
conclusion — parameter count could plausibly have been the real factor
(more parameters generally means more capacity to develop useful
structure during training). Finding that it wasn't is informative
precisely because the alternative was a reasonable prior expectation, not
a straw man.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 18 — Stage C (real): does the behavioral-robustness effect appear on a literal pretrained model?

**Context.** `huggingface.co` was blocked by this session's network policy
through Experiments 8–17 (confirmed via the egress proxy's status
endpoint), which is why Stage C-lite substituted a from-scratch Transformer
on a real text task instead of mission Stage C's literal "manageable open
pretrained model." Partway through this session the user changed the
environment's network policy; `huggingface.co` and the HF Hub API now
return `200` and serve real data (verified directly, not assumed from a
changed setting). This is the first experiment in the project to use an
actually-trained, externally-produced model rather than a network trained
from scratch inside this repository.

**Hypothesis.** If the behavioral-robustness effect (Experiments 3, 8: a
trained network's output is far less sensitive to weight-compression error
than tensor error predicts, compared to the same architecture at random
init) reflects something general about trained networks rather than an
artifact of this project's own training runs, it should reproduce on a
real pretrained checkpoint trained by someone else, on real data, with a
real training procedure this project has no visibility into.

**Method.** `atlas_nn/stage_c_real/`: distilgpt2 (82M parameters, 6
GPT-2 Transformer blocks, `transformers.AutoModelForCausalLM`). Given CPU
budget, a fixed depth-balanced subset of 12 Linear-equivalent (`Conv1D`)
layers is tested — blocks 0, 3, and 5 (early/mid/late of 6), all 4 sublayer
types (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) — rather
than the full model (a full sweep is not practical on CPU at this scale;
see the module docstring). Two arms: the real pretrained checkpoint (one
fixed instance — there is only one distilgpt2, so no seeds apply to this
arm) vs. three freshly-initialized, **completely untrained** copies of the
same architecture (seeds 11, 22, 33; no training performed on this arm at
all, unlike Stage B/C-lite's random-init-then-train design — so there is
no training-success confound to check here, unlike every earlier
experiment in this project). Behavioral eval: teacher-forced next-token
accuracy and full-logit relative L2 error on a fixed batch of 8
self-authored English sentences (24 tokens each). `experiments/
run_atlas_nn_stage_c_real_smoke.py`, 7 fixed-parameter methods per layer
(mirroring Experiments 3/8's smoke-test pattern) — 336 total rows.
`results/atlas_nn_stage_c_real_smoke.json`.

**Result 1 — next-token accuracy confirms the two arms are genuinely
different regimes.** Pretrained: 15.8% exact-match next-token accuracy on
the eval batch (real, non-trivial language modeling — GPT-2's own
tokenizer/vocab makes exact top-1 next-token match a hard task even for a
well-trained small model on generic sentences). Random-init: **exactly
0.0%** in all 3 seeds — expected under pure chance given vocab size 50,257
and ~184 scored positions per seed (P(0 correct) ≈ 99.6% under the null),
not a bug, but a clean confirmation the untrained arm is genuinely
untrained.

**Result 2 — the core effect reproduces cleanly across every method with
non-trivial baseline error (mean over all 12 layers, all seeds pooled for
random-init):**

| method | pretrained rel_logit | random-init rel_logit | gain |
|---|---|---|---|
| svd_rank4 | 0.116 | 0.564 | **4.9×** |
| vector_codebook_k16 | 0.098 | 0.428 | **4.4×** |
| prune_50pct | 0.023 | 0.144 | **6.2×** |
| atlas_block_dict16_res4bit | 0.007 | 0.047 | **6.8×** |
| quantize_4bit_block64 | 0.006 | 0.048 | **7.6×** |
| quantize_8bit_pertensor | 0.005 | 0.006 | 1.2× (both already near-lossless) |
| zlib_lossless | 0.000 | 0.000 | — (exact, no compression either way) |

Every lossy method with room to show a difference (i.e. not already
near-zero error in both states) shows a substantial gain, in the same
4–8× range this project's synthetic/from-scratch experiments have
consistently found. `quantize_8bit`'s near-1× "gain" is exactly what the
existing framework predicts: when tensor error is already tiny in both
states, there's no behavioral degradation left to differentially recover
from.

**A sharper version of the core finding, from `quantize_4bit_block64`
specifically.** Its *tensor*-level error is actually slightly **higher**
for the pretrained checkpoint than for random-init (0.101 vs. 0.090) — the
opposite direction from the usual pattern — yet its *behavioral* error is
7.6× **lower** for pretrained (0.006 vs. 0.048). This is the cleanest
demonstration in the whole project that the effect is specifically about
behavioral robustness, not about the trained tensor being "easier to
compress" in a generic tensor-error sense — the same or even larger tensor
perturbation does far less behavioral damage once the network is trained,
independent of whether tensor-level compressibility itself improved.

**Result 3 — the depth pattern is not the same clean monotonic gradient
Stage C-lite found, but a genuine result, not noise (svd_rank4, mean
rel_logit_error by block):**

| block | pretrained | random-init | gain |
|---|---|---|---|
| 0 (early) | 0.189 | 0.991 | 5.2× |
| 3 (mid) | 0.050 | 0.394 | **7.9×** |
| 5 (late) | 0.108 | 0.307 | **2.8×** |

Block 3 (middle) shows the largest gain, block 5 (last) the smallest —
not the "gain increases monotonically with depth" pattern Stage C-lite's
2-block Transformer showed (Experiments 8–9). This does not contradict
Stage C-lite's finding (that Transformer was only 2 blocks deep, so
"monotonic across 2 points" is a much weaker claim than a genuine
non-monotonic pattern across 3 points here); it does mean "gain
increases with depth" should not be treated as a general law without
more architectures/depths tested. A real, reportable divergence, not
smoothed over.

**Result 4 — extremely tight reproducibility across random-init seeds**
(svd_rank4, pooled over all 12 layers): seed 11 → gain 4.94×, seed 22 →
4.88×, seed 33 → 4.80× — a spread of under 3% across seeds. Tighter than
any earlier experiment's seed-to-seed spread in this project, which makes
sense: unlike Stage B/C-lite's random-init arm (a snapshot *before*
training, subject to that specific training run's optimization dynamics
elsewhere in the same experiment), this random-init arm involves no
training step of any kind — pure weight initialization variance only.

**Interpretation.** This is the strongest single piece of evidence in the
project for the mission's central hypothesis: the behavioral-robustness
effect (Experiments 3, 8) was previously demonstrated only on networks
trained inside this repository, on synthetic or self-authored tasks, with
this project's own training procedure. Here it reproduces, at similar or
larger magnitude (4.4–7.6× vs. the 3–11× range from Experiments 3/8), on
a model this project did not train, whose training data, procedure, and
duration are unknown here — exactly the kind of external validation the
mission's falsification discipline calls for. The one genuine surprise
(non-monotonic depth pattern) is reported as found, narrowing rather than
overturning the earlier depth-gradient finding.

**Scope of the claim.** One pretrained checkpoint (no seeds possible for
that arm — it is a single fixed model), one small model (82M parameters,
still far below "large" by current standards), a curated 12-layer subset
rather than the full model, fixed compression parameters rather than a
budget-search ratio (that follow-up — mirroring Experiments 4/6/9 — is
`experiments/run_atlas_nn_stage_c_real_budget_search.py`, not yet run at
the time of this writeup). The eval set is 8 short self-authored
sentences — enough to distinguish "untrained" from "trained" cleanly
(Result 1) but not a rigorous language-modeling benchmark.

**Next experiment.** Run the budget-search version (achievable compression
ratio at matched 5% behavioral-error quality, mirroring Experiments 4/6/9)
on the same model, to turn this into an actionable "Nx compression"
number the way Experiment 9 did for Stage C-lite.

---

## Experiment 19 — Stage C (real) budget search: achievable compression ratio on distilgpt2, and a much cleaner depth gradient than Experiment 18's fixed-parameter view suggested

**Hypothesis.** Given Experiment 18's qualitative finding (behavioral
error 4.4–7.6× smaller for pretrained distilgpt2 than for untrained copies,
at fixed compression parameters), the achievable compression ratio at a
fixed 5% behavioral-error bar (the same methodology as Experiments 4/6/9)
should be substantially higher for the pretrained checkpoint, and should
increase with depth.

**Method.** `experiments/run_atlas_nn_stage_c_real_budget_search.py`.
`atlas_nn.stage_b.budget_search.run_budget_search` (unchanged, reused as-is
since the Experiment 8 refactor) applied to a smaller, still
depth-balanced layer subset than Experiment 18 — blocks 0 and 5 (early and
late) × 3 sublayer types (`attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` — the
fused `attn.c_attn` was dropped for this script specifically to keep
runtime manageable; see the script's docstring) — 6 layers, pretrained
(1 fixed checkpoint) vs. 3 random-init seeds, full 31-config sweep per
layer. Real wall-clock cost: ~11 min for the first layer-state search,
settling to ~7–9 min/search once warmed up, ~3.3 hours total for all 24
searches (6 layers × 4 states) — reported here because the user explicitly
asked to wait for the full run rather than a reduced-scope version, and
CPU cost at this model scale is itself a real, useful data point for
anyone reproducing this.

**Result — the achievable-ratio gain is much larger, and far more clearly
depth-graded, than Experiment 18's single-fixed-method view showed (mean
best-ratio-at-5%-error over 3 random-init seeds):**

| layer | pretrained ratio | random-init ratio (mean) | gain |
|---|---|---|---|
| block 0 attn.c_proj | 8.00 | 5.33 | 1.5× |
| block 0 mlp.c_fc | 6.34 | 5.33 | 1.2× |
| block 0 mlp.c_proj | 6.34 | 5.33 | 1.2× |
| block 5 attn.c_proj | 63.1 | 8.00 | 7.9× |
| block 5 mlp.c_fc | 8.00 | 5.33 | 1.5× |
| block 5 mlp.c_proj | **307.2** | 5.33 | **57.6×** |

Block 0's mean gain (1.3×) is modest; block 5's (22.3×, driven mostly by
the `mlp.c_proj` outlier but present in `attn.c_proj` too at 7.9×) is the
largest depth-gradient magnitude found anywhere in this project — larger
than Stage C-lite's original 1.5×–4.4× block0→block1 gradient
(Experiment 9). The single largest number in the whole project:
`transformer.h.5.mlp.c_proj` (a 3072×768 matrix) meets the 5%
behavioral-error bar at **SVD rank 2** on the pretrained checkpoint
(307.2× ratio, 2.3% actual behavioral error) vs. needing the safest
6-bit-quantization fallback on random-init (5.33×, no low-rank structure
usable at all pre-training).

**A structural finding, not just a magnitude one: training changes which
method family is even viable, not just how far the same method goes.**
Every random-init search's winning family was `quantize` (the safe,
structure-agnostic fallback) in all 24 random-init searches without
exception. Every pretrained search's winner was something else —
`atlas_block_dict` (2 of 6 layers), `vector_codebook` (1), `svd` (1), and
`quantize` only where nothing else cleared the bar (2 of 6, both still
matching or beating random-init's ratio). Structure-aware methods
(low-rank, dictionary/codebook) are entirely unusable on the untrained
weights at this error threshold, and become the best available option
after training, on 4 of 6 layers.

**Reconciling this with Experiment 18's odd depth pattern.** Experiment
18's fixed-`svd_rank4` view found a non-monotonic depth pattern (block 3 >
block 0 > block 5) that didn't match Stage C-lite's clean gradient. This
experiment's true achievable-ratio view (each layer/state picking its own
best method, not forced through one fixed config) shows a much cleaner,
strongly monotonic block0 ≪ block5 gradient instead. The most likely
explanation: forcing every layer through the same fixed method (rank-4
SVD) in Experiment 18 measured how well *that one method* happened to fit
each layer's specific weight structure, which is not the same question as
"how compressible is this layer at all" — block 5's `mlp.c_proj`, for
instance, actually wants rank 2, not rank 4, to hit its best ratio; a
fixed rank-4 probe would under- or over-shoot depending on the layer's
own natural rank, adding noise unrelated to the underlying depth trend.
The budget-search view, which lets each layer find its own best
config, is the more trustworthy measure of the true depth pattern.

**Interpretation.** This is the clearest, largest-magnitude confirmation
of the mission's central hypothesis in the whole project: on a real
82M-parameter pretrained model this project did not train, one specific
layer near the network's output tolerates 307× compression at under 2.3%
behavioral error, using structure (low rank) that simply doesn't exist in
the same layer before training. The depth gradient — modest near the
input, large near the output — replicates and sharpens the pattern found
on the from-scratch Stage C-lite Transformer (Experiments 8–9), now on a
real pretrained checkpoint at larger absolute magnitude.

**Scope of the claim.** 6 of the model's 24 Conv1D layers (a smaller
subset than Experiment 18's 12, chosen for CPU feasibility — see the
script's docstring), one pretrained checkpoint (no seeds possible for that
arm), 3 random-init seeds, a discrete method/parameter grid (so ratios
like 307.2 reflect the actual best grid point, not a continuous optimum —
same caveat as Experiment 9). The `mlp.c_proj` outlier (57.6×) dominates
the block-5 mean; without it, block 5's other two layers still show a
clear 1.5×–7.9× gain, so the qualitative depth-gradient claim does not
rest on that one number alone.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 20 — Cross-model generalization: does the effect (and Experiment 18's depth-pattern puzzle) replicate on gpt2, or is it distilgpt2-specific?

**Hypothesis.** If the behavioral-robustness effect found on distilgpt2
(Experiments 18–19) reflects something general about GPT-2-family
pretrained models rather than being specific to that one checkpoint, it
should reproduce at similar magnitude on `gpt2` (124M parameters, 12
blocks, same 768-dim/12-head configuration as distilgpt2 but not
distilled and twice as deep).

**Method.** `atlas_nn/stage_c_real/model.py` was generalized to accept a
`model_name` parameter (previously hardcoded to `"distilgpt2"`), deriving
the tested-block subset from the model's own `config.n_layer` (first,
middle, last — blocks 0, 6, 11 for gpt2's 12) rather than a value pinned
by hand. `experiments/run_atlas_nn_stage_c_real_smoke_gpt2.py` reruns
Experiment 18's exact smoke-test methodology (7 fixed-parameter methods,
12 tested Conv1D layers, pretrained vs. 3 untrained random-init seeds)
against `gpt2` instead. `results/atlas_nn_stage_c_real_smoke_gpt2.json`.

**Result 1 — the core effect reproduces, at closely matching magnitude to
distilgpt2 (mean rel_logit_error, pooled over all 12 layers/seeds):**

| method | pretrained | random-init | gain | (distilgpt2's gain, Exp. 18) |
|---|---|---|---|---|
| prune_50pct | 0.015 | 0.121 | **8.0×** | 6.2× |
| atlas_block_dict16_res4bit | 0.006 | 0.038 | **6.8×** | 6.8× |
| quantize_4bit_block64 | 0.007 | 0.040 | **5.3×** | 7.6× |
| vector_codebook_k16 | 0.108 | 0.354 | **3.3×** | 4.4× |
| svd_rank4 | 0.180 | 0.466 | **2.6×** | 4.9× |
| quantize_8bit_pertensor | 0.004 | 0.005 | 1.4× | 1.2× |

Every method's gain falls within the same rough 1.4×–8× band found on
distilgpt2, on a model this project also did not train, of a different
size (124M vs. 82M) and depth (12 vs. 6 blocks). Reproducibility across
random-init seeds is tight, matching distilgpt2's pattern (svd_rank4 gain:
2.63×, 2.64×, 2.49× across seeds 11/22/33 — under 6% spread).

**Result 2 — a genuinely new wrinkle: two different fixed methods give
two different, method-dependent depth pictures, neither matching a clean
monotonic gradient (mean rel_logit_error by block):**

| block | svd_rank4 gain | atlas_block_dict gain |
|---|---|---|
| 0 (early) | 2.78× | **12.33×** |
| 6 (middle) | **4.65×** | 5.31× |
| 11 (late) | 1.36× | 2.85× |

`svd_rank4` peaks at the *middle* block (echoing Experiment 18's own
non-monotonic, middle-peaking distilgpt2 result almost exactly — block 3
of 6 there, block 6 of 12 here). `atlas_block_dict`, in contrast, shows a
*monotonically decreasing* gain with depth — the opposite direction from
what Experiment 19's budget search found for distilgpt2 (where the
achievable-ratio gradient rose sharply toward the *last* block). Two
different fixed compression methods on the same model, same layers,
disagree with each other about which end of the network benefits more.

**Interpretation.** This strengthens Experiment 19's core methodological
lesson rather than adding a new mystery: Experiment 19 already showed
that a *single fixed* compression method's depth reading (Experiment 18's
`svd_rank4`-only view) doesn't reliably reflect the true achievable-ratio
depth gradient, because different layers have different natural
structure that one fixed method probes unevenly. Here, on a second model,
two different fixed methods give two different, mutually contradictory
depth stories — direct confirmation that **no single fixed-parameter
method's depth pattern should be trusted as "the" depth gradient**; only
a budget search (letting each layer pick its own best method, as
Experiment 19 did for distilgpt2) is a trustworthy measure of it. This
experiment has not yet run that budget search on `gpt2`.

**What is robust across both models, despite the depth-pattern
disagreement.** The *overall*, depth-pooled magnitude of the behavioral-
robustness effect (1.4×–8× depending on method) is essentially identical
between distilgpt2 and gpt2. This is the part of the finding that
generalizes cleanly; the specific shape of the depth gradient is the part
that needs the more expensive budget-search methodology to measure
reliably, on either model.

**Scope of the claim.** Same 12-of-many-more layer subset limitation as
Experiment 18, one model beyond distilgpt2, fixed compression parameters
(not yet a budget search on `gpt2` — that is the natural next step,
mirroring Experiment 19).

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 21 — gpt2 budget search: the depth gradient is REVERSED relative to distilgpt2, not just differently shaped

**Hypothesis.** Given Experiment 20's finding that two fixed compression
methods disagreed with each other about gpt2's depth pattern, and the
established lesson (Experiment 19) that only a budget search gives a
trustworthy depth reading, running the actual budget search on gpt2
should resolve which end of the network benefits more from training —
either matching distilgpt2's late-block-dominant pattern, or revealing
something genuinely different.

**Method.** `experiments/run_atlas_nn_stage_c_real_budget_search_gpt2.py`
(the budget-search script generalized to accept a `model_name`, with
gpt2's first/last blocks — 0 and 11 of 12 — in place of distilgpt2's 0
and 5). Same 6 layers (2 blocks × 3 sublayer types), same 4 states (1
pretrained + 3 random-init seeds), same 31-config sweep, same 5%
behavioral-error bar as Experiment 19. `results/
atlas_nn_stage_c_real_budget_search_gpt2.json`.

**Result — the depth gradient is not just differently shaped from
distilgpt2's, it is reversed (mean best-ratio-at-5%-error over 3
random-init seeds):**

| layer | pretrained ratio | random-init ratio (mean) | gain |
|---|---|---|---|
| block 0 attn.c_proj | **384.0** | 5.33 | **72.0×** |
| block 0 mlp.c_fc | 6.34 | 5.33 | 1.2× |
| block 0 mlp.c_proj | 6.34 | 5.33 | 1.2× |
| block 11 attn.c_proj | 96.0 | 8.00 | 12.0× |
| block 11 mlp.c_fc | 63.8 | 8.00 | 8.0× |
| block 11 mlp.c_proj | 10.0 | 8.00 | 1.3× |

Block 0's mean gain (**24.8×**) is more than 3× larger than block 11's
(7.1×) — on distilgpt2 (Experiment 19), block 0's mean gain was the
*small* one (1.3×) and block 5 (last) was the *large* one (22.3×). Same
methodology, same layer types, same quality bar, opposite depth story.
The single largest number in this experiment (`transformer.h.0.attn.c_proj`,
a 768×768 matrix, 384× compression at 5% behavioral error via SVD) is on
the *first* block — the layer type and rough position that showed the
*least* benefit from training on distilgpt2.

**What still replicates cleanly across both models: the structural
finding.** Exactly as on distilgpt2, every one of the 18 random-init
searches (6 layers × 3 seeds) was won by the safe `quantize` fallback,
with no exceptions; every pretrained search was won by a structure-aware
method (`svd`, `atlas_block_dict`, `vector_codebook`, or `prune`) instead.
Training unlocking entire method families, not just improving ratios
within one family, is robust across both real models tested so far — only
*where in the network* this shows up most strongly differs.

**Interpretation.** This falsifies the specific claim (implicit in
Experiments 8/9/19's consistent pattern) that "gain increases with depth"
is a general property of trained Transformers. It doesn't falsify the
core behavioral-robustness effect itself (Experiment 20 already showed
the *overall* magnitude generalizes cleanly, 1.4×–8× at fixed parameters
on both models) — only the specific, appealing story about *where* in the
network that magnitude concentrates. Two candidate explanations, both
currently unverified: (a) gpt2's larger scale (124M vs. 82M) or extra
depth (12 vs. 6 blocks) changes where representational slack
accumulates during training — plausible, but Stage C-lite's from-scratch
Transformer (2 blocks) and distilgpt2 (6 blocks) both showed the same
late-block-dominant direction, so simple "more blocks" doesn't obviously
predict a full reversal; (b) `gpt2` and `distilgpt2` were trained
differently enough (distillation target vs. from-scratch language-model
objective) that this is fundamentally a training-procedure effect, not a
architecture/depth effect at all — distilgpt2's training explicitly
optimizes it to match a teacher model's *later*-layer behavior via
knowledge distillation, which could plausibly concentrate representational
change (and thus post-training slack) differently across depth than
gpt2's own from-scratch causal-LM training did. Neither is tested here.

**Why this is good news for the project's discipline, not a setback.**
Every step of this thread (Experiment 18's confusing fixed-method reading
→ Experiment 19's budget-search resolution on distilgpt2 → Experiment
20's contradictory fixed-method readings on gpt2 → this budget search)
was exactly the kind of chained, falsification-driven follow-up the
mission calls for, and it just caught a real overgeneralization
(depth-gradient direction) before it could calcify into an unqualified
claim in `docs/BEST_RESULTS.md`.

**Scope of the claim.** Two models, 6-of-24 (distilgpt2) and 6-of-many
(gpt2) layers tested, one behavioral-error threshold, discrete parameter
grids. The training-procedure-difference hypothesis (b above) is
untested — would need a third model with yet another training recipe
(e.g. a from-scratch causal LM at a similar scale to distilgpt2, or
another distilled model) to distinguish from the scale/depth hypothesis
(a).

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 22 — gpt2-medium smoke test: a third model, testing scale within the non-distilled family

**Context.** Working autonomously overnight per the user's explicit
request. Also built and verified (see the commit immediately before this
one) a multiprocess wrapper around `run_budget_search`
(`atlas_nn/stage_c_real/parallel_budget_search.py`), needed because
gpt2-medium's larger matrices make a full budget search too slow
sequentially on this container's 4 CPU cores. A correctness check
(`experiments/verify_parallel_budget_search.py`) confirmed the parallel
version produces byte-identical results to the sequential one on a
distilgpt2 subset (1.80× speedup with only 2 of 4 cores exercised).

**Hypothesis.** Experiment 21 found gpt2's depth gradient reversed
relative to distilgpt2's, with two candidate explanations: (a) model
scale/depth, or (b) distilgpt2's knowledge-distillation training
procedure specifically. `gpt2-medium` (355M parameters, 24 blocks, 1024-
dim — same non-distilled OpenAI training recipe as `gpt2`, but much
larger) tests whether scaling up *within the non-distilled family*
preserves gpt2's early-block-dominant pattern (supporting hypothesis b)
or diverges yet again (supporting hypothesis a, i.e. that scale/depth
itself is what matters, not distillation specifically).

**Method.** `experiments/run_atlas_nn_stage_c_real_smoke_gpt2_medium.py`,
identical smoke-test protocol to Experiments 18/20 (7 fixed-parameter
methods, 12 tested Conv1D layers — blocks 0/12/23 of 24 × 4 sublayer
types — pretrained vs. 3 untrained random-init seeds, 336 rows). This run
was interrupted once by a container restart (killing an in-progress run
at ~89% completion with no partial results saved) and rerun from scratch
using the locally-cached model weights (no re-download needed).
`results/atlas_nn_stage_c_real_smoke_gpt2_medium.json`.

**Result 1 — the magnitude generalizes a third time, to a third model, at
the largest range yet (mean rel_logit_error, pooled over 12 layers/seeds):**

| method | pretrained | random-init | gain | gpt2 (Exp. 20) | distilgpt2 (Exp. 18) |
|---|---|---|---|---|---|
| atlas_block_dict16_res4bit | 0.004 | 0.035 | **9.5×** | 6.8× | 6.8× |
| prune_50pct | 0.017 | 0.108 | **6.4×** | 8.0× | 6.2× |
| quantize_4bit_block64 | 0.005 | 0.036 | **7.6×** | 5.3× | 7.6× |
| vector_codebook_k16 | 0.231 | 0.315 | 1.4× | 3.3× | 4.4× |
| svd_rank4 | 0.254 | 0.413 | 1.6× | 2.6× | 4.9× |
| quantize_8bit_pertensor | 0.003 | 0.005 | 1.7× | 1.4× | 1.2× |

Every gain falls in the same broad band found on the two smaller models —
three real pretrained checkpoints now agree on the *overall* magnitude of
the behavioral-robustness effect, even though (per Experiments 20–21)
they disagree sharply on *where in the network* it concentrates.
Random-init next-token accuracy was again exactly 0.0% in all 3 seeds
(expected under chance given this model's shared GPT-2 vocab and eval
set — see Experiment 18's identical result and its probability
calculation).

**Result 2 — the depth pattern, once again, depends entirely on which
fixed method is probing it (mean rel_logit_error by block):**

| block | svd_rank4 gain | atlas_block_dict gain |
|---|---|---|
| 0 (early) | 1.56× | **25.0×** |
| 12 (middle) | **4.60×** | 4.4× |
| 23 (late) | 1.00× (no measurable gain at all) | 3.2× |

`svd_rank4` peaks in the middle — the same shape gpt2 showed under this
exact method (Experiment 20: block 6 of 12 was the peak there too).
`atlas_block_dict` is monotonically *decreasing* with depth here, the
same *qualitative shape* gpt2 showed under this method (Experiment 20:
12.3×→5.3×→2.9×) even though the *absolute* numbers differ. Both fixed
methods here point toward early/middle blocks mattering more than the
last block — consistent in *direction* with gpt2's fixed-method readings,
not with distilgpt2's. This is suggestive but, per Experiments 19–21's
established lesson, **not yet trustworthy as a depth-gradient claim** —
only a full budget search (Experiment 23, next) settles this the way it
did for distilgpt2 and gpt2.

**Interpretation, held provisionally.** Every fixed-method reading so far
(smoke tests on all 3 models) has pointed one direction and every budget
search so far has told a *different* story once each layer could pick
its own best method (Experiment 19 vs. 18; presumably Experiment 21 vs.
20, though that comparison was less direct). So this experiment's
depth-pattern hint (gpt2-medium's fixed-method readings resembling
gpt2's, not distilgpt2's) should not be treated as evidence for or
against either scale or training-procedure hypothesis until the actual
budget search is run and produces the same kind of achievable-ratio
picture Experiments 19 and 21 did for the other two models.

**Scope of the claim.** Same limitations as Experiments 18/20 (12-of-many
layer subset, fixed compression parameters). The magnitude-generalization
result (Result 1) is solid; the depth-pattern discussion (Result 2) is
explicitly flagged as provisional pending the budget search.

**Next experiment.** Run the budget search on gpt2-medium (using the new
parallel wrapper, needed given this model's larger per-layer compute
cost) to get the first trustworthy depth-gradient reading for a third
model.

---

## Experiment 23 — gpt2-medium budget search: two non-distilled models agree with each other, against the one distilled model — training procedure, not scale, looks like the real driver

**Context.** Run twice. The first attempt (using the newly-built
`atlas_nn.stage_c_real.parallel_budget_search`, correctness-verified
against the sequential implementation beforehand) reached 16 of 24
layer-searches (~67%) before the container restarted unannounced
(confirmed via `uptime` showing a fresh boot), killing the process with
no partial results saved — the second time in one overnight session a
multi-hour run was lost this way. Before rerunning, added per-layer
checkpointing to `parallel_budget_search.py` (each completed search is
written to disk immediately; a rerun skips anything already
checkpointed) and re-verified correctness including a resume check. The
second attempt ran to completion.

**Hypothesis.** Experiment 21 found gpt2's achievable-ratio depth
gradient reversed relative to distilgpt2's, with two untested
explanations: (a) model scale/depth, or (b) distilgpt2's knowledge-
distillation training procedure specifically. gpt2-medium (355M, 24
blocks — much larger than gpt2's 124M/12 blocks, but trained the same
non-distilled way) is the direct test: if its true depth gradient matches
gpt2's despite the large scale jump, that argues against scale/depth and
for training procedure.

**Method.** `experiments/run_atlas_nn_stage_c_real_budget_search_gpt2_medium.py`,
same protocol as Experiments 19/21 (6 layers — blocks 0 and 23 of 24 × 3
sublayer types — × 4 states, full 31-config sweep, 5% behavioral-error
bar), run via the parallel wrapper. `results/
atlas_nn_stage_c_real_budget_search_gpt2_medium.json`.

**Result — gpt2-medium's depth gradient matches gpt2's shape, not
distilgpt2's, and by a similar relative margin (mean best-ratio-at-5%-
error over 3 random-init seeds):**

| layer | pretrained ratio | random-init ratio (mean) | gain |
|---|---|---|---|
| block 0 attn.c_proj | **512.0** | 5.33 | **96.0×** |
| block 0 mlp.c_fc | 8.00 | 5.33 | 1.5× |
| block 0 mlp.c_proj | 6.34 | 5.33 | 1.2× |
| block 23 attn.c_proj | 64.0 | 8.89 | 7.2× |
| block 23 mlp.c_fc | 63.9 | 8.00 | 8.0× |
| block 23 mlp.c_proj | 102.4 | 8.00 | 12.8× |

Block 0's mean gain (**32.9×**) is more than 3× block 23's (9.3×) — the
same *direction* gpt2 showed (block 0: 24.8×, block 11: 7.1×), and a
strikingly similar *relative margin* (both ≈3.5:1, early:late) despite
gpt2-medium having twice gpt2's depth (24 vs. 12 blocks) and nearly 3×
its parameter count (355M vs. 124M). The single
largest number here — 512× compression at 5% behavioral error, via SVD,
on `transformer.h.0.attn.c_proj` — again lands on the *first* block, as
it did for gpt2 (384× at the same relative position), not the last block
the way distilgpt2's single largest number (307×) did.

**This directly discriminates between Experiment 21's two hypotheses.**
Two models with a completely different scale and depth from each other
(gpt2: 124M/12 blocks; gpt2-medium: 355M/24 blocks) — but the *same*
non-distilled, from-scratch OpenAI training recipe — agree closely with
each other on where the gain concentrates. One model with a different
training procedure (distilgpt2: knowledge distillation from a teacher)
disagrees with both, in the opposite direction. If scale/depth were the
real driver, gpt2 and gpt2-medium's 3× parameter and 2× depth difference
should have produced at least some divergence between them; instead they
agree more closely with each other (same ≈3.5:1 ratio) than either does
with distilgpt2. **Training procedure — specifically, whether the model
was trained via knowledge distillation — is now the better-supported
explanation for Experiment 21's reversal**, though still not proven
without a model that varies training procedure while holding scale fixed
(the reverse of what these three checkpoints happen to vary).

**What still holds across all three real models.** The structural
finding is now confirmed a third time without exception: every
random-init search across all three models (18 + 6 = 24 total budget
searches on distilgpt2/gpt2, plus these 18 on gpt2-medium — 42 in total)
was won by the safe `quantize` fallback; every pretrained search that met
the quality bar at all was won by a structure-aware method instead.

**Interpretation.** This is the most direct, best-powered evidence yet on
what causes the depth-gradient reversal — not proof (that would need a
fourth model isolating training procedure from scale directly, e.g. a
second distilled model or an equivalently-sized non-distilled model), but
a real narrowing from "two untested hypotheses" to "one substantially
better-supported hypothesis." It also demonstrates the project's
falsification discipline paying off a third time in this specific thread
(Experiment 19 found the pattern, Experiment 20-21 found and confirmed
the reversal, this experiment narrows *why*) — each step resolved exactly
the ambiguity the previous one raised.

**A methodological note, kept for anyone extending this project.** The
container running this session restarted twice, unannounced, during
long unattended background computation (see the Context section above).
Both times, on-disk state (git history, downloaded model weights, and —
after this experiment's own infrastructure work — checkpointed partial
results) survived the restart; in-memory process state did not. Any
future multi-hour unattended run in this environment should checkpoint
incrementally rather than only saving results at the very end.

**Scope of the claim.** Three models, 6-layer subsets in the two larger
ones, one behavioral-error threshold, discrete parameter grids. The
scale-vs-procedure question is narrowed, not conclusively resolved — a
fourth model varying only one of the two factors would be the decisive
test.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 24 — Effective rank vs. compression gain on real models: another distilled-vs-non-distilled split

**Hypothesis.** Effective rank shrinkage predicted the MLP's
compression-gain magnitude moderately-strongly (Experiment 7, r≈0.67) and
the from-scratch Transformer's more weakly and LayerNorm-dependently
(Experiments 10–11, r≈0.23–0.55). Never checked against a real pretrained
model. If the mechanism generalizes, rank shrinkage should correlate
positively with the compression-gain numbers already measured for
distilgpt2, gpt2, and gpt2-medium (Experiments 19, 21, 23).

**Method.** `experiments/analyze_stage_c_real_capacity_metric.py`. Cheap
by design: reuses the `compression_gain` values already computed by the
budget searches (no new compression sweeps) and computes effective rank
(`atlas_nn.stage_b.capacity_metrics`, unchanged) via one SVD per already-
tested layer, on the already-downloaded pretrained models and freshly-
loaded random-init copies (3 seeds each). No multi-hour compute, no
restart risk. `results/atlas_nn_stage_c_real_capacity_metric.json`.

**Result — a third, independent piece of evidence for the same
distilled-vs-non-distilled split Experiments 21–23 found in the depth
gradient (Pearson r, rank shrinkage vs. log compression-gain):**

| model | n | Pearson r |
|---|---|---|
| distilgpt2 (distilled) | 18 | **−0.29** |
| gpt2 (non-distilled) | 18 | **+0.83** |
| gpt2-medium (non-distilled) | 18 | **+0.49** |
| pooled (all 3 models) | 54 | +0.23 |

Both non-distilled models show a real positive relationship — gpt2's
r=0.83 is the strongest correlation found anywhere in this project,
stronger even than the MLP's original r≈0.67 (Experiment 7). distilgpt2
is not just weaker, it's the *only* one of the three real models where
the correlation is negative. Pooling across all three, as if they were
one population, gives a misleadingly weak r=0.23 that hides a real,
strong, model-dependent effect — the same kind of pooling trap Experiment
7 already flagged for cross-layer pooling within a single model.

**A genuine caveat on statistical power, reported rather than hidden.**
Each model's n=18 is 6 layers × 3 seeds, but random-init effective rank
barely varies across seeds at this scale (e.g. distilgpt2
`h.0.attn.c_proj`: erank_random = 618.33, 618.14, 617.59 across seeds
11/22/33 — under 0.2% spread) — the three seeds are close to redundant
measurements of the same 6 data points, not 18 independent ones. The
*effective* sample size per model is closer to 6 than 18. The r values
above should be read as suggestive at this power, especially
gpt2-medium's more moderate 0.49, though gpt2's 0.83 and distilgpt2's
clearly-negative sign are large enough effects to likely survive the
power caveat.

**Interpretation.** This is now the *second* independent measure (after
Experiments 21/23's depth-gradient shape) where distilgpt2 behaves
differently from both non-distilled models, and both non-distilled
models resemble each other despite gpt2-medium being 3× gpt2's size.
This further strengthens the training-procedure explanation over the
scale explanation for what makes distilgpt2 different: it's not just
*where* in the network the gain concentrates that differs, but whether a
basic spectral property of the weight matrix (effective rank) relates to
that gain at all. A plausible (unverified) unifying story: whatever
knowledge distillation does to a weight matrix's structure may decouple
compression-relevant behavior from that matrix's own rank, similar in
spirit to what Experiment 13 found LayerNorm does for the from-scratch
MLP — though this is speculation, not measured here.

**Scope of the claim.** Three models, 6-of-many layers per model, the
seed-redundancy caveat above, correlational not causal (as with every
capacity-metric result in this project since Experiment 7). Does not
establish *why* distillation would produce this pattern, only that it
does, on two independent measures now.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 25 — Back to the MLP input-layer mystery: is the training UPDATE itself low-rank, not just the final matrix?

**Context.** The MLP input-layer question (no post-training compression
gain, sometimes a penalty, unique among all layer types) has been open
since Experiment 4, with three specific hypotheses tested and ruled out
(effective rank of the *trained* matrix — Experiment 7, weak/inconsistent;
task-irrelevant input noise fraction — Experiments 12/15, cleanly ruled
out; raw vs. pre-transformed input — Experiment 16, no effect).
`docs/NEXT_RESEARCH_DECISION.md` had explicitly flagged that resolving it
"would need a different kind of tool ... not another ablation of the same
shape" — this experiment is the first attempt at that different tool.

**Hypothesis.** Every prior test examined a property of the *final
trained* weight matrix. None asked the more basic dynamical question: how
much does each layer's weight matrix actually *move* during training
(relative to its own random-init scale), and is that movement itself
structured (low-rank) or diffuse (near full-rank)? If the input layer's
training-induced update is diffuse while other layers' updates are
concentrated in a few directions, that would explain reduced post-training
compressibility directly — there would be less *low-rank structure in what
training actually changed*, even if (per Experiment 7) the *final* matrix's
own rank doesn't cleanly predict it.

**Method.** `experiments/run_atlas_nn_stage_b_weight_delta_analysis.py`.
The exact original Stage B setup (3-Linear-layer MLP, 32→64→64→2, 2-XOR
task) used in Experiments 3/4/6/7, 8 seeds (the project's established
standard). For each layer, in each seed: `Δ = W_trained − W_random`,
relative weight movement (`‖Δ‖_F / ‖W_random‖_F`), and — the new
measurement — the effective rank of `Δ` itself, expressed as a fraction
of that layer's max possible rank (`shannon_effective_rank(Δ) /
min(shape)`), directly comparable across layers of different shapes.
`results/atlas_nn_stage_b_weight_delta_analysis.json`.

**Result — a clean, large, fully-separated effect on delta rank
fraction; no comparable effect on movement magnitude (mean over 8
seeds):**

| layer | known compression-gain direction | mean relative weight movement | mean delta-rank fraction |
|---|---|---|---|
| 0 (input, 64×32) | flat-to-negative | 2.19 | **0.877** |
| 2 (hidden, 64×64) | positive | 2.59 | **0.349** |
| 4 (output, 2×64) | positive | 4.23 | 0.500 (degenerate — max_rank=2, see caveat) |

The movement-magnitude column does **not** separate the layers by
compression-gain direction — layer 0 (no gain) moves *less* than layer 2
(positive gain), as a naive "changes less ⇒ less structure" story would
predict, but layer 4 (also positive gain) moves *even more* than layer 2,
so magnitude alone is not the explanation.

**The delta-rank-fraction column separates them cleanly, with zero
overlap across all 8 seeds.** Layer 0's per-seed values range
0.865–0.890; layer 2's range 0.278–0.437 — the two distributions never
touch. The input layer's training-induced *change* uses 88% of its
available rank directions — it's close to full-rank, diffuse, unstructured.
The hidden layer's change uses only 35% — concentrated in a much smaller
effective subspace. This is exactly the shape of result compression
methods would predict matters: a diffuse, near-full-rank update leaves
little exploitable structure for a compression method to find in what
training actually added; a concentrated, low-rank update does.

**Layer 4's degenerate case, noted not smoothed over.** The output layer
has `max_rank=2` (a 2×64 matrix), the same edge case Experiment 7 already
flagged for this layer — "effective rank" over only 2 possible dimensions
is close to a binary variable, and `delta_rank_fraction=0.500` exactly
(consistently, every seed) is more likely an artifact of that degenerate
scale than a real "half of available structure" measurement. The
headline layer-0-vs-layer-2 comparison, both large matrices, is unaffected
by this caveat.

**Interpretation.** This is the first candidate explanation for the
input-layer mystery that shows a large, clean, zero-overlap effect across
all 8 seeds — a qualitatively different result from every prior attempt
(Experiment 7's weak/inconsistent rank correlation, Experiments 12/15's
flat null result, Experiment 16's no-effect result). It reframes the
question productively: it's not that the input layer's *final* matrix
fails to show a rank signal (Experiment 7 already found that, and it's
still true) — it's that what training actually *did* to the input layer
was diffuse, while what it did to the hidden layer was concentrated. Why
training would produce a diffuse update specifically for the layer that
sees raw external input, while producing a concentrated update for
internal layers, is not established here — a plausible (unverified)
story is that the input layer must preserve information about every
task-relevant input coordinate somewhat independently (nothing internal
to the network can recover a coordinate this layer discards), forcing a
broader, less compressible correction, while internal layers can route
information through fewer effective channels because downstream layers
can adapt around whatever specific subspace they receive. Not tested
here.

**Falsification note.** This result should itself now be treated the way
every other promising finding in this project has been: as a hypothesis
needing its own replication check (e.g. does delta-rank fraction predict
the *magnitude*, not just the direction, of compression gain across
harder tasks/widths, the way Experiment 6's capacity-sweep conditions
did for the final-matrix effective rank in Experiment 7?) before being
treated as resolved.

**Scope of the claim.** One task (2-XOR), one architecture width/depth,
8 seeds. A real, well-powered, novel result — not yet cross-checked
against the harder/wider conditions Experiments 6–7 used to strengthen
the original effective-rank finding, and not yet checked on the
Transformer or any real pretrained model.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 26 — Does delta-rank fraction predict gain MAGNITUDE, not just direction? Yes, and it's stronger than Experiment 7's original finding

**Hypothesis.** Experiment 25 found delta-rank fraction cleanly separates
the input layer (no gain, diffuse update) from the hidden layer (positive
gain, concentrated update) — but on one task/width only. If this is a
real mechanism rather than a single-condition coincidence, it should
predict the *magnitude* of compression gain across Experiment 6's full
capacity-sweep grid (2 tasks × 3 widths), the same cross-check that
turned Experiment 7's final-matrix effective-rank finding into a
quantitative, multi-condition relationship.

**Method.** `experiments/analyze_stage_b_delta_rank_capacity_sweep.py`.
Reproduces all 6 of Experiment 6's conditions (`xor2`/`parity3` ×
widths 16/64/256), 3 seeds each, computing delta-rank fraction per layer
and correlating it (Pearson, Spearman) against `log(compression_gain)`
already measured there. Uses the "near ceiling" training-success filter
from the start (the fix Experiment 7's analysis needed after the fact).
`results/atlas_nn_stage_b_delta_rank_capacity_sweep.json`.

**Result — confirmed and strengthened, once read layer-by-layer (n=16
per layer after the training-success filter):**

| layer | Pearson r | Spearman r | mean delta-rank fraction |
|---|---|---|---|
| 0 (input) | 0.18 | −0.09 | 0.841 (narrow range: 0.642–0.928) |
| 2 (hidden) | **−0.75** | **−0.70** | 0.399 (wide range: 0.157–0.676) |
| 4 (output) | 0.62 | 0.65 | 0.500 exactly (degenerate, see caveat) |

Read the sign correctly: delta-rank fraction is *low* when a layer's
training update is concentrated/low-rank, *high* when diffuse. A
**negative** correlation with `log(compression_gain)` therefore means
*lower* delta-rank fraction (more concentrated update) predicts *higher*
gain — exactly Experiment 25's cross-layer direction, now confirmed
*within* the hidden layer across 6 different conditions. At r=−0.75, this
is a **stronger** correlation than Experiment 7's original final-matrix
effective-rank finding (r≈0.67) — the first time in this project a
follow-up cross-check has produced a *larger* effect than the result it
was checking.

**A concrete, visible pattern within `parity3` alone (the hard task, same
one Experiment 6 used to confirm the capacity/slack hypothesis):** as
width rises 16→64→256, the hidden layer's delta-rank fraction falls
0.60→0.39→0.19 while compression gain rises ≈0.7×→1.0×→2.1× (means).
More spare capacity doesn't just create more compressible *final*
weights (Experiment 6/7's story) — it produces a more *concentrated
training update* in lockstep, giving a mechanistic complement to the
capacity/slack story: spare capacity → the update training makes is more
low-rank → the layer becomes more compressible.

**Layer 0 (input) still shows no reliable within-layer relationship**
(r=0.18, inconsistent Spearman sign) — but for a different reason than
"delta rank doesn't matter here": its delta-rank fraction stays
uniformly high (0.642–0.928) across every condition tested, barely
varying with task or width, unlike the hidden layer's wide swing
(0.157–0.676). The input layer's update appears to be diffuse
*regardless* of available capacity — it doesn't get to concentrate the
way internal layers do when given more room, which may itself be a clue
(see Interpretation).

**Layer 4 (output) is degenerate, as flagged before.** Its
`delta_rank_fraction` is exactly 0.500 in nearly every row (this layer's
`max_rank=2`, making effective rank close to a binary variable) — the
r=0.62 here is not treated as a meaningful measurement, same caveat
Experiment 7 raised for this layer.

**The pooling trap, again.** Pooling all three layers together gives
r=−0.58 — technically in the "right" direction only because layer 2
dominates the pooled sample, but it hides that layer 0 shows no real
relationship and layer 4's number is an artifact. The same lesson
Experiment 7 and Experiment 24 already established: always check
per-layer (or per-model) before trusting a pooled correlation.

**Interpretation.** This is the strongest replication of any mechanism
finding in the MLP thread of this project. It reframes the capacity/slack
story (Experiments 5–6) mechanistically: spare capacity doesn't just
leave the final weights more compressible, it changes *how training
updates the weights in the first place* — with more room to work with,
the hidden layer's update concentrates into fewer effective directions.
The input layer's update stays diffuse no matter how much spare capacity
exists elsewhere in the network, consistent with (though not proof of)
the candidate story from Experiment 25: this layer must preserve
information about every task-relevant input coordinate somewhat
independently, since nothing internal to the network can recover a
coordinate this layer discards, which may force a broader correction
regardless of how much "slack" the rest of the network has.

**Scope of the claim.** One architecture family (the Stage B MLP), two
tasks, three widths, 3 seeds per condition (matching Experiment 6/7's
original power, not the project's later 8-seed standard — n=16 per layer
after filtering is smaller than Experiment 25's own n=8 single-condition
comparison, though spread across more conditions). Correlational, not
causal. Not yet checked on the Transformer, nor on any real pretrained
model — the natural next step, mirroring how Experiments 8–9 extended
the original capacity-sweep finding beyond the MLP.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 27 — Does delta-rank fraction transfer to the Transformer? No — echoing Experiment 10's non-replication of the final-matrix finding

**Hypothesis.** Given delta-rank fraction is now the strongest mechanism
correlation found in this project (r=−0.75, Experiment 26, MLP hidden
layer), it should — if it reflects something general about trained
networks — show a similar relationship on the Stage C-lite Transformer.
Flagged going in as a real test, not a foregone conclusion: Experiment 10
already found the *final*-matrix effective-rank finding did not transfer
cleanly to this same Transformer (initial r=−0.25 to −0.54, later
corrected to a real but much weaker r=0.23–0.55 after Experiment 11's
larger seed count).

**Method.** `experiments/analyze_stage_c_lite_delta_rank.py`, reproducing
Experiment 9's exact training setup (3 seeds), computing delta-rank
fraction for all 7 Linear layers and correlating against the
`compression_gain` already measured in `results/
atlas_nn_stage_c_lite_budget_search.json`. `results/
atlas_nn_stage_c_lite_delta_rank.json`.

**Result — no clean relationship, in either pooled or per-block form
(n=21, 3 seeds × 7 layers):**

| scope | n | Pearson r |
|---|---|---|
| overall (pooled) | 21 | −0.15 |
| block 0 | 9 | +0.16 (wrong direction) |
| block 1 | 9 | −0.15 (right direction, weak) |
| classifier head | 3 | 0.95 (**not meaningful — see caveat**) |

The classifier head's r=0.95 is an artifact, not a finding: its
`delta_rank_fraction` is 0.500 in all 3 seeds to 3 decimal places (this
layer has `max_rank=2`, the same degenerate-scale caveat Experiment 7
flagged for the MLP's output layer) — the reported correlation comes from
sub-rounding floating-point noise around an effectively constant value,
not a real relationship. Excluded from interpretation.

**A sublayer-type breakdown that directly contradicts the naive
prediction.** `out_proj` has the *lowest* mean delta-rank fraction (0.372,
most concentrated) of any sublayer type, which the MLP finding would
predict should mean the *highest* gain — instead it has the *lowest*
mean gain (2.230) of the three real sublayer types. `linear2` has a
middling delta-rank fraction (0.473) and the *highest* gain (3.813). If
anything, this specific breakdown points the opposite direction from the
MLP's finding, though with too few sublayer types (3) to treat as a
reliable pattern on its own.

**Interpretation.** A genuine, reportable non-replication — not
softened, not buried. Delta-rank fraction's strong correlation with
compression gain (Experiments 25–26) appears to be a real property of
this project's MLP specifically, not a general property of "how trained
networks work" that automatically transfers across architectures. This
echoes Experiment 10's finding almost exactly: a mechanism metric
computed on the final matrix's structure predicted the MLP well and the
Transformer poorly; the same now holds for a metric computed on the
*update's* structure instead. Both the "what" (rank-based capacity
metrics) and now "when" (final matrix vs. training update) versions of
this mechanism-hunting approach hit the same architecture-generalization
wall. A plausible (unverified) reading: the Transformer's residual
connections and LayerNorm — already implicated in Experiments 11/13/14/17
as providing an error-absorbing pathway independent of any single
sublayer's own weight structure — may make *both* kinds of rank-based
metric (final matrix, training update) less informative there than they
are for the MLP, which has neither.

**Scope of the claim.** One Transformer, one real (self-authored) task,
3 seeds — the project's original Stage C-lite power level, not the later
8-seed standard. A larger-seed rerun (mirroring how Experiment 11's
addendum raised its own seed count from 3 to 8 and materially changed the
picture) has not been tried and could plausibly sharpen or reverse this
reading, per that exact precedent.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

### Experiment 27 addendum — raised to 8 seeds: the non-replication holds, and sharpens rather than reverses

**Why.** Experiment 11's addendum (raising the same script's seed count
from 3 to 8) turned a weak, ambiguous correlation into a clear positive
one — direct precedent for not treating Experiment 27's 3-seed null
result as final. Reran both `run_atlas_nn_stage_c_lite_budget_search.py`
(to get matching `compression_gain` values for 5 new seeds) and
`analyze_stage_c_lite_delta_rank.py` at the full 8-seed set
(11/22/33/44/55/66/77/88).

**Result — unlike Experiment 11's precedent, more seeds did not reveal a
hidden relationship; if anything, the non-relationship sharpened
(n=56, pooled across all 8 seeds × 7 layers):**

| scope | n (3 seeds → 8 seeds) | Pearson r (3 seeds → 8 seeds) |
|---|---|---|
| overall (pooled) | 21 → 56 | −0.15 → **−0.06** |
| block 0 | 9 → 24 | +0.16 → +0.16 (unchanged) |
| block 1 | 9 → 24 | −0.15 → **+0.15** (flipped to the wrong direction) |
| classifier head | 3 → 8 | 0.95 → 0.74 (still an artifact — delta-rank fraction constant at 0.500) |

Block 1, the one scope that had shown a weak result in the *right*
direction at 3 seeds, flips sign at 8 — the opposite of what happened
when this same seed-count increase resolved Experiment 10/11's
ambiguity. The sublayer-type breakdown continues to point away from the
MLP's prediction: `out_proj` still has both the lowest delta-rank
fraction (0.328, most concentrated) *and* the lowest mean gain (2.508) —
now confirmed at nearly double the sample size, not a fluke of the
original 3 seeds.

**Interpretation.** This settles the question this addendum set out to
answer: Experiment 27's non-replication was not a small-sample artifact.
Delta-rank fraction's strong MLP correlation (Experiments 25–26, r=−0.75)
genuinely does not transfer to the Stage C-lite Transformer, at good
statistical power now. This is a real, stable architectural difference,
not noise — worth treating with the same confidence the project affords
its positive findings, per mission section 11.

**Why this cuts differently from Experiment 11's precedent, and that's
fine.** Experiment 11's addendum was cited as a reason to double-check
rather than as a prediction of what would happen — more seeds resolving
ambiguity in one direction there doesn't mean they always will. Here,
more seeds resolved the ambiguity in the *other* direction (toward
"genuinely doesn't transfer" rather than "secretly does"), which is
exactly as informative and exactly as valid a use of the same
falsification discipline.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.

---

## Experiment 28 — Delta-rank fraction on real pretrained models: a third independent measure splitting distilgpt2 from gpt2/gpt2-medium, and evidence Experiment 27's non-replication was about Stage C-lite specifically, not Transformers in general

**Context.** Experiment 26 found delta-rank fraction (effective rank of
`W_trained − W_random`) is the strongest mechanism correlation in the
project on the from-scratch MLP (r=−0.75). Experiment 27 (+ its 8-seed
addendum) found it does not transfer to the from-scratch Stage C-lite
Transformer (pooled r=−0.06). Real pretrained models are architecturally
Transformers too, but trained completely differently (real data, an
unknown but almost certainly much larger number of steps, real
optimizer schedules) — this experiment asks whether Experiment 27's
non-replication is about "Transformers in general" or something specific
to Stage C-lite's small-scale, short, from-scratch training.

**Method.** `experiments/analyze_stage_c_real_delta_rank.py`. Cheap by
design like Experiment 24: reuses `compression_gain` already computed by
the Experiment 19/21/23 budget searches, computing delta-rank fraction
(`W_pretrained − W_random`, as a fraction of max rank) via one more SVD
per already-tested layer of each already-downloaded model (distilgpt2,
gpt2, gpt2-medium), 3 random-init seeds each. `results/
atlas_nn_stage_c_real_delta_rank.json`.

**Result — a real relationship on the two non-distilled models, none on
the distilled one (per-model, n=18 each — 6 layers × 3 seeds):**

| model | Pearson r | Spearman r |
|---|---|---|
| distilgpt2 (distilled) | +0.19 (weak, no clear relationship) | +0.05 |
| gpt2 (non-distilled) | **−0.90** | **−0.79** |
| gpt2-medium (non-distilled) | **−0.60** | −0.25 |

Read the sign correctly, as in Experiments 25–27: negative means low
delta-rank fraction (concentrated update) predicts *higher* gain — the
same direction as the MLP's finding. gpt2's r=−0.90 is the **strongest
correlation of any kind found anywhere in this project**, stronger even
than Experiment 26's own MLP result (r=−0.75) that this experiment set
out to test. gpt2-medium shows the same direction, more moderately.
distilgpt2 shows no clear relationship at all.

**A third independent measure now splits the same three models the same
way.** Experiments 21/23 found this exact split in achievable-ratio
depth-gradient shape; Experiment 24 found it in final-matrix
effective-rank correlation (gpt2 r=0.83, gpt2-medium r=0.49, distilgpt2
r=−0.29); this experiment finds it again in delta-rank fraction
(gpt2 r=−0.90, gpt2-medium r=−0.60, distilgpt2 r=+0.19 null). Three
unrelated analyses, three times the same two non-distilled models agree
with each other and the one distilled model disagrees — a steadily
accumulating case that training procedure (distillation), not model
scale, is the real variable behind Experiments 21–24's original
divergence.

**Resolves part of Experiment 27's open question.** Experiment 27 could
not distinguish "delta-rank fraction doesn't work on Transformers in
general" from "it doesn't work on Stage C-lite's small-scale, short,
from-scratch training specifically." This experiment favors the second
reading: real, thoroughly-pretrained, non-distilled Transformers (gpt2,
gpt2-medium) show a *strong* delta-rank relationship — stronger than the
MLP's own, in gpt2's case — while Stage C-lite's toy-scale Transformer
showed none. Distillation (distilgpt2) appears to disrupt the
relationship specifically, independent of whether the underlying
architecture can support it.

**A power caveat, carried over from Experiment 24's identical situation
and equally real here.** Delta-rank fraction is nearly constant across
the 3 random-init seeds for a given layer (e.g. distilgpt2's
`h.0.attn.c_proj`: 0.721, 0.721, 0.721) — the 3 seeds are close to
redundant measurements of the same 6 layers, not 18 independent points,
so each model's *effective* n is closer to 6 than 18. This also means
the per-block breakdowns computed alongside the headline numbers (some
showing implausible correlations like r=−0.9999, fit through what is
effectively 3 real data points tripled) are **not meaningful and are
excluded from interpretation** — the per-model, all-6-layers numbers
above are the trustworthy level of this analysis.

**Interpretation.** This is now the third time three independent
analyses have converged on the same distilled/non-distilled split,
making it substantially harder to attribute to chance or to any single
measurement's idiosyncrasies. It also usefully narrows Experiment 27's
open question: the delta-rank mechanism is not simply "an MLP thing that
doesn't generalize to attention" — it appears on real, well-trained
Transformers, just not on Stage C-lite's small toy version, and not on
distilled models regardless of scale.

**Scope of the claim.** Three models, 6-of-many layers per model (same
subset as Experiments 19/21/23), correlational not causal, the
effective-n caveat above. Does not establish *why* distillation
disrupts the relationship, only that it does, now on three independent
measures.

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.
