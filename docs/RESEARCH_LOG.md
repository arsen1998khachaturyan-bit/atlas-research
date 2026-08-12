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

**Next experiment.** See `docs/NEXT_RESEARCH_DECISION.md`.
