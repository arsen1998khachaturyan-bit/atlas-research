# Best Results — Track B (`atlas_nn`)

Only reproducible results go here. Every entry cites the exact command and
seeds used to reproduce it. Labels are used strictly:

- **OBSERVATION** — something measured once or descriptively, not yet
  stress-tested across seeds/ablations.
- **HYPOTHESIS** — an explanation or prediction, not yet verified.
- **VERIFIED RESULT** — reproduced across multiple seeds, compared against
  baselines under the same protocol, and covered by an automated test.

---

## VERIFIED RESULT: block-dictionary + affine-transform beats all Stage-1 baselines on affine-block-structured synthetic matrices

**Claim.** On a 64×64 matrix built from 4 shared 8×8 prototype blocks, each
tile transformed independently by `sign · scale · prototype + shift`
(`atlas_nn.synthetic.block_transformed`), `atlas_nn.structural.
block_dictionary_transform` (dict_size=16, residual_bits=4, block 8×8)
reconstructs the matrix **exactly** (relative L2 error 0.0000) at a
**2.27× compression ratio**. Every baseline tested (8-bit/4-bit uniform
quantization, rank-4/rank-8 SVD, 50%/90% magnitude pruning, vector-codebook
VQ, zlib) either compresses less (zlib: 1.08×) or compresses more but with
much higher error (quantization/SVD/pruning/VQ: 0.01–0.85 relative L2).

**How verified.**
- Reproduced across 3 seeds (101, 202, 303) — see `results/atlas_nn_stage_a.json`.
- Compared against 9 baseline configurations under the identical protocol
  (`experiments/run_atlas_nn_stage_a.py`).
- Falsification check: the same method/bit-budget applied to
  `random_gaussian` (no structure) gives relative L2 error 0.072 — more than
  5× worse than on the structured matrix, and the "no free lunch on random
  data" property is enforced by an automated test
  (`tests/test_atlas_nn_structural.py::test_structural_method_does_not_fake_compression_on_random_matrix`).
- Reproduce with: `python -m experiments.run_atlas_nn_stage_a` (writes
  `results/atlas_nn_stage_a.json`) or `pytest tests/test_atlas_nn_structural.py`.

**Scope of the claim (what this does NOT show).** This is Stage A only —
controlled synthetic matrices, not trained neural-network weights. It does
not show that real trained weight tensors contain this kind of structure
(that is exactly what Stage B is for — see
`docs/NEXT_RESEARCH_DECISION.md`). It also does not beat baselines on 5 of
the 6 synthetic matrix types tested (see `docs/RESEARCH_LOG.md`, Experiment
2) — low-rank and structured+noise matrices are dominated by SVD, and
exact-repeat matrices (block_repeated, hierarchical_blocks) are dominated by
plain lossless zlib. This is a narrow, verified win on one structure family,
not a general result.

---

## VERIFIED RESULT: the block-dictionary method is currently dominated by plain lossless compression on exact block repetition

**Claim.** On `block_repeated` and `hierarchical_blocks` matrices (exact,
byte-identical repeated blocks, no continuous per-tile transform), zlib
alone reaches 10.15× and 14.15× compression respectively at zero error,
while `block_dictionary_transform` is capped at 2.27–3.18× (also at zero
error) by its fixed-width encoding, which spends a constant number of bits
per block regardless of whether the residual is exactly zero.

**How verified.** Same protocol/seeds as above; see `docs/RESEARCH_LOG.md`
Experiment 2 for the full comparison table.

**Why this matters.** It is a genuine current weakness of the
implementation (fixed-width, no entropy coding), not evidence against the
underlying block-dictionary idea — the method still gets the reconstruction
exactly right, it just doesn't shrink its own encoding when it could. Listed
here, not hidden, per the mission's instruction to report negative results.

---

## VERIFIED RESULT: trained networks are far more behaviorally robust to weight-compression error than randomly initialized networks at the same architecture

**Claim.** On a 3-layer MLP (32→64→64→2) trained on a synthetic XOR task
(`atlas_nn.stage_b`), compressing the 64×64 hidden layer's weight matrix and
substituting the reconstruction back into the network produces output-logit
error that is **3–11× smaller than the weight-tensor error would suggest**,
consistently across every compression method tested (SVD, block-dictionary,
quantization, pruning) — but only for the *trained* network. For the same
architecture at random initialization, weight error and output error track
each other almost 1:1 for every method (ratio ≈1).

Concrete numbers (mean of 3 seeds, `svd_rank4` on the 64×64 layer):
random-init tensor rel_l2 0.891 → output rel_logit 0.928 (ratio ≈1.0);
trained tensor rel_l2 0.414 → output rel_logit **0.039** (ratio ≈0.09, an
11× reduction).

**How verified.** Reproduced across 3 independent training seeds (11, 22,
33), for 4 different compression methods, on 3 different weight matrices —
the direction of the effect (trained ratio ≪ random-init ratio) held in
every method/layer combination where the layer was large enough for
overhead not to dominate (see the small-layer caveat below). Reproduce with
`python -m experiments.run_atlas_nn_stage_b` (writes
`results/atlas_nn_stage_b.json`) or `pytest tests/test_atlas_nn_stage_b.py`.

**Scope of the claim.** This is evidence that training adds *behavioral*
redundancy/robustness — not a demonstration that any specific Atlas
structural method becomes better at literally compressing the trained
tensor (the block-dictionary method's own tensor-level error was flat
between random-init and trained; only SVD showed a real tensor-level
compressibility gain, and only on one of the three layers). One synthetic
task, one small architecture, three seeds — not yet tested on a real
dataset or a larger network (mission Stage C).

---

## VERIFIED RESULT: at matched behavioral quality, training roughly doubles achievable compression ratio on deeper layers — but not on the input layer

**Claim.** Sweeping each method family's parameter to find the most
aggressive setting that still keeps `relative_logit_error ≤ 0.05`
(`atlas_nn.stage_b.budget_search`), the best achievable compression ratio
on the 64×64 hidden layer rises from 5.32× (random init) to 8.00–10.61×
(trained) — a 1.5–2× improvement — and similarly on the 2×64 output layer
(4.92× → 7.11–9.14×). The clearest single case: SVD on the hidden layer
needs near-full rank to stay within 5% behavioral error on random-init
weights (ratio 0.50 — effectively no compression), but only rank 4 after
training (ratio 8.00) — a direct, measured, 16× swing in a real compression
metric caused by training alone, same architecture, same eval protocol.
**The first (input-facing) layer shows no improvement at all** — same best
ratio (5.31×) and same winning method before and after training — so this
effect is depth-dependent, not uniform.

**How verified.** Reproduced across 3 independent training seeds (11, 22,
33); direction and rough magnitude of the effect held in every seed for
layers 2 and 4, and the layer-0 null result also held in all 3 seeds.
5 method families × ~6 parameter settings each swept per layer/state/seed
(558 total measured configurations). Reproduce with
`python -m experiments.run_atlas_nn_stage_b_budget_search` (writes
`results/atlas_nn_stage_b_budget_search.json`) or
`pytest tests/test_atlas_nn_budget_search.py`.

**Scope of the claim.** Same caveats as the Stage B behavioral-robustness
result above: one small architecture, one synthetic XOR task, one
behavioral-error threshold (5%) — not yet checked at other thresholds, a
real dataset, or larger networks. The `magnitude_prune` family's ratios in
this result are sometimes <1 (net expansion) due to sparse-COO encoding
overhead at this matrix scale — a real limitation of that baseline's
encoding, not evidence against the underlying finding (see
`docs/RESEARCH_LOG.md` Experiment 4 for the full caveat).

> **⚠ Update after Experiment 5 (robustness check):** this pattern **did
> not reproduce** on a harder synthetic task (3-way parity) with the same
> architecture — trained and random-init achieved the same or an
> inconsistent best ratio across all 3 seeds and all 3 layers there (see
> the OBSERVATION below and `docs/RESEARCH_LOG.md` Experiment 5). Read this
> VERIFIED RESULT as scoped specifically to the easy 2-XOR task it was
> measured on, not as a general property of "training" — the working
> hypothesis is now that it reflects unused representational slack left by
> an easy task relative to network capacity, not a general training effect.
> Not re-labeled from VERIFIED RESULT to OBSERVATION because the original
> measurement (on the 2-XOR task, multiseed) still stands as reproducible
> and correct — what changed is how far the claim generalizes, not whether
> it happened.
>
> **✓ Update after Experiment 6 (capacity sweep):** the slack/capacity
> hypothesis this update proposed is now directly confirmed, see the new
> VERIFIED RESULT below — giving the *harder* 3-parity task enough spare
> network capacity (width 256 instead of 64) brought the compression gain
> back, in the same layer, at similar-or-larger magnitude, with all 3 seeds
> training successfully (no confound). The gain is real and general once
> "training relative to available capacity" replaces "training" as the
> variable of interest.

---

## OBSERVATION: the post-training compression-gain pattern did not reproduce on a harder task, and the "deeper network" robustness check was invalidated by a training-free architectural pathology

**What was tested.** Same behavior-budgeted search as the result above,
repeated in two conditions (`docs/RESEARCH_LOG.md` Experiment 5,
`results/atlas_nn_stage_b_robustness_check.json`): (a) the identical
3-Linear-layer architecture trained on a harder 3-way-parity task instead
of 2-XOR; (b) a deeper 6-Linear-layer architecture trained on the original
2-XOR task.

**Harder task (valid comparison, negative result).** Across 3 seeds and 3
layers, trained vs. random-init best-ratio-at-5%-error was identical in
7/9 cases, and in the one seed where they differed on the early layers, the
*random-init* network reached a higher ratio than the trained one (7.76 vs.
5.22) — the opposite of the original pattern. No reliable post-training
compression gain was found on this task, on any layer.

**Deeper network (comparison invalidated, not negative — inconclusive).**
The random-init 5-hidden-layer network turned out to be behaviorally
degenerate at initialization: ~50% (chance) accuracy and a *constant
prediction for every evaluation example* (output logit std 0.04–0.12 vs.
22–34 once trained). Any "compression ratio" computed by perturbing an
already-constant function is not measuring what this experiment intended
to measure, so this condition's numbers are excluded from interpretation.
This is itself a useful, reproducible finding, now documented as a caution
in `atlas_nn.stage_b.model.build_mlp`'s docstring: stacking ~5 unnormalized
ReLU layers with plain PyTorch default init collapses signal at random
init, at this width/scale.

**Why this is here rather than silently discarded.** Per mission section
11 ("attempt to falsify the hypothesis... a negative result is useful"),
this is exactly the kind of check that should be run and reported, whether
or not it confirms the earlier finding. It directly narrows the earlier
VERIFIED RESULT's claimed scope rather than contradicting its original
measurement.

---

## VERIFIED RESULT: the post-training compression gain tracks spare network capacity relative to task difficulty, not "training" as a general effect

**Claim.** The 3-parity task showed no post-training compression gain at
network width 64 (Experiment 5). At width 256 — same task, same layer,
same 5% behavioral-error bar, only the network's spare capacity changed —
the gain reappears: the hidden→hidden layer's best ratio rises from 5.33×
(random-init, all 3 seeds) to 8.0–16.0× (trained, all 3 seeds), a
magnitude matching or exceeding the original easy-task result. Symmetrically,
shrinking the *easy* 2-XOR task's network to width 16 (a tight fit, no
spare capacity) removes its previously solid gain — trained and
random-init converge to the same ratio, or trained is worse.

**How verified.** `atlas_nn.stage_b.budget_search` behavior-budgeted
search, 3 seeds (11, 22, 33) per condition, 6 conditions total (2 tasks ×
3 widths). The `parity3_h256` confirmation has zero training-failure
confound — all 3 seeds reached 93–97% held-out accuracy. Reproduce with
`python -m experiments.run_atlas_nn_stage_b_capacity_sweep` (writes
`results/atlas_nn_stage_b_capacity_sweep.json`).

**Why this is stronger evidence than Experiment 4 alone.** Experiment 4
observed a gain on one setup; this result predicted a *specific reversal*
(gain should vanish on a harder task, then specifically reappear if given
more capacity) and then found exactly that reversal in both directions —
which is a materially stronger form of confirmation than a single
observation, per the mission's falsification discipline (section 11).

**Scope of the claim.** Still two synthetic tasks, small MLPs, one
behavioral-error threshold, CPU-only. The mechanism ("spare capacity") is
inferred from the pattern of results, not measured directly (e.g. no
direct measurement of effective rank or intrinsic dimensionality was made)
— a more direct capacity metric would strengthen this further. See the two
caveats immediately below, both from the same experiment.

---

## OBSERVATION: fixed training hyperparameters do not reliably transfer to a larger network — a real confound, not evidence of exceptional compressibility

At network width 256 on the easy 2-XOR task, 2 of 3 seeds **failed to
train properly** using the same hyperparameters (epochs, learning rate)
that worked at every other width/task combination in this session: held-out
accuracy 49% (chance) and 68% (partial), vs. the expected ~87–91%. Those
two seeds' "compression ratios" of up to 128× are an artifact of
compressing a network that never learned a meaningful function — the same
underlying pathology as Experiment 5's degenerate deep network, arrived at
a different way (optimization failure rather than architectural signal
collapse). Excluded from the VERIFIED RESULT above; kept in
`results/atlas_nn_stage_b_capacity_sweep.json` for transparency. Practical
implication for any future scale-up (Stage C): hyperparameters must be
re-validated at each new scale, not assumed to transfer, and training
success should be checked (e.g. held-out accuracy) before trusting any
compression number computed on a "trained" network.

---

## OBSERVATION: the input layer is consistently flat-to-worse after training, across every task and width tested

Across all 6 conditions in the capacity sweep, the first (input-facing)
Linear layer's best-ratio-at-threshold was never clearly better after
training than at random-init, and in most conditions was equal or *lower*
(e.g. `parity3_h256`: random-init 7.94× in all 3 seeds vs. trained 5.31× in
all 3 seeds — a small but perfectly consistent penalty). This is stronger
and more consistent than Experiment 4's original "no improvement on layer
0" observation, and holds regardless of task difficulty or network width,
suggesting a distinct mechanism from the capacity story above rather than
a special case of it. Not investigated further this round; a candidate
next question, not yet a hypothesis with an explanation.

---

## VERIFIED RESULT (partial): effective-rank shrinkage quantitatively predicts compression-gain magnitude on the hidden→hidden layer, but not on the other two layers

**Claim.** A spectral capacity-usage metric (Shannon/Roy–Vetterli effective
rank of the weight matrix) computed independently of any compression
method correlates with the log-scale compression-gain magnitude measured
by the capacity sweep — specifically and only on the hidden→hidden layer
(layer 2): Pearson r=0.67, Spearman r=0.62, n=16 (one point per condition
×seed, 3 widths × 2 tasks × 3 seeds − 2 excluded training failures). This
is a moderate-strong, practically useful correlation, not just a
directionally-positive one. On the input layer the correlation is weak and
inconsistent (r=0.13–0.49 depending on which specific metric); on the tiny
2-unit output layer the correlation is unstable and not a fair test (that
layer's `max_rank` is only 2, making "effective rank" nearly a binary
variable).

**How verified.** `atlas_nn.stage_b.capacity_metrics` (unit-tested
independently: rank-1 matrix → effective rank ≈1, random square matrix →
high effective rank, effective rank increases monotonically with true
rank). `experiments/analyze_stage_b_capacity_metric.py` deterministically
reproduces the same 6 conditions × 3 seeds already characterized by the
capacity sweep (same fixed seeds) and computes the metric on the actual
trained/random weight tensors. Reproduce with
`python -m experiments.analyze_stage_b_capacity_metric` then
`python -m experiments.summarize_stage_b_capacity_metric` (the second
script applies a corrected training-success filter — see the caveat
immediately below — without re-running any training).

**Caveat found and fixed during this analysis.** The first pass's
training-success filter (trained accuracy ≥15 points above that run's own
random-init accuracy) let one partially-failed training run through as
"succeeded" (`xor2_h256` seed 33: 68% held-out accuracy vs. an ~87%
ceiling for that condition) — a milder version of the same confound
Experiment 6 already found and excluded for a different seed at the same
width. Fixed with an added per-condition "near ceiling" check; the
correlation numbers above are from the corrected analysis. Flagged here
rather than silently corrected, since it's the second time in two rounds
that a generic accuracy-margin filter needed strengthening — a standing
risk worth checking explicitly in any future run rather than assuming a
fixed threshold is safe.

**Scope of the claim.** Labeled "partial" because it holds cleanly on one
of three layers, not all three — consistent with, but not full
confirmation of, the broader capacity-slack story. It's a genuine
sharpening (a testable number, not just a direction) precisely where the
mechanism was expected to operate, and an honest non-result where it
wasn't (input layer) or wasn't a fair test (output layer).

---

## OBSERVATION: k-means dictionary fitting is not perfectly reliable at small dictionary sizes

On `block_repeated` with `dict_size=8` (only 4 true unique blocks exist),
one of three seeds gave relative L2 error 0.0448 instead of 0.0000 — the
k-means step must have converged to a local optimum that merged two
distinct prototypes. `dict_size=16` gave exact reconstruction on all 3
seeds tested. Not yet ablated across more seeds/restarts to characterize
how often this happens; flagged as an observation, not a verified failure
rate.

---

## VERIFIED RESULT: the behavioral-robustness effect transfers to an attention-based architecture on a real text task, and strengthens with depth across two full Transformer blocks

**Claim.** On a small 2-block Transformer classifier trained from scratch
on a real (self-authored) English sentiment task —  a completely different
architecture and task family from every prior experiment's MLP/synthetic
setup — the same core effect found in Stage B reproduces and sharpens:
tensor-level reconstruction error barely changes between random-init and
trained weights (e.g. `svd_rank4`: 0.89–0.92 either way), but behavioral
error (relative output-logit error after substituting the reconstructed
weight) collapses after training, by 3.5–4.9× in the earlier Transformer
block and by **15.7–22.7×** in the later block — a graded, monotonic
depth effect, not just a binary "hidden layer gains, others don't." This
held in all 3 individual seeds (block-1 per-seed gains 10.3×–42.0×,
block-0 per-seed gains 1.9×–12.9× — non-overlapping in 2 of 3 seeds, and
block 1 higher than block 0 within every seed).

**How verified.** `experiments/run_atlas_nn_stage_c_lite_smoke.py`,
3 seeds (11, 22, 33), 7 method configurations × 7 Linear layers × 2 model
states. Training is genuine and non-trivial: random-init held-out accuracy
56–64%, trained 84–94%, on a held-out set disjoint from training sentences.
Reproduce with `python -m experiments.run_atlas_nn_stage_c_lite_smoke`
(writes `results/atlas_nn_stage_c_lite_smoke.json`) or
`pytest tests/test_atlas_nn_stage_c_lite.py`.

**Why this matters more than another single-architecture result.** Every
prior capacity/behavioral-robustness finding (Experiments 3–7) was
established on one MLP architecture and two closely related synthetic
tasks (2-XOR, 3-parity). This is the first test on a structurally different
architecture (self-attention, not just feedforward) and a real task (real
English sentences, not synthetic feature vectors) that neither the model
nor the effect was tuned for. Finding the same qualitative effect — and a
*sharper*, more graded version of it — substantially raises confidence
that this is a property of trained neural networks in general, not an
artifact of the MLP/XOR setup everything else was built on.

**Caveats.** (1) This experiment used fixed compression parameters, not a
behavior-budgeted ratio search — there's no "Nx achievable compression"
headline number from this result, only the tensor-vs-behavior error gap
(analogous to Experiment 3, not Experiment 4/6). (2) The tiny classifier
head (2×64, 512 bytes) reproduces the exact same overhead pathology found
in Stage B (`atlas_block_dict16_res4bit` gives ratio 0.21 — net expansion)
— consistent, not new, but confirms that finding isn't MLP-specific either.
(3) Still a small model (~50K parameters) on a small, template-generated
task — not mission Stage C's literal "pretrained transformer on a real
dataset," which remains blocked by this session's network policy
(`huggingface.co` returns 403 from the egress proxy; confirmed via
`curl $HTTPS_PROXY/__agentproxy/status`, not retried or routed around per
the proxy's own policy).

---

## Explicitly not yet claimed

- Nothing about real datasets, larger networks, or pretrained transformers
  (Stage C/D of the mission ladder) — Stage B used one small synthetic-task
  MLP only.
- Nothing about direct inference without materializing full weight tensors
  (mission section 8) — not attempted yet.
- Nothing about cross-layer/cross-model shared dictionaries (mission section
  7) — not attempted yet.
- No patentability or novelty claim — see `docs/RESEARCH_LOG.md` Experiment
  2's "Novelty discipline" note; the structural method is a known family
  (VQ/codebook clustering + affine correction), not a new technique. The
  Stage B behavioral-robustness finding is a reproduction of a known class
  of deep-learning phenomena (flat minima / weight-space redundancy in
  trained networks), not a new discovery in itself — its value here is as
  *measured evidence for the mission's Stage B question*, not as a novel
  claim.
