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

> **⚠ Update after Experiment 10 (Transformer capacity-metric check):**
> this positive correlation did **not** transfer to the Stage C-lite
> Transformer, despite the underlying compression/behavioral-robustness
> effect itself transferring cleanly (Experiments 8–9). On the
> Transformer, rank shrinkage vs. compression gain trends **negative**
> (block 1: r=−0.54, the layer type with the *largest* gains) — see the
> OBSERVATION below. Effective rank of a layer's own weight matrix is
> therefore not a general explanation for the capacity-slack effect
> across architectures; it may be specific to feedforward layers without
> a residual/normalization pathway. This entry's original MLP measurement
> still stands as reproducible and correct on the MLP — what's now known
> not to hold is generalizing it as "the" mechanism.

---

## OBSERVATION: effective rank does NOT predict the Transformer's compression gain — a genuine divergence from the MLP

Repeating Experiment 7's analysis on the Stage C-lite Transformer
(`experiments/analyze_stage_c_lite_capacity_metric.py`, same 3 seeds as
Experiments 8–9) found no positive relationship between effective-rank
shrinkage and compression-gain magnitude — overall r=−0.25 (n=21), and
specifically **r=−0.54** on block 1 (n=9), the exact layer type showing
the *largest* compression gains in Experiment 9. This is the opposite
sign from the MLP's r≈+0.67. A more basic divergence underlies it: the
Transformer's effective rank barely shrinks with training at all (FFN
layers: ~92–93% of max rank before training, ~91–92% after — a 1–2%
relative change) even though its compression/robustness gains are as
large or larger than the MLP's, where the best-correlated layer showed up
to 44% relative rank shrinkage. Working (unverified) hypothesis: the
Transformer's residual connections and layer normalization may provide an
error-absorbing pathway that makes a sublayer's output robust to
perturbation independent of that sublayer's own weight-matrix rank — i.e.
the robustness may live in architectural connectivity, not in any single
layer's spectral structure. Statistical power is limited (n=9 per block,
several exactly-repeated `compression_gain` values from
`run_budget_search`'s coarse discrete grid) — the qualitative
"no positive relationship, unlike the MLP" finding is the load-bearing
result here, not the precise r values. Reported per mission section 11:
a negative result is exactly as useful as a positive one.

> **⚠ Update after Experiment 11 (residual/LayerNorm ablation, first
> pass, 3 seeds):** the specific r=−0.54 above should be read with
> additional caution — a same-condition rerun with a different (but
> conceptually equivalent) implementation gave r=+0.33 for block 1, not a
> matching negative value. At n≈9 per block, not stable across
> reimplementations.
>
> **⚠⚠ Second update, after raising Experiment 11 to 8 seeds (n=49–56
> pooled):** the corrected picture is now clear enough to state
> positively rather than just as a caution. Rank-shrinkage correlates
> **positively** with the Transformer's gain in all 4 ablation conditions
> tested (pooled Pearson r = 0.23 to 0.55, see the VERIFIED RESULT
> below) — the original r=−0.54 was a small-sample artifact, not a real
> negative effect. Effective rank *does* relate to the Transformer's gain
> after all, just more weakly (r≈0.2–0.3) than the MLP's r≈0.67 when
> LayerNorm is present, and moderately (r≈0.4–0.55) when it isn't. This
> supersedes both this entry's original claim and the first update above.

---

## VERIFIED RESULT: LayerNorm drives the magnitude of the Transformer's robustness gain and weakens its rank-shrinkage correlation; the depth gradient is independent of both

**Claim.** Across two independent seed counts (3 seeds, then 8 seeds; the
mean-gain numbers agreed within 15% between runs), training a Stage
C-lite Transformer variant with LayerNorm enabled produces 2–4× larger
behavioral-robustness gains than the same architecture with LayerNorm
disabled (pooled means: 16.3 and 18.4 with LayerNorm vs. 8.1 and 4.1
without, at 8 seeds), while removing residual connections alone changes
the gain by less than 15% (16.3→18.4, a slight *increase*, not a
decrease) — the opposite of what an original "residual connections as the
error-absorbing pathway" hypothesis predicted. Separately, removing
LayerNorm consistently **raises** the effective-rank-shrinkage
correlation with gain (0.23→0.55 holding residual connections on; 0.29→
0.37 holding them off) — i.e. LayerNorm doesn't just amplify the gain,
it also decouples it from the layer's own weight-matrix rank structure.
**The block-1-gains-more-than-block-0 depth gradient (Experiment 8)
survives all 4 conditions**, including with both features removed
(4.0×–9.4× at 8 seeds) — it is not explained by either ablated feature.

**How verified.** `experiments/run_atlas_nn_stage_c_lite_residual_ablation.py`,
rerun at 8 seeds after an initial 3-seed pass showed the same direction;
n=49–56 pooled per condition (7 layers × 8 seeds, minus rare training
failures — one seed failed under `no_residual` specifically, itself a
minor data point re: residual connections' effect on optimization
stability). Reproduce with
`python -m experiments.run_atlas_nn_stage_c_lite_residual_ablation`
(writes `results/atlas_nn_stage_c_lite_residual_ablation.json`).

**Scope.** One small architecture, one real-but-simple task, block-level
(rather than pooled) correlations are noisier (n=21–24, individual block
r ranging −0.22 to +0.64) and shouldn't be read as precisely as the
pooled numbers. The *why* of the depth gradient itself remains
unexplained — this result narrows candidate mechanisms (rules out
residual connections as the primary driver, implicates LayerNorm for
magnitude specifically) without fully resolving the question.

> **⚠ Update after Experiment 13 (LayerNorm added to the MLP):** the
> *magnitude* half of this claim ("LayerNorm amplifies the gain") does
> **not** generalize to the MLP — adding LayerNorm there *shrinks* the
> gain by ~3.3× (9.7→2.9), the opposite sign from the Transformer. The
> *decoupling* half ("LayerNorm weakens the rank-shrinkage correlation")
> **does** generalize (MLP: 0.59→0.16, same direction as the
> Transformer's 0.55→0.23 / 0.37→0.29). Read this entry's magnitude claim
> as scoped to the Transformer specifically, not as a property of
> LayerNorm in general — see the new entry below for the full picture.

---

## VERIFIED RESULT: LayerNorm's rank-decoupling effect generalizes across architectures; its magnitude effect reverses sign

**Claim.** Adding `nn.LayerNorm` to the plain MLP (which otherwise has
none) reproduces one half of Experiment 11's Transformer finding and
inverts the other. Reproduced: LayerNorm presence weakens the
relationship between a layer's rank shrinkage and its post-training gain
in **both** architectures (MLP: Pearson r 0.59→0.16 with LayerNorm added;
Transformer: 0.55→0.23 and 0.37→0.29 with LayerNorm *removed* — same
direction, LayerNorm present ⇒ weaker correlation, either way). Reversed:
LayerNorm **shrinks** the MLP's gain magnitude by ~3.3× (9.7→2.9) where
it had **amplified** the Transformer's by 2–4×. Both effects are
concentrated in the layer where each architecture's capacity-slack
mechanism already lives (the MLP's hidden layer: gain 25.3→5.1, rank
shrinkage 0.36→0.19 with LayerNorm added; the input and output layers are
essentially unaffected either way).

**How verified.** `experiments/run_atlas_nn_stage_b_layernorm_ablation.py`,
8 seeds per condition (matching Experiment 11's final power from the
start), same 2-XOR task and width as the original Stage B experiments.
Reproduce with
`python -m experiments.run_atlas_nn_stage_b_layernorm_ablation` (writes
`results/atlas_nn_stage_b_layernorm_ablation.json`).

**Why this is more useful than either a clean confirmation or a clean
refutation would have been.** It splits a single bundled claim
("LayerNorm explains the Transformer's gain") into two genuinely separate
sub-claims with different scopes: rank-decoupling looks like a general
property of normalization; magnitude amplification looks specific to
something about the Transformer (attention is the untested remaining
candidate). Forces the next mechanism question to be more precise than
"is it LayerNorm" — see `docs/NEXT_RESEARCH_DECISION.md`.

**Scope.** One task (2-XOR), one width, 8 seeds — solid within this
setup, not yet tested on a harder MLP task or a different width, and the
underlying reason for the magnitude-sign reversal is not identified,
only located (concentrated in the hidden layer, absent from input/output).

> **⚠ Update after Experiment 14 (attention ablation):** the "untested
> remaining candidate" mentioned above (attention) has now been tested
> directly and confirmed as the dominant magnitude driver — bigger than
> LayerNorm's own contribution. See the new VERIFIED RESULT below for the
> complete picture: attention and LayerNorm turn out to drive two
> different aspects of the effect (magnitude vs. decoupling)
> semi-independently.

---

## VERIFIED RESULT: attention is the dominant driver of gain magnitude (and partly the depth gradient); LayerNorm remains the dominant driver of rank-decoupling

**Claim.** Removing attention from the Stage C-lite Transformer (`atlas_nn.
stage_c_lite.model.NoMixingAttention`, a same-shaped drop-in replacement
with no cross-token mixing) while keeping LayerNorm collapses the
post-training behavioral-robustness gain by **79%** (16.3→3.4, pooled
over 7 layers × 8 seeds, all seeds trained successfully) — a larger drop
than removing LayerNorm alone causes (16.3→8.1, 50%). **3.4 lands close
to Experiment 13's independently-measured MLP-with-LayerNorm result
(2.9)** — a Transformer with attention disabled behaves, in magnitude
terms, almost like the MLP that never had attention to begin with,
providing convergent (not just directional) confirmation. The
block-1-vs-block-0 depth gradient also shrinks substantially without
attention (6.9×→2.4×, vs. only 6.9×→4.0× when LayerNorm alone is
removed) — attention contributes to the depth gradient itself, unlike
residual connections (Experiment 11, no effect on the gradient) or
LayerNorm alone (smaller effect on the gradient than on magnitude).
Separately, the **rank-shrinkage correlation stays governed by LayerNorm**
regardless of attention (low whenever LayerNorm is present: 0.23, 0.02;
higher whenever absent: 0.55, 0.33) — attention's own removal, if
anything, pushes correlation slightly lower, not higher.

**How verified.** `experiments/run_atlas_nn_stage_c_lite_attention_ablation.py`,
4 conditions (attention × LayerNorm, residual held on), 8 seeds each, all
32 seed-condition combinations trained successfully (no exclusions
needed). Reproduce with
`python -m experiments.run_atlas_nn_stage_c_lite_attention_ablation`
(writes `results/atlas_nn_stage_c_lite_attention_ablation.json`).

**Why the convergence with Experiment 13 matters.** Two independent
routes to "a network with no attention" — an MLP that was never given
attention (Experiment 13), and a Transformer with attention explicitly
disabled (this experiment) — produced nearly identical gain magnitudes
(2.9 vs 3.4) under LayerNorm. Independent convergence like this is
stronger evidence than either measurement alone, and it resolves
Experiment 13's open question cleanly: the reason LayerNorm suppresses
the MLP's gain but amplifies the Transformer's is that the Transformer
has attention and the MLP doesn't — not some other unidentified
architectural difference.

**Current overall mechanism picture (as of this result):** attention
primarily drives gain *magnitude* and contributes to the *depth
gradient*; LayerNorm primarily drives *rank-decoupling* and contributes a
smaller, largely independent boost to magnitude. This is the most
complete and internally consistent account the project has produced,
though it stops short of explaining *why* attention and LayerNorm have
these specific effects mechanistically (only *that* they do, and by how
much).

**Scope.** One small Transformer, one real-but-simple task, 8 seeds — the
strongest single ablation result in the project so far by sample size and
convergent-evidence structure, but still one architecture family and one
task.

> **✓ Update after Experiment 17 (parameter-count disentanglement):**
> `NoMixingAttention` (used above) removed both mixing and ~3/4 of
> attention's parameters at once, leaving it ambiguous which mattered.
> Resolved: a variant with the exact same parameter count as real
> attention (verified: 16,640 either way) but still no mixing lands at
> 4.4 — close to the original no-mixing condition's 3.4, nowhere near
> full attention's 16.3. **Mixing, not parameter count, is confirmed as
> the operative factor.** See the new VERIFIED RESULT below.

---

## VERIFIED RESULT: attention's magnitude effect is specifically about cross-token mixing, not parameter count

**Claim.** A same-parameter-count, non-mixing replacement for attention
(`MatchedParamNoMixingAttention` — keeps `in_proj_weight`/`bias` as raw
Parameters exactly like real `nn.MultiheadAttention`, verified 16,640
parameters either way, but skips the softmax cross-token mixing step)
produces a mean gain of 4.4 — statistically indistinguishable from the
original parameter-poor no-mixing condition's 3.4, and nowhere near real
attention's 16.3. Restoring ~12,000 parameters while still withholding
mixing changed the outcome by less than one unit of gain, while
withholding mixing alone (with or without those parameters) cuts the
gain to roughly a quarter of full attention's.

**How verified.** `experiments/run_atlas_nn_stage_c_lite_attention_param_disentangle.py`,
3 conditions × 8 seeds, LayerNorm and residual connections held on
throughout. The script's `full_attention` and `no_mixing_no_param_match`
conditions exactly reproduce Experiment 14's numbers (16.3 and 3.4)
before the new `no_mixing_matched_params` condition is trusted. Reproduce
with `python -m experiments.run_atlas_nn_stage_c_lite_attention_param_disentangle`
(writes `results/atlas_nn_stage_c_lite_attention_param_disentangle.json`).

**Why this was a real test, not a foregone conclusion.** More parameters
generally means more capacity for useful structure to develop during
training — parameter count was a reasonable candidate explanation, not a
straw man. Finding that it explains almost none of the effect (0.9 of
the 12.9-point gap between full attention and no-mixing) is genuine
information, not a confirmation of the obvious.

**Current, most precise mechanism picture:** cross-token mixing
(attention specifically, not just "having an attention-shaped sublayer")
drives gain magnitude and part of the depth gradient; LayerNorm drives
rank-decoupling and a smaller, independent magnitude contribution. Four
architectural factors (rank, residual connections, LayerNorm, attention)
have now each been isolated to a specific, tested role.

**Scope.** One small Transformer, one task, 8 seeds. Explains *that*
mixing matters far more precisely than parameter count, not *why* mixing
specifically produces this effect at a mechanistic (e.g. information-
theoretic or optimization-dynamics) level.

---

## OBSERVATION: the MLP input layer's compressibility does not track task-irrelevant input noise fraction (a specific hypothesis ruled out, not confirmed)

**What was tested.** `experiments/run_atlas_nn_stage_b_input_layer_noise_sweep.py`:
2-XOR task, informative dimensions fixed at 2, noise fraction swept from
50% to 97% (`n_features` 4→64) at fixed network width, 5 seeds per
condition (more than the project's usual 3, given this question — open
since Experiment 4 — deserved better power). Measured the input layer's
achievable-compression-ratio gain (Experiment 4/6 methodology) at each
noise level.

**Result.** No trend supporting the hypothesis that more noise dimensions
create more input-layer compressibility slack: gain stayed at ≈1.0 or
below across the entire tested range (1.07, 1.00, 0.67, 1.00, 0.69 for
noise fractions 50%→97%), including two conditions with a real training-
induced *penalty*. A separate check at the zero-noise extreme
(`n_features=2`) showed a small, inconsistent positive gain (≈1.14×,
4 of 5 seeds positive) — a possible hint that very low noise fractions
behave differently, but underpowered and on an atypically small (64×2)
weight matrix, not confirmed as a real effect.

**Why this belongs here even though it's a clean negative.** It rules out
a specific, plausible, previously-untested explanation for the input
layer's now three-experiments-old unexplained behavior (Experiments 4, 6,
7), with real statistical power (25 measurements across 5 noise levels) —
narrowing, not just restating, the open question. Per mission section 11,
a negative result that closes off a hypothesis is exactly as valuable as
a positive one.

**Confound noted for anyone extending this.** Training data size was held
fixed while `n_features` varied, so higher-noise conditions were also
harder generalization problems with proportionally less data (trained
accuracy dropped from 98.4% to 73.6% across the sweep) — a cleaner version
would scale data with `n_features` to isolate noise fraction from task
difficulty.

> **✓ Update after Experiment 15 (controlled re-run):** the confound is
> fixed (8x more training data, held-out accuracy band tightened to
> 91.0%–99.6%) and the result **holds, more cleanly** — gain stays at
> 0.73–1.00 across the *entire* 0%–97% noise-fraction range at 8 seeds,
> including the zero-noise point now properly inside the main sweep. The
> earlier "possible hint" at zero noise (≈1.14×, 4/5 seeds, n=5) **did
> not replicate** — the controlled, better-powered measurement gives
> exactly 1.00 (flat) at that point. Read as a stronger, more definitive
> version of this OBSERVATION: the noise-fraction hypothesis is cleanly
> ruled out, not just weakly disfavored, and the input layer's behavior
> remains unexplained after four separate experiments now (4, 6, 7, 12,
> 15).

---

## OBSERVATION: the MLP input layer's compressibility is unaffected by whether it sees raw or pre-transformed input (a third hypothesis ruled out)

**What was tested.** `experiments/run_atlas_nn_stage_b_input_layer_raw_vs_projected.py`:
a frozen (never trained), orthogonally-initialized linear projection
prepended before the trainable stack, so the first trainable layer sees a
fixed transform of the raw input instead of raw input directly. 2-XOR
task, 8 seeds, `raw_input` vs. `projected_input`.

**Result.** No positive gain in either condition (mean gain 0.94 raw vs.
0.80 projected) — if anything slightly worse once the input is
projected, the opposite of the hypothesis's prediction. Rules out
"seeing raw, untransformed input" as the explanation for the input
layer's distinct behavior.

**Where this leaves the input-layer question.** Three specific hypotheses
now tested and ruled out: effective rank (Experiment 7, weak/inconsistent),
noise fraction (Experiments 12, 15, cleanly ruled out at good power), and
raw-vs-processed input (this experiment). The behavior — no post-training
compressibility gain, sometimes a penalty, unique among all layer types in
every architecture tested — remains real, reproducible across many
experiments, and still unexplained.

> **✓ Update after Experiment 25 (weight-delta rank analysis):** a
> fourth, structurally different kind of test — not another property of
> the *final* trained matrix, but of the training *update* itself — found
> a real, large, cleanly-separated candidate signal. See the new VERIFIED
> RESULT (partial) below.

---

## VERIFIED RESULT: the input layer's training-induced weight UPDATE is diffuse (near full-rank); the hidden layer's is concentrated (low-rank) and its degree of concentration predicts compression-gain magnitude better than any prior metric in this project

**Claim.** Measuring not the final trained matrix's rank (Experiment 7)
but the effective rank of `W_trained − W_random` itself, expressed as a
fraction of that layer's max possible rank: the input layer's update
uses 87.7% of its available rank directions (mean over 8 seeds, range
0.865–0.890) — close to full-rank, diffuse. The hidden layer's update
uses only 34.9% (range 0.278–0.437) — concentrated in a much smaller
effective subspace. **The two distributions never overlap across any of
the 8 seeds tested.** Movement *magnitude* alone (`‖Δ‖/‖W_random‖`) does
not separate the layers the same way (input: 2.19, hidden: 2.59, output:
4.23) — it's specifically the *structure* of the change, not its size,
that differs.

**How verified.** `experiments/run_atlas_nn_stage_b_weight_delta_analysis.py`,
the exact original Stage B setup (3-Linear 2-XOR MLP, Experiments 3/4/6/7)
at the project's 8-seed standard. Reproduce with
`python -m experiments.run_atlas_nn_stage_b_weight_delta_analysis`
(writes `results/atlas_nn_stage_b_weight_delta_analysis.json`).

**Why this is a genuinely new lead, not a repeat of Experiment 7.**
Experiment 7 measured the rank of the *final* trained matrix and found a
weak, inconsistent relationship on the input layer specifically. This
measures the rank of what training *changed* — a different quantity that
happens to separate the two layers far more cleanly than the final-matrix
measurement ever did, on the same architecture and task.

> **✓ Update after Experiment 26 (capacity-sweep cross-check):** the open
> question this entry originally flagged — whether delta-rank fraction
> predicts compression-gain *magnitude*, not just direction — is now
> answered: yes, and more strongly than any prior mechanism metric in
> this project. See the extended claim below.

**Why this is a genuinely new lead, not a repeat of Experiment 7.**
Experiment 7 measured the rank of the *final* trained matrix and found a
weak, inconsistent relationship on the input layer specifically. This
measures the rank of what training *changed* — a different quantity that
happens to separate the two layers far more cleanly than the final-matrix
measurement ever did, on the same architecture and task.

**Extended and strengthened by Experiment 26.** Reproducing Experiment
6's full capacity-sweep grid (2 tasks × 3 widths, 3 seeds each) and
correlating delta-rank fraction against `log(compression_gain)`
*within* the hidden layer specifically gives **Pearson r=−0.75,
Spearman r=−0.70** (n=16, after the same training-success filter
Experiment 7's analysis needed) — read the sign correctly: *low*
delta-rank fraction means a concentrated update, so a *negative*
correlation with gain means concentrated updates predict *higher* gain,
the same direction as the original cross-layer finding. This is
**stronger than Experiment 7's original final-matrix effective-rank
correlation (r≈0.67)** — the first time in this project a cross-check
has produced a larger effect than the finding it was verifying. A
visible, concrete pattern within the hard `parity3` task alone: as width
rises 16→64→256, the hidden layer's delta-rank fraction falls
0.60→0.39→0.19 while compression gain rises ≈0.7×→1.0×→2.1× — spare
capacity doesn't just leave the final weights more compressible
(Experiments 6–7), it produces a more concentrated training update in
lockstep, giving the capacity/slack story a mechanistic complement.
**Layer 0 (input) still shows no reliable within-layer relationship**
(r=0.18) — its delta-rank fraction stays uniformly high (0.642–0.928)
regardless of task or width, unlike the hidden layer's wide swing
(0.157–0.676): the input layer's update appears diffuse no matter how
much spare capacity exists elsewhere in the network.

**The pooling trap, again.** Pooling all three layers together gives a
misleadingly consistent-looking r=−0.58 that hides layer 0's real
non-relationship and layer 4's degenerate artifact — the same lesson
Experiment 7 and Experiment 24 already established, worth checking every
time before trusting a pooled correlation.

**How verified (Experiment 26 addition).**
`experiments/analyze_stage_b_delta_rank_capacity_sweep.py`, reproducing
Experiment 6's 6 conditions × 3 seeds. Reproduce with `python -m
experiments.analyze_stage_b_delta_rank_capacity_sweep` (writes
`results/atlas_nn_stage_b_delta_rank_capacity_sweep.json`).

**Scope of the claim.** Correlational, not causal — *why* the input
layer's update stays diffuse regardless of capacity is not established
(one unverified, untested story is offered in `docs/RESEARCH_LOG.md`
Experiment 25). One architecture family (the Stage B MLP), two tasks,
three widths, 3 seeds per condition. Not yet checked on the Transformer
or any real pretrained model. The output layer's own delta-rank-fraction
(exactly 0.500 in nearly every row, both experiments) remains flagged as
a likely artifact of that layer's degenerate 2-dimensional max rank (the
same caveat Experiment 7 raised), excluded from the headline claim in
both experiments.

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
(analogous to Experiment 3, not Experiment 4/6). **Resolved by Experiment
9, next entry below** — the budget-search version confirms the same
depth gradient as an actual achievable-ratio number. (2) The tiny classifier
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

## VERIFIED RESULT: achievable compression ratio at matched quality deepens with Transformer block depth, mirroring the MLP capacity pattern on a real task

**Claim.** At a fixed 5% behavioral-error bar, mean achievable compression
ratio (3 seeds) rises from 9.7–14.2× (random-init) to 14.2–42.7× (trained)
across the Stage C-lite Transformer's 7 Linear layers, and the *size* of
that gain grows with depth: block 0 layers gain 1.5×–3.1× from training,
block 1 layers gain 3.0×–4.4×, consistently across all three matched
layer-type pairs (attention-output, FFN-in, FFN-out). Trained block-1
layers reach ~32–43× compression at matched quality — more than 3× what
those same layers support pre-training, and more than block 0 reaches even
after training.

**How verified.** `experiments/run_atlas_nn_stage_c_lite_budget_search.py`,
same 3 seeds, same 5% `relative_logit_error` bar as Experiments 4/6.
Reproduce with `python -m experiments.run_atlas_nn_stage_c_lite_budget_search`
(writes `results/atlas_nn_stage_c_lite_budget_search.json`).

**Relationship to the previous entry.** This is the quantitative
(achievable-ratio) counterpart to the qualitative (tensor-vs-behavior-error)
finding immediately above — same experiment family, same model, same
seeds, converted from "how much does behavioral error drop" into "how much
more can we actually compress." SVD becomes the dominant winning method
for trained block-1 layers (8 of 9 cases), consistent with Experiment 7's
finding that SVD is the baseline most sensitive to training-induced
effective-rank changes — now observed on a second architecture.

**Caveat.** The exact gain multipliers here (1.5×–4.4×) are smaller than
Experiment 8's raw relative-logit-error gains (3.5×–22.7×) because
`run_budget_search` only evaluates a fixed, discrete parameter grid per
method (e.g. SVD ranks 1/2/4/8/16/32/64) — a real improvement that doesn't
cross the next grid point isn't reflected in the achievable ratio. The
direction and per-layer ordering are unaffected; the reported multipliers
should be read as a conservative lower bound on the underlying effect
size, not a precise measurement of it.

---

## VERIFIED RESULT: the behavioral-robustness effect reproduces on a real pretrained model (distilgpt2), not just networks trained inside this project

**Claim.** On distilgpt2 (82M parameters, a genuinely pretrained
checkpoint from the HF Hub — this project did not train it and has no
visibility into how it was trained), the same core effect found on every
from-scratch architecture tested so far reproduces: at fixed compression
parameters, behavioral error (`relative_logit_error` after weight
substitution) is 4.4–7.6× smaller for the real pretrained checkpoint than
for freshly-initialized, completely untrained copies of the same
architecture, across every compression method with room to show a
difference (SVD, vector-codebook, magnitude pruning, block-dictionary,
4-bit quantization). The sharpest single case: `quantize_4bit_block64`'s
*tensor*-level error is actually **higher** for the pretrained checkpoint
(0.101 vs. 0.090 random-init) yet its *behavioral* error is **7.6× lower**
(0.006 vs. 0.048) — direct evidence the effect is about behavioral
robustness specifically, not generic tensor-level compressibility.

**How verified.** `experiments/run_atlas_nn_stage_c_real_smoke.py`, a
fixed depth-balanced subset of 12 Conv1D layers (blocks 0/3/5 × 4 sublayer
types), 7 compression methods, real pretrained checkpoint (one fixed
instance) vs. 3 independently seeded, fully untrained copies of the same
architecture (no training performed on this arm at all — unlike every
earlier experiment, there is no training-success confound to check here).
Reproduce with `python -m experiments.run_atlas_nn_stage_c_real_smoke`
(writes `results/atlas_nn_stage_c_real_smoke.json`) — requires
`huggingface.co` to be reachable (see `pyproject.toml`'s `stage_c_real`
optional dependency group).

**Why this is stronger evidence than Experiments 3/8 alone.** Every prior
version of this finding used a network trained inside this repository, on
a synthetic or self-authored task, with this project's own training
procedure — leaving open the possibility the effect was somehow specific
to how this project trains networks. distilgpt2 removes that possibility
entirely: unknown training data, unknown procedure, unknown duration, and
the effect still appears, at a magnitude in the same range as every
from-scratch measurement (3–22.7×) found before it.

**Scope of the claim.** One pretrained checkpoint (a single fixed model —
no seeds apply to that arm), 82M parameters (still small by current
standards), a curated 12-of-many-more layer subset (a full sweep is not
practical on CPU at this scale), fixed compression parameters rather than
a behavior-budgeted ratio search (that follow-up exists as
`experiments/run_atlas_nn_stage_c_real_budget_search.py`; see
`docs/RESEARCH_LOG.md` Experiment 18 for whether it had been run yet as of
this writing). The depth pattern found here (block 3 gains most, block 5
least) is **not** the same clean monotonic-with-depth gradient Stage
C-lite found (Experiments 8–9) — reported as a genuine divergence, not
smoothed into the earlier finding; see Experiment 18 for the full
breakdown.

---

## VERIFIED RESULT: on real pretrained distilgpt2, achievable compression at matched quality reaches 307x on one layer, and the depth gradient is the largest found anywhere in this project

**Claim.** At a fixed 5% behavioral-error bar, mean achievable compression
ratio (3 random-init seeds vs. one fixed pretrained checkpoint) on
distilgpt2's Conv1D layers rises from 5.3–8.0× (random-init, every case)
to 6.3–307.2× (pretrained) across 6 tested layers (blocks 0 and 5, 3
sublayer types each). The gradient with depth is the largest in the
project: block 0's mean gain is a modest 1.3×; block 5's is **22.3×**,
driven by `transformer.h.5.mlp.c_proj` reaching 307.2× compression (SVD
rank 2 of a possible 768) at only 2.3% actual behavioral error, where the
same layer pre-training needs the safest fallback (6-bit quantization,
5.33×) because no structure-aware method clears the quality bar at all.

**A structural finding alongside the magnitude one.** In all 24
random-init searches, `quantize` (the structure-agnostic fallback) was
the only method family that ever cleared the 5% bar. In the pretrained
searches, `quantize` won only where nothing else did (2 of 6 layers);
elsewhere `atlas_block_dict`, `vector_codebook`, or `svd` won instead —
i.e. training doesn't just make the same method compress further, it
makes entire families of structure-aware compression *viable* where they
were unusable before.

**How verified.** `experiments/run_atlas_nn_stage_c_real_budget_search.py`,
6 layers × 4 states (1 pretrained + 3 random-init seeds), full 31-config
sweep per search (same grid as Experiments 4/6/9). Reproduce with
`python -m experiments.run_atlas_nn_stage_c_real_budget_search` (writes
`results/atlas_nn_stage_c_real_budget_search.json`) — real wall-clock
cost is substantial, roughly 3.3 hours on CPU for the full run.

**Relationship to Experiment 18.** Experiment 18 (fixed `svd_rank4` only)
found a confusing, non-monotonic depth pattern (block 3 > block 0 >
block 5). This result, which lets each layer pick its own best method and
parameter rather than forcing all layers through one fixed config, shows
a much cleaner monotonic block0 ≪ block5 gradient instead — strong
evidence the earlier non-monotonic reading was an artifact of probing
every layer with the same fixed rank rather than each layer's own natural
structure (block 5's best layer specifically wants rank 2, not rank 4).
See `docs/RESEARCH_LOG.md` Experiment 19 for the full reconciliation.

**Scope of the claim.** 6 of the model's 24 Conv1D layers (a CPU-feasible
subset, not exhaustive), one pretrained checkpoint (no seeds possible for
that arm), a discrete parameter grid (so 307.2× is the best available
grid point, not a continuous optimum). The block-5 mean is dominated by
one outlier layer; the other two block-5 layers still show a clear
1.5×–7.9× gain on their own, so the depth-gradient claim does not rest on
the outlier alone.

> **⚠ Update after Experiment 21 (gpt2 budget search):** the 307.2×
> *magnitude* number and the underlying "training unlocks structure-aware
> methods" finding both stand as measured on distilgpt2. But "the gain
> concentrates near the end of the network" does **not** generalize —
> gpt2 shows the opposite, largest gains near the *start* of the network.
> Read this entry's depth-gradient claim as scoped specifically to
> distilgpt2, not as a general property of trained Transformers. See the
> new VERIFIED RESULT below.

---

## VERIFIED RESULT: the behavioral-robustness effect's magnitude generalizes across two real pretrained models (distilgpt2 and gpt2); the depth-pattern shape does not generalize under a single fixed method

**Claim.** Rerunning Experiment 18's exact methodology on `gpt2` (124M
parameters, 12 blocks — larger and undistilled, vs. distilgpt2's 82M/6
blocks) reproduces the behavioral-robustness effect at closely matching
magnitude: gains of 1.4×–8.0× across 6 compression methods, vs.
distilgpt2's 1.2×–7.6× on the same methods. Reproducibility across
random-init seeds is tight on both models (under 6% spread). This is the
first cross-model generalization check in the project and the effect's
overall magnitude holds up cleanly.

**What does not generalize: the depth pattern under any single fixed
method.** `svd_rank4` peaks at the *middle* block on gpt2 (block 6 of 12),
almost exactly echoing distilgpt2's own middle-peaking, non-monotonic
result (Experiment 18). But `atlas_block_dict16_res4bit` on the same gpt2
layers shows the *opposite* shape — monotonically *decreasing* gain with
depth — directly contradicting what Experiment 19's budget search found
for distilgpt2 (achievable ratio rising sharply toward the *last* block).
Two different fixed-parameter methods on the same model disagree with
each other about which end of the network benefits more from training.

**How verified.** `experiments/run_atlas_nn_stage_c_real_smoke_gpt2.py`
(same protocol as Experiment 18, `atlas_nn/stage_c_real/model.py`
generalized to take a `model_name` parameter), 7 methods × 12 layers ×
4 states (1 pretrained + 3 random-init seeds), 336 rows. Reproduce with
`python -m experiments.run_atlas_nn_stage_c_real_smoke_gpt2` (writes
`results/atlas_nn_stage_c_real_smoke_gpt2.json`).

**Why this strengthens rather than undermines the project's depth-pattern
methodology.** Experiment 19 already established that a single fixed
compression method's depth reading is not a trustworthy measure of the
*true* achievable-ratio depth gradient (only a budget search, letting
each layer pick its own best method, is). This result is independent
confirmation of exactly that lesson, on a second model: two different
fixed methods here give two contradictory depth stories, which is exactly
what you'd expect if fixed-method depth readings are unreliable in
general, not just on distilgpt2's specific case.

**Scope of the claim.** Fixed compression parameters only — a budget
search on `gpt2` (mirroring Experiment 19) has not yet been run, so
whether gpt2's *true* achievable-ratio depth gradient matches
distilgpt2's (rising toward the last block) remains untested.

> **⚠ Update after Experiment 21 (gpt2 budget search):** it does not
> match — it's reversed. See the new VERIFIED RESULT below.
>
> **✓ Update after Experiment 22 (gpt2-medium smoke test):** the
> magnitude generalization now holds on a *third* real model —
> gpt2-medium (355M, 24 blocks) shows gains of 1.4×–9.5×, the same broad
> band as distilgpt2 (1.2×–7.6×) and gpt2 (1.4×–8.0×). Its fixed-method
> depth readings resemble gpt2's shape (early/middle blocks favored under
> both `svd_rank4` and `atlas_block_dict`) rather than distilgpt2's — but
> per this entry's own lesson, that is not yet trustworthy without a
> budget search, which had not been run on gpt2-medium as of this update.

---

## VERIFIED RESULT: the achievable-ratio depth gradient is REVERSED on gpt2 relative to distilgpt2 — "gain increases with depth" does not generalize

**Claim.** The gpt2 budget search (same methodology as Experiment 19)
found block 0's mean gain (24.8×) more than 3× larger than block 11's
(7.1×) — the opposite direction from distilgpt2, where block 0's mean
gain (1.3×) was the small one and block 5 (last)'s (22.3×) was the large
one. The single largest number in this experiment — 384× compression at
5% behavioral error, via SVD, on `transformer.h.0.attn.c_proj` — is on
the *first* block, the position that showed the *least* benefit from
training on distilgpt2. This directly falsifies "achievable-ratio gain
increases with depth" as a general property of trained Transformers —
a pattern that had held, in the same direction, across Stage C-lite's
2-block from-scratch Transformer (Experiments 8–9) and distilgpt2's
6-block real checkpoint (Experiment 19), until this second real
checkpoint broke it.

**What still holds across both real models.** The structural finding is
robust: every random-init search (18 of 18, both models combined) was won
by the safe `quantize` fallback; every pretrained search was won by a
structure-aware method instead. And Experiment 20 already established the
*overall*, depth-pooled magnitude of the behavioral-robustness effect
(1.4×–8× at fixed parameters) generalizes cleanly between the two models.
Only the specific claim about *where in the network* the largest gains
concentrate is falsified by this result — not the underlying phenomenon.

**How verified.** `experiments/run_atlas_nn_stage_c_real_budget_search_gpt2.py`,
6 layers (blocks 0 and 11, 3 sublayer types) × 4 states (1 pretrained + 3
random-init seeds), same 31-config sweep and 5% quality bar as Experiment
19. Reproduce with
`python -m experiments.run_atlas_nn_stage_c_real_budget_search_gpt2`
(writes `results/atlas_nn_stage_c_real_budget_search_gpt2.json`).

**Two unverified candidate explanations for the reversal.** (a) Scale/
depth: gpt2 is larger (124M vs. 82M) and deeper (12 vs. 6 blocks) — but
this doesn't obviously predict a full *reversal*, since both Stage
C-lite's 2-block and distilgpt2's 6-block Transformers agreed on the
late-block-dominant direction before gpt2 broke the pattern. (b) Training
procedure: distilgpt2 is trained via knowledge distillation from a larger
teacher, while gpt2 is trained from scratch on its own causal-LM
objective — a real, unexamined difference in *how* each model was
trained, not just its size. Distinguishing these needs a third model with
a different training recipe (e.g. another from-scratch causal LM near
distilgpt2's size, or a second distilled model); not attempted yet.

**Scope of the claim.** Two models, a 6-layer subset in each (not
exhaustive layer coverage), one behavioral-error threshold, discrete
parameter grids. This is a genuine falsification of a specific
generalization, reported per mission section 11 rather than smoothed
over or left implicit in the two models' separate writeups.

> **✓ Update after Experiment 23 (gpt2-medium budget search):** the two
> hypotheses this entry raised (a: scale/depth, b: distilgpt2's
> distillation training) are now narrowed, not just listed. gpt2-medium
> (355M, 24 blocks) — same non-distilled training recipe as gpt2, but 2×
> its depth and ~3× its parameters — shows the *same* early-block-
> dominant shape as gpt2, at a strikingly similar relative margin
> (≈3.5:1 early:late for both). Two very differently-scaled models trained
> the same (non-distilled) way agree with each other more closely than
> either agrees with the one distilled model. Training procedure is now
> the better-supported explanation. See the new VERIFIED RESULT below.

---

## VERIFIED RESULT: two non-distilled models of very different scale agree on the depth gradient's shape; the one distilled model disagrees — training procedure, not scale, is the better-supported explanation for the reversal

**Claim.** gpt2-medium's (355M, 24 blocks) achievable-ratio depth
gradient closely matches gpt2's (124M, 12 blocks) shape: block 0's mean
gain is 32.9× vs. block 23's 9.3× (≈3.5:1), nearly identical in ratio to
gpt2's own 24.8× vs. 7.1× (≈3.5:1) — despite gpt2-medium having roughly
double gpt2's depth and triple its parameter count. distilgpt2 (82M, 6
blocks, trained via knowledge distillation) shows the opposite direction
entirely (block 0: 1.3×, block 5: 22.3×). Two models that differ hugely
in scale from each other, but share the same non-distilled training
recipe, agree with each other; the one model with a different training
recipe disagrees with both, regardless of its scale sitting between
theirs. The single largest number found (512× compression at 5%
behavioral error, SVD, `transformer.h.0.attn.c_proj`) is again on the
*first* block, matching gpt2's own largest-number position (384×, same
relative location) rather than distilgpt2's (307×, last block).

**How verified.** `experiments/run_atlas_nn_stage_c_real_budget_search_gpt2_medium.py`,
same 6-layer (blocks 0/23 × 3 sublayer types), 4-state, 31-config-sweep,
5%-quality-bar methodology as Experiments 19 and 21, run via the new
`atlas_nn.stage_c_real.parallel_budget_search` (correctness-verified
against the sequential implementation, including a resume-from-
checkpoint check, in `experiments/verify_parallel_budget_search.py`).
Reproduce with
`python -m experiments.run_atlas_nn_stage_c_real_budget_search_gpt2_medium`
(writes `results/atlas_nn_stage_c_real_budget_search_gpt2_medium.json`).

**Why "agree despite a 3x scale difference" is stronger evidence than
either model alone.** If scale or depth were the real driver of the
depth-gradient's shape, gpt2 and gpt2-medium's substantial difference
from each other (3x parameters, 2x depth) should have produced at least
some divergence between them. Instead they match closely, while
distilgpt2 — whose scale (82M/6 blocks) sits *between* neither model
particularly closely, but whose *training procedure* (distillation) is
the one clear qualitative difference — disagrees with both. This is the
pattern you'd expect if training procedure, not model size, drives which
end of the network benefits more from training.

**Still not a proof.** Distinguishing scale from training procedure
conclusively would need a fourth model that holds one factor fixed while
varying the other directly relative to distilgpt2 — e.g. a second
distilled model (would confirm distillation specifically, not just "any
non-scale difference") or a non-distilled model at distilgpt2's exact
scale (would more cleanly isolate scale). Not attempted yet.

**What continues to hold across all three models, no exceptions.** Across
42 total budget searches (distilgpt2 + gpt2 + gpt2-medium combined), every
single random-init search was won by the safe `quantize` fallback; every
pretrained search that met the 5% quality bar at all was won by a
structure-aware method instead. This structural finding has never once
failed to replicate.

**Scope of the claim.** Three models, 6-layer subsets in two of them, one
behavioral-error threshold, discrete parameter grids. Narrows the
scale-vs-procedure question substantially; does not conclusively resolve
it.

---

## VERIFIED RESULT (partial): effective-rank shrinkage correlates with compression gain on both non-distilled real models, but not the distilled one — a second independent confirmation of the training-procedure split

**Claim.** Effective-rank shrinkage vs. log(compression_gain), computed
for the first time on real pretrained models: gpt2 shows r=+0.83 (the
strongest correlation found anywhere in this project, stronger than the
MLP's original r≈0.67 from Experiment 7), gpt2-medium shows a more
moderate r=+0.49, and distilgpt2 — the one distilled model of the three —
shows **r=−0.29**, the wrong sign entirely. Pooling all three models
together produces a misleadingly weak r=0.23 that hides this real,
strong, model-dependent split.

**Why this matters beyond one more correlation number.** This is now a
*second* independent measure (after Experiments 21/23's depth-gradient
shape) where distilgpt2 diverges from both non-distilled models, which
resemble each other despite a 3× parameter-count difference between
them. Two unrelated analyses — achievable compression ratio by depth, and
effective rank's relationship to compression gain — both split real
models the same way: distilled vs. not, not big vs. small. This
strengthens the training-procedure explanation (over model scale) for
Experiments 21/23's depth-gradient reversal.

**How verified.** `experiments/analyze_stage_c_real_capacity_metric.py`,
reusing `compression_gain` values already computed by the Experiment
19/21/23 budget searches (no new compression sweeps), effective rank via
`atlas_nn.stage_b.capacity_metrics` (unit-tested, unchanged since
Experiment 7) on each already-tested layer, 3 random-init seeds per
model. Reproduce with
`python -m experiments.analyze_stage_c_real_capacity_metric` (writes
`results/atlas_nn_stage_c_real_capacity_metric.json`).

**Labeled "partial" because of a real power caveat, not hidden.** Each
model's n=18 (6 layers × 3 seeds) is closer to an effective n=6, since
random-init effective rank barely varies across seeds at this scale
(under 0.2% spread on the example checked) — the 3 seeds are close to
redundant measurements of the same 6 layers, not 18 independent points.
gpt2's r=0.83 and distilgpt2's negative sign are large effects that
likely survive this caveat; gpt2-medium's more moderate r=0.49 should be
read with more caution.

**Scope of the claim.** Three models, 6-of-many layers each, correlational
not causal (as with every capacity-metric result in this project). Does
not establish *why* distillation produces this pattern — only that it
does, on two independent measures now.

---

## VERIFIED RESULT: delta-rank fraction does NOT transfer to the Stage C-lite Transformer — confirmed at 8 seeds, not a small-sample artifact

**What was tested.** Whether Experiments 25–26's strongest-in-the-project
correlation (delta-rank fraction vs. compression-gain magnitude, r=−0.75
on the MLP hidden layer) holds on the Stage C-lite Transformer.

**Result at 3 seeds (Experiment 9's original power).** No clean
relationship: pooled r=−0.15, block 0 r=+0.16 (wrong direction), block 1
r=−0.15 (right direction, weak).

**Result at 8 seeds (addendum, matching the project's later standard) —
the non-relationship sharpens, it does not resolve into a hidden
positive one.** Pooled r=−0.06 (weaker still), block 0 r=+0.16
(unchanged), **block 1 flips to r=+0.15** (the one scope with the
"right" sign at 3 seeds now has the wrong one at 8) — the opposite of
what happened when this exact seed-count increase (3→8) resolved
Experiment 10/11's earlier ambiguity in a positive direction. The
classifier head's apparent correlation (0.95 at 3 seeds, 0.74 at 8) is
confirmed as a numerical artifact both times — that layer's
degenerate 2-dimensional max rank leaves `delta_rank_fraction` constant
at 0.500. The sublayer-type breakdown is stable across both seed counts:
`out_proj` has both the lowest (most-concentrated) mean delta-rank
fraction of any sublayer type *and* the lowest mean compression gain —
the opposite of what the MLP's finding would predict — at both 3 and 8
seeds.

**How verified.** `experiments/analyze_stage_c_lite_delta_rank.py` at
`SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)`, against a matching 8-seed
rerun of `experiments/run_atlas_nn_stage_c_lite_budget_search.py`.
Reproduce with `python -m experiments.run_atlas_nn_stage_c_lite_budget_search`
then `python -m experiments.analyze_stage_c_lite_delta_rank` (writes
`results/atlas_nn_stage_c_lite_delta_rank.json`).

**Why this belongs here even though it's a clean negative, now confirmed
twice.** It directly echoes Experiment 10's non-replication of the
*final*-matrix effective-rank finding on this same Transformer — both
the "final matrix" and "training update" versions of this rank-based
mechanism-hunting approach hit the same architecture-generalization
wall, and raising statistical power made the wall more solid, not less.
Reported per mission section 11: a negative result that narrows a
promising finding's scope is exactly as valuable as the positive finding
itself — and checking whether it survives more power, in *either*
direction, is exactly the discipline this project has applied to every
promising result so far.

---

## VERIFIED RESULT: delta-rank fraction predicts compression gain strongly on real non-distilled models (gpt2 r=-0.90, the strongest correlation in the project) but not on the distilled one — a third independent measure of the same split

**Claim.** Delta-rank fraction (effective rank of the training-induced
weight change) vs. compression-gain magnitude, on real pretrained
models: gpt2 shows r=−0.90 (Pearson) — the strongest correlation of any
kind found anywhere in this project, stronger than this experiment's own
MLP source finding (Experiment 26, r=−0.75). gpt2-medium shows r=−0.60,
the same direction. distilgpt2 — the one distilled model of the three —
shows no clear relationship (r=+0.19). This is now the **third**
independent analysis (after Experiments 21/23's depth-gradient shape and
Experiment 24's final-matrix effective-rank correlation) to split these
same three real models the same way: the two non-distilled models
resemble each other; the distilled one does not resemble either,
regardless of its scale sitting between theirs.

**How verified.** `experiments/analyze_stage_c_real_delta_rank.py`,
reusing `compression_gain` already computed by the Experiment 19/21/23
budget searches (no new compression sweeps), delta-rank fraction via one
SVD per already-tested layer, 3 random-init seeds per model. Reproduce
with `python -m experiments.analyze_stage_c_real_delta_rank` (writes
`results/atlas_nn_stage_c_real_delta_rank.json`).

**Why this also narrows Experiment 27's open question.** Experiment 27
found delta-rank fraction does not predict compression gain on the Stage
C-lite Transformer (a small, from-scratch, briefly-trained model) and
could not tell whether that was about Transformers generally or about
that specific toy setup. This result favors "that specific toy setup":
real, thoroughly-pretrained, non-distilled Transformers show a *strong*
delta-rank relationship — stronger than the MLP's own — while Stage
C-lite's from-scratch Transformer showed essentially none.

**Power caveat, carried over from Experiment 24's identical situation.**
Delta-rank fraction is nearly constant across the 3 random-init seeds
for a given layer, so each model's n=18 (6 layers × 3 seeds) has an
*effective* sample size closer to 6. Per-block breakdowns computed
alongside this result show implausibly perfect correlations (e.g.
r=−0.9999, fit through what is effectively 3 tripled data points) and are
excluded from interpretation — only the per-model, all-6-layer numbers
above are treated as trustworthy.

**Scope of the claim.** Three models, 6-of-many layers per model,
correlational not causal. Does not establish *why* distillation disrupts
the relationship — only that it does, on three independent measures now
(depth-gradient shape, final-matrix rank correlation, delta-rank
correlation).

---

## VERIFIED RESULT: a fourth model refutes "distillation specifically" and supports a broader split — training from scratch vs. starting from another model's already-trained weights

**Claim.** `microsoft/DialoGPT-small` (124M, 12 blocks, config-identical
to gpt2, but initialized from gpt2's own weights and fine-tuned on
dialogue data — never distilled) shows the *same direction* of
achievable-ratio depth-gradient reversal as distilgpt2, and more
strongly: block 0's mean gain is **0.59×** (pretrained early-block
weights are *harder* to compress at the 5% quality bar than a same-shape
random matrix — the only sub-1 block-mean gain seen on any real
pretrained model in this project), block 11's is **17.0×**, a 28.6:1
late:early ratio exceeding distilgpt2's own 17.3:1. gpt2 and gpt2-medium
(both trained from scratch, at very different scales) agree closely with
each other in the opposite direction (≈3.5:1 early:late, both). Since
DialoGPT-small was never distilled — it started from gpt2's own weights,
the very model whose pattern it does *not* match — "distillation
specifically" cannot be the mechanism. What distilgpt2 and DialoGPT-small
share instead, and gpt2/gpt2-medium don't, is that neither started
training from a random initialization: both began from an
already-trained model's weights (a teacher's, or gpt2's own) and were
further optimized from there.

**How verified.** `experiments/run_atlas_nn_stage_c_real_budget_search_dialogpt.py`,
identical 6-layer (blocks 0/11 × 3 sublayer types), 4-state,
31-config-sweep, 5%-quality-bar methodology as Experiments 19/21/23, run
via the checkpointed parallel wrapper. Interrupted by a fourth container
restart at 16/24 layer-searches; resumed correctly from checkpoints
rather than restarting, confirming the checkpointing infrastructure
built after Experiment 23's restart generalizes to a new model without
modification. Reproduce with
`python -m experiments.run_atlas_nn_stage_c_real_budget_search_dialogpt`
(writes `results/atlas_nn_stage_c_real_budget_search_dialogpt.json`).

**Why this is stronger evidence than three converging measures on the
same three checkpoints.** Experiments 21/23/24/28 all split the *same*
three models (distilgpt2 vs. gpt2 vs. gpt2-medium) the same way, which
is repeated agreement between the same data points, not new information
about *why*. DialoGPT-small is a genuinely new test: it isolates
"distillation" from "not trained from scratch" by being an example of
the latter without the former. That it patterns with distilgpt2 anyway —
and more strongly — directly rules out the narrower hypothesis and
narrows the explanation to the broader one.

**Still correlational, small sample.** Four checkpoints, two per
category (from-scratch: gpt2, gpt2-medium; derived-from-prior-weights:
distilgpt2, DialoGPT-small). The two derived models used mechanically
different procedures (teacher-distillation vs. ordinary fine-tuning) that
happen to agree here; a fifth model would be needed to rule out
coincidence at this sample size.

**What continues to hold across all four models, no exceptions.** Across
48 total budget searches now, every random-init search was won by the
safe `quantize` fallback; every pretrained search that met the quality
bar was won by a structure-aware method instead. Never once failed to
replicate across four independent checkpoints.

**Scope of the claim.** Four models, 6-layer subsets, one behavioral-
error threshold, discrete parameter grids. Narrows but does not
conclusively prove the training-origin hypothesis.

---

## VERIFIED RESULT: a fifth model confirms the training-origin split holds across three different derivation procedures — five for five on direction, with magnitude that appears to track how much post-initialization training occurred

**Claim.** `lvwerra/gpt2-imdb` (124M, 12 blocks, config-identical to
gpt2, ordinary supervised fine-tuning of gpt2's own weights on IMDB
movie reviews — a third derivation procedure, distinct from both
distillation and DialoGPT-small's dialogue fine-tuning) shows the same
*direction* of depth-gradient reversal as distilgpt2 and DialoGPT-small:
late-block mean gain (3.06×) exceeds early-block (1.12×), a 2.73:1
ratio. Direction matches both other "derived" models; magnitude is an
order of magnitude weaker (2.73:1 vs. 17.3:1 and 28.6:1). All five real
pretrained models tested now agree on sign with zero exceptions:
from-scratch models (gpt2, gpt2-medium) show early-block dominance
(ratio 0.28–0.29); derived-from-prior-weights models (distilgpt2,
DialoGPT-small, gpt2-imdb) show late-block dominance (ratio 2.73–28.6),
regardless of which specific derivation procedure was used.

**How verified.** `experiments/run_atlas_nn_stage_c_real_budget_search_gpt2imdb.py`,
identical 6-layer (blocks 0/11 × 3 sublayer types), 4-state,
31-config-sweep, 5%-quality-bar methodology as Experiments 19/21/23/29,
run via the checkpointed parallel wrapper. Completed in a single pass
(no container restart this time). Reproduce with
`python -m experiments.run_atlas_nn_stage_c_real_budget_search_gpt2imdb`
(writes `results/atlas_nn_stage_c_real_budget_search_gpt2imdb.json`).

**Why the graded magnitude strengthens rather than weakens the finding.**
A binary "derived vs. from-scratch" hypothesis predicts only sign, not
magnitude — gpt2-imdb's much weaker ratio could have looked like
evidence against the split if the hypothesis demanded uniform strength.
Instead the ordering (gpt2-imdb, lightly fine-tuned on a modest corpus <
DialoGPT-small, fine-tuned on a large dialogue corpus < distilgpt2, a
full distillation run) tracks a plausible continuous variable — how far
training moved the weights from their initial distribution — that a
strictly binary framing would not have predicted. This is offered as a
new hypothesis suggested by the data, not yet independently tested.

**Still correlational.** Five checkpoints, one behavioral-error
threshold, discrete parameter grids, and the magnitude-tracks-training-
depth idea above is untested — it would need training duration/data
volume varied directly on one base model, not comparison across
unrelated checkpoints with many confounded differences.

**What continues to hold across all five models, no exceptions.** Across
120 total layer-state budget searches now (6 layers × 4 states × 5
models), every random-init search was won by the safe `quantize`
fallback; every pretrained search that met the quality bar was won by a
structure-aware method instead. Never once failed to replicate.

**Scope of the claim.** Five models, 6-layer subsets, one behavioral-
error threshold, discrete parameter grids. The direction of the
training-origin split is now well-supported across three independent
derivation procedures; the magnitude-tracks-training-depth idea is a new
untested hypothesis, not a confirmed finding.

---

## Explicitly not yet claimed

- Nothing about *larger* networks (mission Stage D) or models above ~100M
  parameters — Experiment 18 covers one 82M-parameter pretrained model
  (distilgpt2), on a curated layer subset, at fixed compression parameters
  only. A real dataset and a real pretrained transformer are no longer
  unclaimed (see the VERIFIED RESULT above), but "larger networks" still
  is.
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
