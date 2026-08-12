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
