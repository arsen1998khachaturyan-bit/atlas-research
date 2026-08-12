# Next Research Decision

Updated after Experiment 19 (Stage C real: distilgpt2 budget search), the
largest-magnitude confirmation of the mission hypothesis found anywhere in
this project so far. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–17:** established, refined, and mechanistically explained
the core behavioral-robustness effect on two from-scratch architectures.

**Experiment 18:** the effect reproduces on distilgpt2, a real pretrained
model this project did not train — the first evidence untied from this
project's own training procedure. One odd result: a non-monotonic depth
pattern under a single fixed compression method (`svd_rank4`).

**Experiment 19 (this round) — resolved and sharpened.** The
achievable-compression-ratio view (each layer choosing its own best
method/parameter, not forced through one fixed config) shows a clean,
strongly monotonic depth gradient after all: block 0 gains ~1.3× from
training, block 5 gains ~22.3×, with one layer
(`transformer.h.5.mlp.c_proj`) reaching **307.2× compression at 2.3%
behavioral error** post-training (SVD rank 2) vs. 5.33× pre-training. This
is the largest compression number in the whole project, on a real model.
Alongside the magnitude finding: training doesn't just improve ratios, it
changes which method *families* are viable at all — every random-init
search's winner was the safe `quantize` fallback; most pretrained
searches were won by a structure-aware method instead.

## 2. What failed / remains untested

- Experiment 18's non-monotonic depth reading is now understood as an
  artifact of forcing every layer through the same fixed rank-4 SVD probe
  rather than a real property of the model — but this explanation, while
  well-supported, has not been independently stress-tested (e.g. by
  re-running Experiment 18's fixed-method view at each layer's own
  *natural* best rank instead of a uniform rank 4).
  the mechanism questions from Experiments 10-17 (effective rank, residual
  connections, LayerNorm, attention mixing) have still not been re-tested
  on distilgpt2 -- the from-scratch Transformer's mechanism picture has
  not been checked against a real pretrained one.
- Only 6 of distilgpt2's 24 Conv1D layers tested (CPU feasibility); the
  other 18, and the fused `attn.c_attn` sublayer type specifically
  (dropped from this script for runtime reasons), remain unmeasured.
- Only one pretrained model. Whether the 307x number and the depth
  gradient's magnitude are distilgpt2-specific or reflect something more
  general about pretrained language models remains untested.
- The MLP input-layer question (Experiments 4, 6, 7, 12, 15, 16) is still
  the project's longest-standing open thread, untouched this round.

## 3. What worked

- Running the full budget search rather than a reduced-scope version (the
  user's explicit choice after being told the real ~4-5 hour cost) paid
  off directly: the reduced-scope alternative offered would have cut
  layers/methods in ways that plausibly could have masked exactly the
  large block-5 outlier that turned out to be the headline result.
- Treating Experiment 18's confusing non-monotonic finding as unfinished
  business rather than either dismissing it or overinterpreting it,
  continuing the project's established pattern from Experiments 10->11,
  12/15->15, and 14->17.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, more strongly than at any earlier point in the project.** A real,
externally-trained, 82M-parameter model shows a compression gain (up to
307x on one layer, at under 2.3% behavioral error) larger than anything
found on this project's own from-scratch architectures, using structure
that provably doesn't exist in the same layer before training. This is
close to the strongest form of evidence the mission's Stage C ladder step
was designed to produce.

## 5. The single most informative next experiment

No single next step clearly dominates; in rough priority order:

**(a)** A second pretrained checkpoint (e.g. `gpt2` or `gpt2-medium`) to
test whether the 307x-scale outlier and the depth gradient's magnitude
are distilgpt2-specific or general across model sizes/families -- the
most direct way to test how far Experiment 19's headline number
generalizes.

**(b)** Extend layer coverage on distilgpt2 itself (the remaining 18
layers, including `attn.c_attn`) to check whether block 5's large gain is
specific to `mlp.c_proj` or a genuine property of "late in the network"
more broadly -- cheaper than (a), refines rather than extends the current
finding.

**(c)** Revisit the MLP input-layer question, still open since
Experiment 4 and untouched for several rounds now.

**(d)** A consolidation pass incorporating Experiments 18-19 into the
project-wide synthesis artifact (the "Slack Hypothesis" retrospective
built earlier this session), which currently stops at Experiment 17 and
does not yet reflect the real-pretrained-model result.
