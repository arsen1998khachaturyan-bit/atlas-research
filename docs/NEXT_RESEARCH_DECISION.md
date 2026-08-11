# Next Research Decision

Updated after Stage C-lite (Experiment 8): a small Transformer trained
from scratch on a real sentiment task, run because mission Stage C's
literal requirement (a pretrained open model) is currently unreachable —
this session's network policy blocks `huggingface.co` with a 403 (confirmed
via the egress proxy's own status endpoint; per that proxy's rules, policy
denials are reported, not routed around). Covers Track B (`atlas_nn`) only
— Track A is unaffected and out of scope for this document.

## 1. What we learned

**Experiments 3–7 (MLP, synthetic XOR/parity tasks):** established that
trained-network behavior is far more robust to weight-compression error
than tensor error predicts, that this tracks spare capacity relative to
task difficulty (confirmed via a predicted reversal), and that effective
rank partially predicts the gain's magnitude on the hidden layer
specifically.

**Experiment 8 (this round) — first test outside the MLP/XOR setting:**
the core behavioral-robustness effect reproduces on a structurally
different architecture (a 2-block Transformer with self-attention) and a
real task (self-authored English sentiment sentences), and does so more
strongly with depth: 3.5–4.9× robustness gain in the earlier Transformer
block, 15.7–22.7× in the later one, holding per-seed across all 3 seeds
tested. This is the most externally valid evidence in the project so far
— nothing about this architecture or task was tuned to reproduce the
earlier finding.

**Also confirmed to transfer, not just the headline effect:** the tiny
output-layer overhead pathology (encoding overhead exceeding a small
tensor's own size) found in Stage B reproduces identically on the
Transformer's classification head — a second piece of evidence that
finding isn't MLP-specific either.

## 2. What failed / remains untested

- This was a fixed-parameter smoke test (Experiment 3-style), not a
  behavior-budgeted search (Experiment 4/6-style) — there is no "Nx
  achievable compression ratio" number for the Transformer yet, only the
  tensor-vs-behavior error gap.
- The capacity-metric correlation work (Experiment 7) has not been
  repeated on this architecture — unknown whether effective rank predicts
  gain magnitude here the way it did on the MLP's hidden layer.
- Still small scale (~50K parameters, ~200 template-generated sentences) —
  not mission Stage C's literal target (a real pretrained model on a real
  dataset), which remains blocked by network policy, not by anything this
  session did wrong.
- The attention mechanism's own weights (`in_proj_weight`, the combined
  Q/K/V projection) were not tested — `nn.MultiheadAttention` stores that
  as a raw parameter rather than an `nn.Linear` submodule, so the existing
  `linear_layer_names`-based layer discovery doesn't see it. Only
  `out_proj` (attention output) and the two FFN layers were tested per
  block.

## 3. What worked

- The `get_weight`/`set_weight`/`evaluate`/`snapshot`/`load_snapshot`
  dependency-injection refactor (done to enable this experiment) revealed
  that Stage B's training/evaluation machinery was already fully
  architecture-agnostic — no MLP-specific logic had to be duplicated for
  the Transformer, only a new dataset and a new `build_transformer_
  classifier`. This is worth keeping in mind for any future architecture:
  the existing `atlas_nn.stage_b.experiment`/`budget_search` machinery
  should work unchanged on any classifier whose `forward` takes one input
  tensor and returns class logits.
- Checking training success (56–64% random-init vs. 84–94% trained,
  disjoint held-out set) before trusting any result — now standard
  practice after Experiments 6–7 — confirmed this experiment's training
  was genuine, not a repeat of the width-256/deep-net confounds found
  earlier.
- Reporting the network-policy block plainly and asking the user how to
  proceed, rather than attempting a workaround, kept the research honest
  about what was and wasn't actually tested (a "lite" substitute, clearly
  labeled as such throughout the code and docs, not silently presented as
  the mission's literal Stage C).

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and now with cross-architecture, real-task evidence, not just
cross-task evidence within one architecture.** The behavioral-robustness
effect (trained networks tolerate weight-compression error far better than
tensor error predicts, more so in deeper layers) has now been shown on:
(1) an MLP on two synthetic tasks of different difficulty and at three
network widths, and (2) a Transformer on a real text task. Two
structurally different architectures showing the same qualitative pattern,
with the second showing an even sharper depth gradient, is meaningfully
stronger support than either alone. The mission's Stage C intent (test
behavioral degradation on real tasks, not just synthetic reconstruction)
is now partially met — with a real task and real (if simple) language
data, even though the "pretrained" half of Stage C's literal definition
is still blocked.

**What's still missing:** the quantitative budget-search number for this
architecture (how much compression ratio, not just how much robustness);
whether the depth-gradient pattern continues if the Transformer had more
than 2 blocks; and, unavoidably, a genuine pretrained-model test, which
needs either a policy change or an alternate model source.

## 5. The single most informative next experiment

**Run the budget-search machinery (already architecture-agnostic) on the
Stage C-lite Transformer**, turning Experiment 8's qualitative
tensor-vs-behavior gap into the same kind of quantitative "achievable
ratio at matched quality" number Experiments 4 and 6 produced for the MLP.
Given Experiment 8's clean depth gradient, the specific, falsifiable
prediction is: achievable compression ratio at a fixed behavioral-error
bar should be higher in block 1 (later) layers than block 0 (earlier)
layers, and the gap should be *larger* than the roughly-flat pattern the
MLP showed between its single hidden layer and output layer.

Why this one: it's the natural, low-cost next step (reuses
`atlas_nn.stage_b.budget_search` unchanged, per this round's refactor),
directly extends the strongest and newest finding (Experiment 8) rather
than opening a new thread, and would give a real "Nx compression, Y%
quality" number for a non-MLP architecture for the first time — the kind
of number that would matter most if/when a genuine pretrained-model source
becomes available.

**Separately, worth raising with the user again given today's finding:**
a real pretrained checkpoint is still the most valuable single upgrade
available (mission's literal Stage C) — if network policy changes, or an
alternative reachable model source is identified, that should take
priority over further scratch-trained experiments, since the mission
places real weight structure firmly above synthetic/from-scratch analogs
in its evidentiary value.
