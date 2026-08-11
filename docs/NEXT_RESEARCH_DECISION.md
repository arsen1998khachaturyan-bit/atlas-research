# Next Research Decision

Updated after Experiment 9 (Stage C-lite budget search), which converted
Experiment 8's qualitative Transformer finding into the same quantitative
"achievable ratio at matched quality" form used for the MLP in Experiments
4/6. Covers Track B (`atlas_nn`) only — Track A is unaffected and out of
scope for this document.

## 1. What we learned

**Experiments 3–7 (MLP, synthetic tasks):** trained-network behavior is
far more robust to weight-compression error than tensor error predicts;
this tracks spare capacity relative to task difficulty (confirmed via a
predicted reversal); effective rank partially predicts gain magnitude on
the hidden layer specifically.

**Experiment 8 (Transformer, real task, qualitative):** the same
behavioral-robustness effect reproduces on a structurally different
architecture and a real task, strengthening with depth across two
Transformer blocks (3.5–4.9× in block 0, 15.7–22.7× in block 1,
per-seed).

**Experiment 9 (this round) — the same finding, now as an actual
compression-ratio number:** at a fixed 5% behavioral-error bar, achievable
compression ratio rises from ~10–14× (random-init) to ~14–43× (trained),
and the *increase from training* itself grows with depth: block 0 gains
1.5×–3.1×, block 1 gains 3.0×–4.4×, consistently across all three matched
layer-type pairs. Trained block-1 layers reach the highest ratios recorded
for any real (non-tiny-output) layer in the project so far (~32–43×).
SVD becomes the dominant method for trained block-1 layers, mirroring
Experiment 7's MLP finding that SVD is most sensitive to training-induced
rank structure — now on a second architecture.

**Taken together, Experiments 8–9 are the strongest transfer evidence in
the project:** the core finding now holds not just across tasks and
network widths within one MLP (Experiments 4–7), but across a
structurally different architecture and a real (if simple) task,
in both its qualitative (behavioral robustness) and quantitative
(achievable ratio) forms.

## 2. What failed / remains untested

- The numeric gain multipliers from budget-search (1.5×–4.4×) are smaller
  than the raw relative-logit-error gains from the smoke test
  (3.5×–22.7×) — a known artifact of the discrete parameter grid, not a
  contradiction; noted so the two numbers aren't confused with each other.
- The capacity-metric (effective-rank) analysis from Experiment 7 has
  still not been repeated on the Transformer — unknown whether it predicts
  gain magnitude here the way it did for the MLP's hidden layer.
- The MLP's still-unexplained input-layer behavior (flat-to-negative
  after training, Experiments 4/6) has not been investigated — still an
  open question, not addressed this round.
- Attention `in_proj_weight` (combined Q/K/V) remains untested — stored as
  a raw parameter, not an `nn.Linear` submodule, so it's invisible to the
  current layer-discovery mechanism.
- Still small scale, still not mission Stage C's literal target (a real
  pretrained model on a real dataset) — `huggingface.co` remains blocked
  by network policy.

## 3. What worked

- Reusing `run_budget_search` completely unchanged on a new architecture
  (zero new logic needed beyond the dataset/model files) validated the
  Experiment 8 refactor's payoff immediately — this is now demonstrated
  twice (MLP, Transformer), not just claimed as a design goal.
- Converting a qualitative finding into a quantitative one caught a subtlety
  worth remembering: discrete search grids compress the *apparent* size of
  a continuous effect. Both numbers are true; they answer different
  questions ("how much does behavior degrade" vs. "how much can I actually
  compress at this precision"), and conflating them would overstate or
  understate the practical takeaway depending on which is quoted.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, with the broadest evidentiary base in the project to date.** The
central refined claim — trained networks can be represented with
substantially less independent information in proportion to unused
capacity, concentrated in deeper layers — now has quantitative,
multiseed support on two structurally different architectures (MLP,
Transformer), across three tasks (2-XOR, 3-parity, real sentiment text),
in both tensor-adjacent (Experiment 7) and purely behavioral (Experiments
3, 8) and combined ratio-at-quality (Experiments 4, 6, 9) forms. This is
a materially broader base than any single experiment could provide, and
the depth-gradient pattern — new information from the Transformer that
the 3-layer MLP couldn't reveal (only 1 hidden layer to test) — is itself
a specific, useful, falsifiable refinement: compression headroom does not
merely exist somewhere in a trained network, it appears to *increase
with representational depth*, at least in these two architectures.

**What's still missing:** direct capacity-metric evidence on the
Transformer (Experiment 7's analog); the input-layer explanation (open
since Experiment 4); and, as always, real pretrained-model validation.

## 5. The single most informative next experiment

Two candidates queued from before, still both open:

**(a) Effective-rank capacity-metric analysis on the Transformer**
(Experiment 7's analog) — does effective rank predict the *magnitude* of
the depth-graded gain found in Experiments 8–9, the way it did (r≈0.67) for
the MLP's hidden layer? This is the more direct extension of this round's
work and would test whether the *mechanism*, not just the *pattern*,
transfers across architectures.

**(b) The MLP input-layer mystery** (flagged in Experiments 4, 6, and 7,
never explained) — does its flat-to-negative post-training compressibility
relate to how much of the input is task-relevant (2 of 32 dims for 2-XOR)
rather than to capacity?

Recommendation: **(a) first** — it's the more direct continuation of
today's strongest finding (the depth gradient), reuses
`atlas_nn.stage_b.capacity_metrics` unchanged (already architecture-
agnostic, pure linear algebra on weight tensors), and would either
strengthen the "effective rank explains it" story across architectures or
reveal that the Transformer's depth gradient has a different underlying
cause than the MLP's capacity story — informative either way. (b) remains
next after that.

**Standing recommendation, unchanged:** a real pretrained checkpoint is
still the single most valuable possible upgrade to this research line if
network policy allows it — everything in Experiments 8–9 is evidence
*for* prioritizing that investment (the pattern is real and transfers),
not a replacement for testing it directly.
