# Next Research Decision

Updated after Experiment 13 (LayerNorm added to the MLP), which split
Experiment 11's LayerNorm finding into a part that generalizes across
architectures and a part that reverses sign. Covers Track B (`atlas_nn`)
only.

## 1. What we learned

**Experiments 3–9:** the core phenomenon (trained networks tolerate
weight-compression error far better than tensor error predicts, scaling
with depth) is well-established across two architectures, three tasks.

**Experiment 11 (8 seeds):** on the Transformer, LayerNorm amplifies
gain magnitude 2–4× and weakens the rank-shrinkage correlation
(0.23–0.29 with it, 0.37–0.55 without).

**Experiment 13 (this round) — the LayerNorm claim splits cleanly in
two:**
- **Generalizes:** LayerNorm presence weakens the rank-shrinkage
  correlation in *both* architectures (MLP: 0.59→0.16 adding it;
  Transformer: 0.55→0.23 removing it — same direction either way).
  This looks like a real, general property of normalization: it makes a
  layer's compressibility less dependent on that layer's own weight-matrix
  rank.
- **Reverses:** LayerNorm's effect on gain *magnitude* is opposite
  between architectures — it roughly quadruples the Transformer's gain
  but *shrinks* the MLP's by ~3.3× (9.7→2.9). Whatever determines the
  sign of this effect is not LayerNorm alone; attention (present in the
  Transformer, absent in the MLP) is the obvious untested remaining
  candidate.
- Both effects, in both architectures, are concentrated in the layer
  where each architecture's capacity-slack mechanism already lives (MLP:
  hidden layer; Transformer: block 1) — input/output-type layers are
  essentially unaffected either way.

**Where this leaves the mechanism question:** three architectural
factors have now been tested (effective rank alone, residual
connections, LayerNorm) across two architectures. The clearest, most
general finding to emerge is not "X causes the effect" for any single X,
but that **normalization decouples compressibility from raw rank
structure quite generally**, while **something else (likely attention)
determines how large the resulting effect is**.

## 2. What failed / remains untested

- LayerNorm as a complete, architecture-general explanation for gain
  *magnitude* — ruled out; it's architecture-dependent in sign.
- Attention as the candidate explanation for the magnitude-sign
  difference — proposed, not yet tested in isolation.
- Whether the MLP's magnitude-suppression from LayerNorm is itself robust
  across tasks/widths, or specific to 2-XOR at width 64 — only one
  configuration tested.
- The MLP input-layer question (Experiment 12) — still open, untouched
  this round.

## 3. What worked

- Testing the LayerNorm hypothesis on a *second* architecture rather than
  accepting the Transformer-only finding as general is what surfaced the
  magnitude-sign reversal — a materially more precise (and more
  interesting) result than either "confirmed" or "not confirmed" would
  have been on their own.
- Running the MLP check at 8 seeds from the start (learning from
  Experiment 11's own history of needing a power upgrade) avoided
  repeating that same mistake.
- The per-layer breakdown (not just pooled) is what revealed *where* the
  effects concentrate (hidden/block-1 only) — consistent with, and
  reinforcing, the project's running finding that input and output-type
  layers behave differently from the "capacity-bearing" middle layers in
  every architecture tested so far.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the mechanism understanding is now more precise, if also more
complicated, than a day ago.** The phenomenon itself needs no further
qualification at this point — well-evidenced, multi-architecture,
multi-task. The mechanism has resolved into two distinct sub-questions:
a likely-general one (normalization decouples compressibility from rank)
that's now supported in two architectures, and an architecture-specific
one (what determines gain magnitude) that isn't resolved and may hinge
on attention specifically. This is normal, healthy progress in
mechanism-hunting — real phenomena often have more than one contributing
factor, and finding that out is not a setback.

## 5. The single most informative next experiment

**Test attention directly as the magnitude-sign factor**: take the
Stage C-lite `AblationTransformerClassifier` and additionally ablate
attention itself (e.g. replace the self-attention sublayer with a
no-op or a plain per-token Linear projection, keeping LayerNorm and the
FFN sublayer), and check whether the LayerNorm-magnitude-amplification
effect survives without attention. If it disappears, attention is
confirmed as the necessary ingredient for LayerNorm's magnitude effect
to flip positive; if it persists, something else about the Transformer
(depth of 2 sublayers per block? residual placement relative to
LayerNorm? something in this experiment's specific hyperparameters?) is
responsible instead.

Why this one: it directly and cheaply (reuses existing
`atlas_nn.stage_c_lite` infrastructure, one more toggle) tests the
specific candidate this round's result raised, continuing the same
chained-hypothesis discipline that has driven every productive result
since Experiment 5.

**Other queued options, unchanged in priority:** the MLP input-layer
question (Experiment 12, no strong lead yet); a genuine pretrained-model
test (still blocked by network policy, still the single highest-value
addition if that changes).
