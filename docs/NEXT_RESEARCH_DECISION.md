# Next Research Decision

Updated after Experiment 10 (Transformer capacity-metric check), which
found a genuine divergence from the MLP: the phenomenon transfers across
architectures, but the effective-rank *explanation* for it does not.
Covers Track B (`atlas_nn`) only — Track A is unaffected and out of scope.

## 1. What we learned

**Experiments 3–7 (MLP):** trained-network behavior is far more robust to
weight-compression error than tensor error predicts; this tracks spare
capacity relative to task difficulty; effective rank partially predicts
gain magnitude on the hidden layer specifically (r≈0.67).

**Experiments 8–9 (Transformer, real task):** the core phenomenon —
behavioral robustness and achievable compression ratio both increasing
after training, more so in deeper layers — reproduces cleanly, in both
qualitative and quantitative form, on a structurally different
architecture and a real task.

**Experiment 10 (this round) — the mechanism does NOT transfer, even
though the phenomenon does:** effective-rank shrinkage does not positively
predict the Transformer's compression gain (overall r=−0.25; block 1,
the layer type with the *largest* gains, r=−0.54 — opposite sign from the
MLP). The Transformer's effective rank barely moves with training (1–4%
relative shrinkage) even where compression/robustness gains are largest —
unlike the MLP, where the best-correlated layer showed up to 44%
shrinkage. **The phenomenon is now well-evidenced across two
architectures; its explanation is not.**

**Working hypothesis, unverified:** residual connections and layer
normalization (present in the Transformer, absent in the plain MLP) may
provide an error-absorbing pathway that makes a sublayer's output
tolerant to weight perturbation independent of that sublayer's own
weight-matrix spectral structure — i.e. robustness from architectural
connectivity, not from any single layer becoming more low-rank.

## 2. What failed / remains untested

- Effective rank as a cross-architecture explanation for the
  capacity-slack effect — worked once (MLP hidden layer), failed on a
  second architecture where it was expected to work even better (the
  Transformer's block 1, which has the largest gains).
- The residual-connection/LayerNorm hypothesis above is stated but not
  tested this round — no experiment yet isolates whether removing
  residual connections (or LayerNorm) from the Transformer would restore
  a rank-based relationship, or whether some other property (e.g. weight
  magnitude/scale distribution, attention-pattern sharpening) is the real
  driver.
- The MLP's still-unexplained input-layer behavior (flagged since
  Experiment 4) remains open, now joined by a second open mechanism
  question (what *does* explain the Transformer's gain, if not rank).
- Statistical power for Experiment 10's correlations was limited (n=9 per
  block, several exactly-repeated gain values from the coarse
  budget-search grid) — the qualitative "no positive relationship" finding
  is solid; the precise r values are not.

## 3. What worked

- Running the exact same analysis (effective rank vs. compression gain)
  on a second architecture, rather than assuming the MLP's mechanism
  would transfer alongside the phenomenon, is what caught this — a direct
  application of the mission's falsification discipline, and it produced
  a more interesting, more specific research question than either a clean
  confirmation or a total non-replication would have.
- The distinction this reveals — *phenomenon* (robustness increases with
  training and depth) vs. *mechanism* (why) — is itself a useful framing
  going forward: two architectures now agree on the phenomenon; zero
  mechanisms have been confirmed to generalize.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes for the phenomenon, not yet for any specific mechanism.** The
practical claim — trained networks can be represented with substantially
less information while preserving behavior, more so in deeper layers —
now has strong, multiseed, cross-architecture, partially-real-task support
(Experiments 3–4, 6, 8–9). But *why* this happens is now demonstrably not
settled: the one candidate mechanism tested (effective rank) explains it
on one architecture and not the other, despite the phenomenon itself
being consistent across both. This is scientifically healthy — a strong
empirical regularity with an open, actively narrowing mechanistic
question — rather than either overclaiming ("training reduces rank, and
that's why compression works") or underclaiming ("we don't know
anything"). The mission's own framing (section 15's "very interesting"
bar: large reduction + negligible degradation + reasonable runtime) is
arguably met for the phenomenon on both architectures tested; the
"why," which the mission does not explicitly require but which matters
for predicting where this will and won't work at scale, is the open
frontier now.

## 5. The single most informative next experiment

**Test the residual-connection/LayerNorm hypothesis directly**: build a
version of the Stage C-lite Transformer block with residual connections
and/or LayerNorm disabled (or a matched plain-MLP-style stack of the same
width/depth without them), retrain, and check whether (a) the
behavioral-robustness gain shrinks toward the "no relationship" pattern,
and/or (b) effective rank starts predicting the gain the way it did for
the MLP. A clean result either way is informative: if disabling residual
connections restores a rank-based relationship, that identifies
architectural connectivity (not raw weight-matrix structure) as the real
capacity-slack mechanism in attention-based models — a genuinely new,
specific, falsifiable finding worth its own entry. If it doesn't, the
residual-connection hypothesis is wrong and the search for the real
mechanism continues elsewhere (weight-magnitude distribution is the next
natural candidate to check).

Why this one: it's the direct, cheapest test of the specific hypothesis
this round produced (reuses all existing Stage C-lite infrastructure,
just needs a `use_residual`/`use_layernorm` toggle in
`atlas_nn.stage_c_lite.model`), and it's a better use of the next research
cycle than either (a) the still-open MLP input-layer question or (b)
jumping to a real pretrained model — both remain valid follow-ups, but
neither directly explains today's most interesting and specific new
finding (the mechanism gap) the way this does.

**Standing recommendation, unchanged:** a real pretrained checkpoint
remains the single most valuable possible upgrade if network policy
allows it. Today's finding makes this *more* valuable, not less — it
shows the phenomenon is architecture-general but poorly understood
mechanistically, which is exactly the situation where testing on a real
model would be most informative rather than merely confirmatory.
