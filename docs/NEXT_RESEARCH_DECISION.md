# Next Research Decision

Updated after Experiment 15 (controlled re-run of the input-layer noise
sweep), which closed Experiment 12's remaining loose end and left the
input-layer question more firmly unexplained than before. Covers Track B
(`atlas_nn`) only.

## 1. What we learned

**Experiments 3–9:** the core phenomenon (trained networks tolerate
weight-compression error far better than tensor error predicts, scaling
with depth) is well-established across two architectures, three tasks.

**Experiments 10–14:** four architectural factors individually tested —
residual connections ruled out; effective rank partially explains the
MLP; LayerNorm drives rank-decoupling (generalizes across architectures)
and a secondary magnitude effect; attention is the dominant magnitude
driver and partly explains the depth gradient. The Transformer mechanism-
hunting thread reached a well-resolved natural stopping point.

**Experiment 12:** the MLP input layer's flat-to-negative post-training
compressibility does not track task-irrelevant input noise fraction — but
flagged an unresolved data-size confound.

**Experiment 15 (this round) — the confound fixed, the null result
strengthened:** with training data scaled up to equalize task difficulty
across the noise-fraction sweep (held-out accuracy tightened from a
73.6%–98.4% band to 91.0%–99.6%), the result holds and is cleaner: gain
stays at 0.73–1.00 across the entire 0%–97% noise-fraction range, 8
seeds. The one loose end from Experiment 12 — a weak positive hint at
zero noise — **disappeared** under better power and control (exactly
1.00, flat). Noise fraction is now firmly ruled out, not just weakly
disfavored.

**Where the input-layer question stands after five experiments (4, 6, 7,
12, 15):** two candidate explanations tested and ruled out (effective
rank only weakly/inconsistently related; noise fraction cleanly
unrelated). The behavior itself — no gain, sometimes a penalty, uniquely
among all layer types in every architecture tested — remains real,
reproducible, and unexplained.

## 2. What failed / remains untested

- Noise-fraction hypothesis — now cleanly ruled out at good power, not
  just suggestively disfavored.
- No new hypothesis for the input layer has been generated this round —
  Experiment 15 closed a loose end rather than opening a new lead.
- Whether the input layer's behavior relates to something identified in
  the Transformer mechanism work (e.g., does an input-facing layer in the
  Transformer show the same pattern? Not yet checked — the Stage C-lite
  experiments never singled out an "input-layer" analog since the
  embedding layer isn't a compressible `nn.Linear` in the same sense).
- A genuine pretrained-model test — still blocked by network policy.

## 3. What worked

- Treating Experiment 12's own flagged confound as unfinished business
  and returning to fix it, rather than letting a caveat sit indefinitely,
  produced a cleaner, more trustworthy null result and eliminated a loose
  thread (the zero-noise hint) that could have misled future work if left
  unchecked.
- Verifying the fix empirically (checking the accuracy band actually
  tightened) before committing to the full 8-seed run avoided wasting
  compute on a re-run that might not have actually fixed the confound.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, unaffected by this round — the input-layer question is a detail
of the mechanism, not the core phenomenon.** The central claim (trained
networks tolerate compression far better than tensor error predicts,
concentrated in specific layers, deepening with representational depth)
remains strongly evidenced regardless of whether the input layer's
distinct non-participation is ever explained. This round's contribution
is negative-but-valuable: it removes one more wrong explanation from
consideration and confirms the earlier null result wasn't a measurement
artifact.

## 5. The single most informative next experiment

No strong, specific lead remains for the input-layer question after two
hypotheses (rank, noise fraction) have been tried and both failed. Three
reasonable paths, in rough order of promise:

**(a)** Check whether an input-facing layer in the *Transformer*
architecture shows the same pattern — would test whether "input layers
are different" is a general phenomenon (worth a real hypothesis-generation
effort) or MLP-specific (in which case the input-layer question may be
lower priority than it's been treated).

**(b)** Try a structural/informational hypothesis instead of a capacity
one: does the input layer's behavior relate to it being the only layer
whose incoming activations are never normalized or reshaped by a prior
layer (raw features in, vs. every other layer receiving already-processed
activations)? Testable cheaply by adding an input-normalization step
(e.g. z-scoring or a fixed random projection) before the first trainable
layer and checking whether *that* first trainable layer (now not
literally seeing raw input) behaves differently.

**(c)** Deprioritize this question for now given two clean negative
results and no strong remaining lead; consider it a standing open
question in `docs/CURRENT_STATE.md` rather than continuing to spend
cycles without a promising hypothesis to test.

**Standing recommendation, unchanged:** a real pretrained checkpoint
remains the single highest-value possible addition to this research line
if network policy allows it at some point.
