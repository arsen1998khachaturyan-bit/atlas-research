# Next Research Decision

Updated after raising Experiment 11 (residual/LayerNorm ablation) from 3
to 8 seeds, which turned a fragile, low-power observation into the
project's most statistically solid mechanistic finding to date. Covers
Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–9:** the core phenomenon (trained networks tolerate
weight-compression error far better than tensor error predicts, scaling
with depth, translating into real achievable-compression-ratio gains) is
well-established across two architectures, three tasks.

**Experiments 7, 10, 11 (first pass):** mechanism-hunting was
inconclusive — effective rank explained the MLP but not (apparently) the
Transformer; residual connections were ruled out; LayerNorm looked
promising but rested on only 2 comparison pairs.

**Experiment 11, strengthened to 8 seeds (this round) — the clearest
mechanistic result in the project:**
- **LayerNorm drives gain magnitude**, reproducibly: 2–4× larger gains
  with it than without, consistent within 15% between the 3-seed and
  8-seed runs (16.3/18.4 with LayerNorm vs. 8.1/4.1 without, pooled over
  n=49–56).
- **Residual connections do not** — removing them changes the gain by
  under 15%, in the opposite direction from the original hypothesis.
- **Experiment 10's r=−0.54 was a small-sample artifact.** With proper
  power, rank-shrinkage correlates *positively* with gain in all 4
  conditions (r=0.23–0.55), and — cleanly — **removing LayerNorm raises
  that correlation** (0.23→0.55, 0.29→0.37). LayerNorm both amplifies the
  gain and decouples it from raw weight-matrix rank structure.
- **The depth gradient (block 1 > block 0) is confirmed independent of
  both features**, now at much better power (4.0×–9.4× across all 4
  conditions) — still unexplained, but now on very solid ground as a
  real, robust phenomenon in its own right.

## 2. What failed / remains untested

- *Why* the depth gradient exists — two architectural features tested and
  ruled out as the primary cause; still open.
- Block-level (rather than pooled) correlations remain noisy (n=21–24,
  signs and magnitudes vary by block) — the pooled per-condition numbers
  are the trustworthy summary, not individual block r values.
- One seed failed to train under `no_residual` — a minor but real
  reminder that removing residual connections isn't free, even though it
  didn't reduce the gain in the seeds that did train.
- The MLP input-layer question (Experiment 12) remains open and untouched
  this round.
- No test yet of whether LayerNorm's effect holds at a different scale
  (more blocks, wider model) or on the MLP (adding LayerNorm to the MLP
  and checking whether it changes *that* architecture's already-partial
  rank correlation, Experiment 7, would be a natural cross-check).

## 3. What worked

- Treating a promising-but-underpowered finding as provisional and
  explicitly re-running it with more seeds — rather than either trusting
  it prematurely or discarding it as noise — is what turned this from a
  shaky observation into the project's strongest mechanistic claim. The
  numbers barely moved between 3 and 8 seeds, which is itself good
  evidence the underlying effect is real and was just poorly estimated
  before, not spurious.
- Reporting the correction to Experiment 10 plainly (r=−0.54 → understood
  as a small-sample artifact, true relationship positive) rather than
  quietly editing history keeps `docs/BEST_RESULTS.md` trustworthy as a
  record of how understanding actually evolved.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the mechanism picture is now the best-evidenced it's been.**
The project has moved from "the phenomenon is real but we don't know
why" (yesterday's honest state) to "the phenomenon is real, and we have
a specific, replicated, moderately well-understood partial mechanism
(LayerNorm affects magnitude and rank-decoupling; residual connections
don't; something else still drives the depth gradient itself)." This is
a meaningfully more mature scientific position than either "no mechanism
found" or a premature single-experiment claim would have been.

## 5. The single most informative next experiment

**Test whether the LayerNorm effect is specific to Transformers or a
general property of normalization: add LayerNorm to the MLP (Stage B)
and re-run Experiment 7's rank-shrinkage correlation.** If adding
LayerNorm to the MLP *reduces* its already-partial rank correlation
(mirroring what removing LayerNorm did to the Transformer, in reverse),
that's strong, cheap, cross-architecture confirmation that LayerNorm
specifically (not something Transformer-specific like attention) is the
operative factor. If it doesn't change anything on the MLP, that narrows
the LayerNorm hypothesis to an attention-specific interaction instead.

Why this one: it's the natural next falsifiable step from today's
strengthened finding, cheap (reuses `atlas_nn.stage_b.model.build_mlp`,
just needs an optional LayerNorm-after-each-hidden-layer toggle), and
would settle whether "LayerNorm" or "LayerNorm-in-a-Transformer" is the
right level of generality for the mechanism — a distinction the current
evidence can't yet make since LayerNorm has only been tested on one
architecture.

**Other queued options, lower priority given today's result:** the MLP
input-layer question (Experiment 12, still open, no strong lead);
a genuine pretrained-model test (still blocked by network policy, still
the single highest-value addition if that changes).
