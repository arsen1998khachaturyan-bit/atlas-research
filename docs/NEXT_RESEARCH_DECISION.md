# Next Research Decision

Updated after Experiment 14 (attention ablation), which resolved
Experiment 13's open question and produced the project's most complete
mechanistic picture to date. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–9:** the core phenomenon (trained networks tolerate
weight-compression error far better than tensor error predicts, scaling
with depth) is well-established across two architectures, three tasks.

**Experiments 10–13:** three architectural factors tested piecemeal
(effective rank, residual connections, LayerNorm) — residual connections
ruled out; LayerNorm found to have two separable effects, one
generalizing (rank-decoupling) and one architecture-dependent in sign
(magnitude).

**Experiment 14 (this round) — the magnitude-sign question resolved,
with convergent evidence:**
- **Attention is the dominant driver of gain magnitude** — removing it
  crashes the gain 79% (16.3→3.4), a bigger effect than removing
  LayerNorm (50%, 16.3→8.1).
- **Convergent confirmation:** a Transformer with attention artificially
  disabled (3.4) lands close to the real MLP with LayerNorm added but no
  attention at all (2.9, Experiment 13) — two independent routes to "no
  attention" agree, which is stronger evidence than either alone.
- **Attention also partly explains the depth gradient** — block1/block0
  ratio drops from 6.9× to 2.4× without attention (vs. only to 4.0×
  without LayerNorm alone) — new information beyond the original
  hypothesis; attention was not expected to touch the gradient itself.
- **LayerNorm remains the dominant driver of rank-decoupling**,
  regardless of attention — confirms Experiment 13's other half is
  correctly attributed to LayerNorm specifically, not confounded with
  attention.

**Current mechanism picture, the most complete yet:** attention drives
magnitude and part of the depth gradient; LayerNorm drives rank-
decoupling and a smaller, largely independent share of magnitude. Four
architectural ablations (rank alone, residual, LayerNorm, attention)
across two architectures have now isolated which factors matter for
which specific aspect of the phenomenon.

## 2. What failed / remains untested

- *Why* attention and LayerNorm have these specific effects mechanistically
  — only *that* they do, and by how much, is established. E.g. does
  attention's magnitude effect come from cross-token information mixing
  specifically, or from something else about `nn.MultiheadAttention`
  (its extra parameters, its softmax nonlinearity)? `NoMixingAttention`
  removes mixing and reduces parameter count simultaneously — not fully
  disentangled.
- Whether this 4-factor picture holds at a different scale (more blocks,
  wider model) or on a harder task.
- The MLP input-layer question (Experiment 12) — still open, now the
  longest-standing unresolved question in the project, untouched since
  that experiment.
- A genuine pretrained-model test — still blocked by network policy.

## 3. What worked

- Following the chain of falsifiable hypotheses (Exp 10 → 11 → 13 → 14)
  to its natural conclusion rather than stopping at an ambiguous
  intermediate result (Experiment 13's "reverses, unexplained") is what
  produced today's clean resolution.
- Designing Experiment 14 to produce a *convergence check* (comparing
  against Experiment 13's independently-collected MLP number) rather than
  just a fresh isolated measurement is what made the result more
  convincing than a bare directional confirmation would have been.
- Consistent 8-seed power across the last three ablations (11, 13, 14)
  avoided repeating Experiment 11's original underpowered-first-pass
  mistake.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and the mechanism is now understood in more useful detail than at
any prior point in the project.** Four architectural factors have been
individually tested and assigned distinct, specific roles rather than
left as one bundled "something about training" explanation. This is a
substantially more mature scientific position than the project had even
one round ago, and it was reached through exactly the kind of chained,
falsifiable experimentation the mission's process asks for.

## 5. The single most informative next experiment

Two reasonable next steps, neither urgent:

**(a)** Disentangle `NoMixingAttention`'s two simultaneous changes
(removes cross-token mixing AND reduces parameter count) — e.g. compare
against a per-token `nn.Linear` sized to match attention's parameter
count, to check whether the magnitude effect is really about mixing or
just about attention's extra capacity.

**(b)** Finally return to the MLP input-layer question (Experiment 12),
the project's oldest unresolved thread, now that the Transformer
mechanism-hunting has reached a natural, well-resolved stopping point.

**(c)**, unchanged: a genuine pretrained-model test remains the single
highest-value addition if network policy allows it at some point.

No strong reason to prefer (a) over (b) from the evidence alone — both
are cheap, both close real open threads. This is a reasonable point to
check with the user on preference, or default to (b) since it's been
open longest.
