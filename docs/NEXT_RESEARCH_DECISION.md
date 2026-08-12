# Next Research Decision

Updated after Experiment 17 (attention parameter-count disentanglement),
which closed the last open ambiguity in the Transformer mechanism-hunting
thread. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–9:** the core phenomenon (trained networks tolerate
weight-compression error far better than tensor error predicts, scaling
with depth) is well-established across two architectures, three tasks.

**Experiments 10–14:** four architectural factors tested — residual
connections ruled out; effective rank partially explains the MLP;
LayerNorm drives rank-decoupling (generalizes) and a secondary magnitude
effect; attention is the dominant magnitude driver, but `NoMixingAttention`
confounded "no mixing" with "fewer parameters."

**Experiment 17 (this round) — the confound resolved, decisively:** a
same-parameter-count, non-mixing attention replacement lands at 4.4 mean
gain — close to the original parameter-poor no-mixing result (3.4), far
from full attention (16.3). Restoring ~12,000 parameters without
restoring mixing changed almost nothing. **Cross-token mixing, not
parameter count, is confirmed as what attention contributes.**

**Experiments 12, 15, 16 (MLP input-layer thread, same period):** three
hypotheses tested and ruled out (effective rank, noise fraction,
raw-vs-processed input) — question remains open, no strong new lead.

**Where the project stands now:** the Transformer mechanism-hunting
thread (Experiments 10, 11, 13, 14, 17) has reached full resolution —
every architectural factor tested has a specific, confirmed role, with no
remaining ambiguity of the kind Experiment 17 just closed. This is very
likely the natural stopping point for that specific thread; further work
there would mean either new architectural factors (not yet identified)
or moving beyond this toy-scale setup entirely.

## 2. What failed / remains untested

- Parameter count as an explanation for attention's magnitude effect —
  ruled out decisively.
- *Why* cross-token mixing specifically produces this effect (information-
  theoretic? optimization-dynamics? something else?) — established that
  it does and by how much, not the deeper mechanistic reason.
- The MLP input-layer question — three hypotheses down, no new lead
  generated this round (that thread was worked in parallel, see the
  session's other work).
- A genuine pretrained-model test — still blocked by network policy.

## 3. What worked

- Treating Experiment 14's own acknowledged confound (mixing vs.
  parameters) as unfinished business, the same way Experiment 15 treated
  Experiment 12's confound, continues to pay off — both times, closing a
  self-identified gap either confirmed the original finding more
  precisely or ruled out an alternative explanation cleanly.
- Building in an internal consistency check (reproducing Experiment 14's
  exact numbers within the same script before trusting the new condition)
  caught nothing wrong here, but is exactly the kind of check that would
  have caught a bug if one existed — worth continuing as standard practice
  for any experiment that extends a previous one.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, with the Transformer mechanism now understood about as precisely
as this project's toy-scale setup allows.** Four architectural factors,
each isolated to a specific role, with the last remaining ambiguity
(mixing vs. parameters) now resolved. This is a natural point to
consider that particular investigative thread complete, not because
every question is answered, but because the remaining questions (why
does mixing have this effect, mechanistically) would need different
tools (e.g. direct analysis of attention patterns, information-theoretic
measures) rather than more ablations of the same kind.

## 5. The single most informative next experiment

With the Transformer-mechanism thread naturally concluded and the
input-layer thread having run through its most obvious hypotheses, three
reasonable directions remain, none clearly dominant:

**(a)** Check whether an input-facing layer in the Transformer shows a
pattern analogous to the MLP's input layer (queued since the previous
round, not yet done) — would clarify whether "input layers are
different" generalizes across architectures.

**(b)** A genuine pretrained-model test, if network policy ever allows
it — still the single highest-value possible addition to this research
line, and now better-motivated than ever: four session's worth of
ablation work has produced a specific, falsifiable mechanistic picture
(mixing → magnitude, LayerNorm → decoupling) that a real pretrained model
could directly test for the first time, rather than just adding another
confirmation of the phenomenon in isolation.

**(c)** Step back from ablation-style mechanism-hunting for now (both the
Transformer and MLP threads have each produced a full, well-documented
set of tested/ruled-out hypotheses) and consolidate: review the full
project (17 experiments) for internal consistency, update
`docs/CURRENT_STATE.md` to reflect the mature state of both tracks, and
let the user decide the next investment given the current evidence base
rather than continuing to generate new hypotheses autonomously.

Given the volume of ablation work completed and the natural conclusion
reached on the Transformer thread specifically, **(c)** is a reasonable
default absent other direction — this is a good point for a consolidation
pass rather than another open-ended hypothesis hunt.
