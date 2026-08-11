# Next Research Decision

Updated after the capacity sweep (Experiment 6), which confirmed the
slack/capacity hypothesis raised after Experiment 5. Covers Track B
(`atlas_nn`) only — Track A (the pre-existing symbolic active-learning
framework) is unaffected and out of scope for this document.

## 1. What we learned

**Stage A (synthetic matrices):** a purpose-built method (block dictionary
+ affine transform + residual) finds and exploits structure standard
baselines miss, and correctly does not fake compression on random data.
Narrow, verified — loses to SVD on low-rank data and to zlib on exact
block repetition.

**Stage B / Experiment 4 (easy 2-XOR task, one architecture):** training
made behavior far more robust to weight-compression error than tensor
error predicted, and unlocked ~1.5–2× more compression on deeper layers at
matched behavioral quality.

**Experiment 5 (robustness check):** that gain did **not** reproduce on a
harder task (3-way parity) at the same network width — raising real doubt
about whether it was a general training effect.

**Experiment 6 (capacity sweep) — this round, and the strongest evidence
yet:** the gain tracks *spare network capacity relative to task
difficulty*, not task identity or "training" in the abstract:
- Shrinking the easy task's network to a tight fit (width 16) **removed**
  its gain — trained and random-init converge, or trained is worse.
- Growing the hard task's network to width 256 **brought the gain back**,
  in the same layer, at matching-or-larger magnitude (5.33× → 8–16×), with
  all 3 seeds training cleanly (no confound). This is the strongest single
  piece of evidence in the project so far, because it's a predicted
  *reversal* that was then observed, not just a repeated correlation.

**Two other findings surfaced along the way, both important for how future
experiments should be run:**
- Fixed training hyperparameters (epochs, learning rate) **did not
  transfer** to a wider network: 2 of 3 seeds at width 256 on the easy task
  failed to train properly (49% and 68% held-out accuracy vs. the expected
  ~87–91%), producing eye-catching but meaningless compression numbers
  (up to 128×) from compressing a network that never learned anything.
  Excluded from all claims; documented as a process lesson for Stage C.
- The input layer is **consistently flat-to-worse** after training, across
  every task and width tested (not just "no gain," often a measurable
  penalty) — a robust pattern with no explanation yet.

## 2. What failed

- Block-dictionary clustering vs. SVD/zlib on Stage-A structure types it
  wasn't designed for.
- `vector_codebook` never met the 5% behavioral-error bar in Experiment 4.
- The original "training creates compressibility" framing, taken literally
  and generally — Experiment 5 showed it doesn't hold at matched capacity
  on a harder task; only the more specific "training creates compressibility
  *when there's spare capacity to give it*" framing survived Experiment 6.
- Using one fixed set of training hyperparameters across network sizes —
  concretely wrong at width 256 for 2/3 seeds. Any future width/scale
  sweep needs per-scale hyperparameter validation, not reuse.

## 3. What worked

- Turning each observation into a falsifiable prediction and then testing
  it (Experiment 5 falsified the general claim; Experiment 6 confirmed the
  more specific one) is what actually built confidence here — a single
  "interesting result" (Experiment 4) would have been much weaker evidence
  on its own.
- Checking training success (held-out accuracy) before trusting any
  compression number caught the width-256 confound immediately, the same
  way checking output variance caught Experiment 5's degenerate deep
  network. This check should be standard practice going forward, not
  ad hoc.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and now with a real, tested, twice-confirmed mechanism rather than a
single correlation.** The refined claim — trained networks can be
represented with substantially less independent information than dense
storage, in proportion to how much capacity the training task left unused,
while preserving behavior — is supported by:
1. A positive result (easy task, matched width) — Experiment 4.
2. A predicted negative result (hard task, same width) — Experiment 5.
3. A predicted reversal back to positive (hard task, more width) —
   Experiment 6.

That three-step pattern (confirm → falsify a broader claim → confirm a
narrower one) is a meaningfully stronger evidentiary structure than any
single experiment, and it directly answers the mission's core question in
a scoped, honest way: yes, under conditions this project can now name
(spare capacity relative to task difficulty), not "yes, always."

**What's still missing:** a direct measure of "spare capacity" (e.g.
effective rank of activations, or a capacity metric independent of the
compression search itself) rather than inferring it post hoc from
under/oversized network width. And everything so far is CPU-scale
synthetic tasks — the mission's Stage C gate (meaningful, *understood*
advantage before scaling) is arguably now met for "understood," which
changes the calculus on when to move toward a real pretrained model.

## 5. The single most informative next experiment

Two reasonable candidates; recommendation is the first, given the size of
the investment case for Stage C now:

**(a) One more cheap confirmation before scaling up:** directly measure a
capacity proxy (e.g. the effective rank of each layer's activations on the
training set, or simply train/eval loss margin) across all conditions
already run, and check whether it *quantitatively* predicts the observed
compression-gain magnitude (not just its sign) — turning "spare capacity"
from a qualitative story into a testable, continuous relationship. This
reuses only already-collected data (no new training runs needed) and would
either sharpen the mechanism into something Stage C can use to *predict*
where compressibility will appear in a real model, or reveal the story is
still too coarse.

**(b) Move to Stage C:** given three consistent, predicted-and-confirmed
results now in hand, a small pretrained transformer (mission Stage C) on a
real task would test whether the capacity-relative-to-task story holds
outside synthetic data — the next real test of external validity.

Recommendation: **(a) first, then (b).** (a) is nearly free (reuses
existing results, no new compute) and would make the eventual Stage C
experiment much better targeted (know *which* layers to expect gains in in
a real model, based on where they're under-utilized relative to task
difficulty, rather than searching blindly). Stage C itself is a real
investment (dataset/checkpoint decisions, more engineering, more compute)
that deserves to start from the sharpest possible hypothesis.

No architectural blockers for either — (a) needs no new dependencies; (b)
would need a dataset/checkpoint source decision from the user when reached.
