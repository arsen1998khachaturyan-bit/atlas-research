# Next Research Decision

Updated after Experiment 21 (gpt2 budget search), which falsified a
pattern this project had trusted since Experiment 9. Covers Track B
(`atlas_nn`) only.

## 1. What we learned

**Experiments 8-9, 19:** "achievable-compression gain increases with
depth" held consistently -- Stage C-lite's 2-block from-scratch
Transformer, then distilgpt2's real 6-block checkpoint, both showed later
blocks gaining more from training than earlier ones.

**Experiment 20:** two different fixed compression methods disagreed with
each other about gpt2's depth pattern -- a warning sign, correctly not
treated as resolved without the actual budget search.

**Experiment 21 (this round) -- the pattern reverses on gpt2, not just
differs.** Block 0's mean gain (24.8x) is over 3x block 11's (7.1x) -- the
opposite direction from distilgpt2. The largest single number in this
experiment (384x compression at 5% behavioral error) is on the *first*
block, the position that gained *least* on distilgpt2. What still holds
across both models: the structural finding (training unlocks
structure-aware compression methods, not just better ratios within one
method) and the overall depth-pooled magnitude range (Experiment 20,
1.4x-8x at fixed parameters).

## 2. What failed / remains untested

- "Gain increases with depth" is now falsified as a general claim -- it
  held on 2 of 3 real/from-scratch Transformers tested and reversed on
  the third. Two candidate explanations, neither tested: (a) gpt2's
  larger scale/depth changes where slack accumulates: weakly supported,
  since Stage C-lite (2 blocks) and distilgpt2 (6 blocks) agreed with
  each other before gpt2 (12 blocks) broke the pattern, so "more depth"
  alone doesn't obviously predict a full reversal; (b) distilgpt2's
  knowledge-distillation training (vs. gpt2's from-scratch causal-LM
  objective) is the real variable, not scale -- distinguishing (a) from
  (b) needs a third model with yet another training recipe.
- No mechanism work (effective rank, LayerNorm, attention mixing --
  Experiments 10-17) has been checked against either real pretrained
  model.
- Only 2 of each model's many layers' depth positions tested (6-layer
  subsets). Whether gpt2's reversal holds at intermediate blocks (not
  just first vs. last) is untested.
- The MLP input-layer question remains the project's longest-standing
  untouched thread.

## 3. What worked

- Not trusting Experiment 20's fixed-method depth readings and running
  the actual budget search anyway, exactly per the project's established
  "verify before trusting a convenient pattern" discipline -- this is the
  second time in three rounds (after Experiment 10's r=-0.54 turning out
  to be a small-sample artifact) that skepticism toward an early reading
  caught something real, though this time in the opposite direction: the
  skepticism was rewarded by finding a genuine reversal, not an artifact.
- Treating the reversal as a reportable finding in its own right (a new
  VERIFIED RESULT plus explicit correction notes on the two entries it
  narrows) rather than either hiding it or overriding the distilgpt2
  numbers.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, on the core claim; the depth-gradient claim is now narrower.** The
central finding -- trained networks tolerate weight-compression error far
better than tensor error predicts, and training unlocks entire families
of structure-aware compression methods -- holds across every architecture
and every real model tested, including both real pretrained checkpoints
in this round. The specific, appealing claim about *where* in the network
this concentrates does not generalize and should not be treated as
established until a third model resolves which of (a) or (b) above is
the real driver.

## 5. The single most informative next experiment

**A third pretrained model with a different training recipe** -- the
direct way to distinguish scale/depth (hypothesis a) from training
procedure (hypothesis b) as the explanation for the depth-gradient
reversal. A second from-scratch (non-distilled) causal LM near
distilgpt2's size would isolate training procedure; a second distilled
model would isolate distillation specifically. Either result is
informative: agreement with gpt2 supports (b), agreement with distilgpt2
supports (a) or narrows it to "distillation specifically," and a third,
different pattern would mean neither simple story is right.

After that, in rough priority order: (a) revisit the MLP input-layer
question, now clearly the project's longest-untouched open thread; (b)
check mechanism questions (effective rank, LayerNorm, attention) against
a real pretrained model for the first time; (c) fold Experiments 18-21
into the project-wide synthesis artifact, which currently stops at
Experiment 17 and does not reflect any of the real-pretrained-model work,
including this reversal.
