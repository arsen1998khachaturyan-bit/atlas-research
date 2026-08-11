# Next Research Decision

Updated after the robustness check (Experiment 5), which substantially
revised confidence in Experiment 4's finding. Covers Track B (`atlas_nn`)
only — Track A (the pre-existing symbolic active-learning framework) is
unaffected and out of scope for this document.

## 1. What we learned

**Stage A (synthetic matrices):** a purpose-built method (block dictionary
+ affine transform + residual) finds and exploits structure standard
baselines miss (affine-transformed shared blocks), and correctly does not
fake compression on random data. Narrow, verified, not general — loses to
SVD on low-rank data and to plain zlib on exact block repetition.

**Stage B + Experiment 4 (trained vs. random-init MLP, easy 2-XOR task):**
training made network *behavior* far more robust to weight compression
error than tensor error predicted, and at a fixed 5% behavioral-error bar,
achievable compression ratio was ~1.5–2× higher post-training on the two
deeper layers of a 3-layer MLP — reproduced across 3 seeds.

**Experiment 5 (robustness check) — the key new finding this round:** that
pattern **did not survive** a harder synthetic task (3-way parity, same
architecture) — trained and random-init reached the same or an
inconsistent ratio across all layers and seeds, with one seed even showing
random-init *beating* trained. A parallel check on a deeper (6-layer)
network was invalidated for an unrelated reason: the random-init deep
network turned out to be behaviorally degenerate (constant prediction for
every input, near-zero output variance) — a vanishing-signal collapse from
stacking unnormalized ReLU layers, not a compression phenomenon. That
condition is inconclusive, not negative.

**Combined picture:** the Experiment 4 result should now be read as
*scoped to the easy 2-XOR task it was measured on*, not as a general
"training creates compressibility" law. The working explanation: an easy
task leaves the network with unused representational capacity after
training, and that unused capacity is what showed up as extra compression
headroom — not something intrinsic to "training" as such. A harder task
that uses more of the network's capacity left no such headroom.

## 2. What failed

- Block-dictionary clustering does not help on globally low-rank or
  low-rank-plus-noise synthetic data (Stage A) — SVD is the right tool.
- The structural method loses to zlib on exact block repetition (Stage A).
- `vector_codebook` never met the 5% behavioral-error bar in Experiment 4,
  in either state.
- The input layer showed no post-training compressibility gain in
  Experiment 4, and **Experiment 5 shows that even the layers that did gain
  in Experiment 4 stop gaining once the task is harder** — the effect is
  not just depth-dependent, it's task-dependent, and possibly not a
  training effect at all so much as a slack/capacity effect.
- The "deeper network" robustness check, as designed, is unusable: default
  PyTorch init produces a degenerate (constant-output) random-init network
  at 5 hidden layers / 64 width on this input scale, so the random-vs-
  trained comparison there is confounded by architecture, not by training.

## 3. What worked

- Byte-honest accounting caught real problems immediately in Stage A/4.
- Measuring behavioral error, not just tensor error, is what produced every
  interesting Stage B/4 finding in the first place.
- Running the robustness check at all is what worked best this round:
  it caught a genuine overgeneralization risk (Experiment 4's finding being
  read as general when it was task-specific) *before* any resources were
  spent building Stage C infrastructure around it, and separately surfaced
  a real, reusable engineering fact (deep unnormalized MLPs collapse at
  random init here) that would otherwise have silently corrupted a future
  experiment's "random baseline."

## 4. Does the evidence currently support the Atlas hypothesis?

**More cautiously than last round.** The mission's central question — can
trained weights be represented with less information while preserving
behavior — still has a positive answer for the original easy-task setup,
unchanged (that measurement was multiseed and reproducible; nothing this
round contradicts it happening). But the *generalization* of that finding
— "training creates additional compressible structure" as a property of
training itself — no longer has support beyond the one setup it was
measured on, and one direct test of generalization (harder task, same
architecture) came back negative. The honest current state: **compression
headroom from training, when it appears, may be a function of how much
spare capacity an easy task leaves behind, rather than something training
adds "for free" regardless of task difficulty.** That is a meaningfully
different, more specific hypothesis than the one Experiment 4 seemed to
support, and it is not yet tested directly.

## 5. The single most informative next experiment

**Directly test the "slack/capacity" hypothesis: hold the architecture
fixed and sweep task difficulty (or equivalently, network capacity relative
to a fixed task) to see whether post-training compression headroom tracks
the gap between train accuracy and the minimum capacity needed to solve the
task** — e.g. compare 2-XOR (very easy, current finding) against 3-parity
(hard, Experiment 5, no gain) against something in between (e.g. 3-parity
with a wider/narrower network, or 2-XOR with a deliberately undersized
network) to see if the compression-gain effect reappears exactly where
spare capacity reappears.

Why this one, specifically:
- It turns Experiment 5's negative result into a mechanistic explanation
  rather than leaving "it didn't reproduce" as a dead end — if the
  slack/capacity story is right, the effect should reappear predictably
  once slack is reintroduced (e.g. a deliberately oversized network on
  3-parity), which would be strong, falsifiable confirmation; if it doesn't
  reappear even with added capacity, the slack hypothesis itself is wrong
  and something else is going on.
- It's the cheapest possible next test — reuses 100% of existing
  infrastructure (`atlas_nn.stage_b.budget_search`,
  `atlas_nn.stage_b.model.build_mlp` already supports variable width via
  `hidden_dim`), no new code beyond a parameter sweep script.
- It keeps faith with the mission's explicit gate: don't scale to Stage C
  (real pretrained models, real datasets) until a method/finding
  demonstrates a *robust*, understood advantage at the current stage — right
  now the finding is real but not yet understood well enough to predict
  where it will and won't appear, which is precisely what this experiment
  would establish.

**Separately, lower priority:** fix the deep-network initialization
pathology (e.g. proper He/Kaiming-for-ReLU init with correct gain, or add
LayerNorm) so the depth question from Experiment 5 can actually be answered
once the slack/capacity question above is resolved — not urgent on its own,
since the capacity question is more fundamental and the deep-net check
would only be worth re-running once there's a specific prediction to test
against it.
