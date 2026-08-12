# Next Research Decision

Updated after Experiment 18 (Stage C real: distilgpt2 smoke test), the
first experiment in this project to use a genuinely pretrained,
externally-trained model. Covers Track B (`atlas_nn`) only.

## 1. What we learned

**Experiments 3–17:** established, refined, and mechanistically explained
the core behavioral-robustness effect (trained networks tolerate
weight-compression error far better than tensor error predicts) across
two from-scratch architectures (MLP, small Transformer) and several
synthetic/self-authored tasks. The Transformer mechanism thread (10, 11,
13, 14, 17) reached full resolution: cross-token mixing drives gain
magnitude, LayerNorm drives rank-decoupling.

**Experiment 18 (this round) — the effect reproduces on a real pretrained
model.** `huggingface.co` was unblocked by a user-side network-policy
change partway through this session (previously blocked, see Experiment
8's note). distilgpt2 (82M parameters, real HF Hub checkpoint, unknown
training procedure) shows the same effect at 4.4–7.6× magnitude — in
range with every from-scratch measurement before it (3–22.7×). The
sharpest single result: `quantize_4bit_block64`'s tensor-level error was
*higher* for pretrained than random-init, yet its behavioral error was
7.6× *lower* — the cleanest demonstration yet that this is about
behavioral robustness specifically, not generic tensor compressibility.
One genuine divergence: the depth pattern (block 3 gains most, block 5
least) is not the same clean monotonic gradient Stage C-lite found —
reported, not smoothed over.

## 2. What failed / remains untested

- The budget-search (achievable-ratio-at-matched-quality) version of this
  experiment — `experiments/run_atlas_nn_stage_c_real_budget_search.py`
  exists but had not been run as of Experiment 18's writeup. This is the
  direct, low-cost next step: turns the qualitative tensor-vs-behavior gap
  into the same "Nx compression" number Experiments 4/6/9 produced for the
  from-scratch models.
- Why the depth pattern differs between distilgpt2 (non-monotonic,
  3 points) and Stage C-lite (monotonic, 2 points) — not investigated;
  could be a real architectural/training difference or simply that 2 data
  points can't distinguish monotonic from non-monotonic in the first
  place.
- Effective-rank / LayerNorm / attention-mixing mechanism questions
  (Experiments 10–17) have not been re-tested on distilgpt2 — the
  mechanism picture from the from-scratch Transformer has not been checked
  against a real pretrained one.
- The MLP input-layer question (Experiments 4, 6, 7, 12, 15, 16) remains
  open — no new lead from this round; Experiment 18 tested Conv1D
  sublayers depth-wise, not an input-embedding-layer analogue.
- Only one pretrained model tested. A second, differently-sized or
  differently-trained checkpoint (e.g. gpt2 vs. gpt2-medium, or a
  different architecture family) would test whether the magnitude/depth
  findings are distilgpt2-specific or general.

## 3. What worked

- The architecture-agnostic `get_weight`/`set_weight`/`evaluate`/
  `snapshot`/`load_snapshot` interface, unchanged since the Stage C-lite
  refactor, worked without modification on a real HF `transformers` model
  using `Conv1D` sublayers (a different module type and weight orientation
  than `nn.Linear`) — a real test of that abstraction's generality, not
  just its reuse across this project's own from-scratch models.
- Scoping `snapshot`/`load_snapshot` to only the tested layers (rather
  than a full 82M-parameter `state_dict()` deep copy per row, as Stage
  B/C-lite do) kept per-row overhead manageable — necessary at this scale,
  not needed at Stage B/C-lite's.
- Deliberately reducing layer/method scope for CPU feasibility, and
  saying so explicitly in the module docstring and the experiment
  writeup, rather than silently running a smaller experiment and
  presenting it as equivalent in power to earlier ones.

## 4. Does the evidence currently support the Atlas hypothesis?

**Yes, and more strongly than before.** Every prior demonstration of the
central effect used a network this project trained itself. Experiment 18
removes that as a possible confound entirely: a real, externally-trained,
82M-parameter checkpoint shows the same effect, in the same magnitude
range, on a task and training procedure this project has no control over
or visibility into. This is the single piece of evidence closest to what
mission Stage C originally asked for.

## 5. The single most informative next experiment

**Run the budget search** (`experiments/run_atlas_nn_stage_c_real_budget_search.py`,
already written, scoped to 6 layers × 4 states for CPU feasibility) to
convert Experiment 18's qualitative finding into an actionable
"achievable compression ratio at 5% behavioral error" number — the same
step Experiment 9 was for Experiment 8. This is queued and ready to run;
no design work remains, only compute time.

After that, in rough priority order: (a) a second pretrained checkpoint
to test whether the depth-pattern divergence and magnitude range are
distilgpt2-specific; (b) revisit the MLP input-layer question, still the
project's longest-standing open thread; (c) a consolidation pass
incorporating Experiment 18 into the project-wide synthesis artifact.
