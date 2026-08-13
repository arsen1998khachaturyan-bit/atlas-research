# Current State of atlas-research

Status as of 2026-08-10, base commit `94aa345` ("Initial Atlas Research
Framework v1.0").

This document is written before any large new experiments are run, per the
Project Atlas process rules. It has two parts:

1. **Track A (existing, as of this commit): symbolic active-learning
   framework.** This is what the repository actually contained before this
   session.
2. **Track B (new, this session): neural-network weight-representation
   research (`atlas_nn/`).** This is the actual subject of the Project Atlas
   research mission (compact/generative representations of trained NN
   weights). It did not exist in the repository before this session and is
   being built up incrementally, in its own module, without touching Track A.

Both tracks share the repository name "Atlas" but investigate unrelated
hypotheses. They are kept side by side rather than merged, so Track A's
working, tested code and results are preserved unchanged.

---

## Track A — Symbolic active-learning framework (pre-existing)

### What it does

Track A studies query efficiency for exact learning of a hidden boolean
program over fixed-length binary sequences from a black-box oracle. The
concrete question it answers: given a pool of candidate symbolic programs
(the "hypothesis space") and a hidden target program from that pool, how many
membership queries does a learner need to identify the target exactly,
compared to passive/random query orders?

This is standard active-learning / exact-learning territory (in the spirit
of Angluin-style query learning), applied to a synthetic DSL of boolean
sequence predicates. It has no connection to neural-network weights.

### Repository structure (Track A)

```
atlas/
  domain.py              enumerate binary sequences up to a max length
  programs.py             Program: (name, complexity_bits, callable)
  dsl.py                  atom programs (palindrome, mod-k patterns,
                           starts/ends/contains-pattern) + AND/OR/XOR/NOT
                           composition
  oracle.py                Oracle ABC + CachedOracle (query budget, caching)
  hypothesis.py           behavior_matrix/behavior_signature, and
                           generate_hypothesis_pool: builds a deduplicated
                           pool of programs by behavioral signature
  learners/
    common.py             LearningResult, fidelity()
    passive.py             PassiveLearner: fixed pseudo-random query order
    random_query.py        RandomLearner: shuffled query order
    active.py               ActiveInformationGainLearner: greedy version-space
                            reduction (min(ones, zeros) split heuristic)
  benchmarks/
    protocol.py             ValidationProtocol (frozen pool size / seeds /
                            tasks-per-seed defaults)
    runner.py               run_multiseed_validation(): runs all 3 learners
                            over multiple seeds/tasks, aggregates win counts
                            and mean query counts, save_json()
experiments/
  run_smoke.py               tiny single-seed run -> results/smoke_results.json
  run_blind_validation.py    (not yet inspected in depth this session)
  run_multiseed_validation.py
tests/
  test_domain.py, test_learners.py, test_oracle.py   (3 tests total)
notebooks/
  Atlas_Framework_v1_SmokeTest.ipynb
```

### Benchmark methodology (Track A)

- A pool of `pool_size` symbolic programs is generated per seed by combining
  DSL atoms via AND/OR/XOR/NOT up to depth 3, deduplicated by their exact
  behavioral signature over the full domain (so no two pool members compute
  the same function).
- For each of `tasks_per_seed` sampled target programs, three learners try to
  identify the target from the pool using only membership-query answers:
  - **Passive**: queries sequences in a fixed, length-grouped pseudo-random
    order.
  - **Random**: queries sequences in a fully shuffled random order.
  - **Active**: greedily picks the query that best splits the current
    "version space" (remaining candidates consistent with observed answers)
    using a `min(ones, zeros)` information-gain proxy.
- Metrics recorded per task: `queries` (how many membership queries until
  only one candidate remains or the budget/domain is exhausted),
  `remaining_candidates`, and `fidelity` (fraction of domain points where the
  recovered program's behavior matches the true target's behavior — this is
  the "behavioral preservation" check, analogous in spirit to what Project
  Atlas asks for compression methods).
- `run_multiseed_validation` aggregates mean query counts and head-to-head
  win counts (active vs. passive, active vs. random) across seeds
  `(101, 202, 303, 404, 505)`.

### Results currently obtainable

Running the smoke benchmark this session (`pool_size=250, max_length=6,
tasks_per_seed=5, seed=101`) reproduces the qualitative claim in the README:

```
mean_passive_queries = 40.0
mean_random_queries  = 22.4
mean_active_queries  = 11.2
active_wins_vs_passive = 5/5
active_wins_vs_random  = 4/5
all_active_full_fidelity = True
```

The active learner used substantially fewer queries than passive/random
while recovering a program with 100% behavioral fidelity to the true target,
on this one seed/config. `tests/` (3 tests) pass with `pytest -q`.

### What can and cannot currently be concluded (Track A)

**Can conclude:** on this synthetic DSL and this frozen protocol, the
information-gain active learner is more query-efficient than passive/random
baselines, and it always recovers an exactly-fidelity-matching program in the
smoke run. The README already states this correctly and conservatively.

**Cannot yet conclude:**
- Anything about performance on non-synthetic (real automata-learning /
  program-synthesis) benchmarks — the README's own "Next scientific
  milestone" already flags this gap.
- Anything about comparison to published active-learning baselines (e.g.
  L*-style algorithms, other query-selection heuristics) — none are
  implemented.
- Robustness at larger `max_length`/pool sizes — only small smoke-scale
  configs were exercised this session; `run_blind_validation.py` and
  `run_multiseed_validation.py` (larger, seeded runs) were not re-run in
  this session due to time budget and because Track A verification was not
  the focus of this session's work.

### Bugs / methodological weaknesses noticed (Track A)

- `TEST_REPORT.txt` at the repo root contains an unrelated stack trace from
  an external "artifact_tool" spreadsheet-warmup process, appended after the
  genuine `pytest` output (`3 passed`). This looks like accidental log
  concatenation from whatever environment produced the file, not a real
  failure in this codebase — the actual `pytest -q` run in this session
  reproduces `3 passed` cleanly with no trace of that error. Worth deleting
  or regenerating that file so it doesn't mislead readers.
- `active_wins_vs_random` was 4/5 rather than 5/5 in the smoke run — not a
  bug, just a reminder that "active always wins" is not universally true
  even in-distribution; multi-seed aggregation is the right way to read this
  (already implemented).
- No baseline from the exact-learning literature (e.g. an L*-style
  algorithm) is implemented, so "active beats passive/random" is a weaker
  claim than "active beats a strong published method," matching the README's
  own caveat.

### Most promising next steps (Track A)

Out of scope for this session (Track B is the actual mission target), but
noted for completeness: implement one published active/exact-learning
baseline, and evaluate on at least one non-synthetic benchmark, per the
README's stated milestone.

---

## Track B — Neural-network weight representation research (`atlas_nn/`, new)

This is the actual subject of the Project Atlas research mission: whether
trained neural-network weight tensors contain structure (low-rank, repeated,
transformed, generative) that lets them be stored/executed with substantially
less independent information than explicit dense tensors, while preserving
behavior.

Nothing about this existed in the repository before this session. Building
it required creating a new module rather than modifying Track A, per the
"repurpose, don't merge" decision (Track A stays intact and untouched).

See `docs/RESEARCH_LOG.md` for the experiment-by-experiment log and
`docs/BEST_RESULTS.md` / `docs/NEXT_RESEARCH_DECISION.md` for verified
results and next steps. As of this update, Stage A (synthetic matrices),
Stage B (trained-vs-random-init small MLP, `atlas_nn/stage_b/`, including
a behavior-budgeted compression search and a capacity-sweep/effective-rank
analysis), and Stage C-lite (a small Transformer trained from scratch on a
real sentiment task, `atlas_nn/stage_c_lite/`) have all been implemented
and run; `torch` (CPU) was added as the `stage_b` optional dependency
(`pip install -e ".[stage_b]"`) — Stage A remains numpy-only.

**Note on Stage C-lite:** mission Stage C calls for a "manageable open
pretrained model." This session's network policy blocks `huggingface.co`
(403, confirmed via the egress proxy's own status endpoint) so a literal
pretrained checkpoint isn't reachable here; per the user's choice,
`atlas_nn/stage_c_lite/` is an agreed substitute — a genuinely different
(attention-based) architecture on a real (self-authored) text task, not a
pretrained model. See `docs/RESEARCH_LOG.md` Experiment 8 for what this
did and didn't establish, and `docs/NEXT_RESEARCH_DECISION.md` for the
standing recommendation to prioritize a real pretrained-model source if
one becomes reachable.

**Overnight autonomous session (Experiments 9–12):** with the user's
explicit permission to continue working unattended and report back, this
session ran a chained sequence of follow-up experiments, each a
falsifiable test of the previous result: quantified the Stage C-lite
depth gradient as an achievable-compression-ratio number (Experiment 9);
tested whether the MLP's effective-rank mechanism (Experiment 7)
transfers to the Transformer (Experiment 10 — it does not); ablated
residual connections and LayerNorm independently to test candidate
explanations (Experiment 11 — LayerNorm plausibly drives gain magnitude,
residual connections do not, and the depth gradient survives removing
both); and tested whether the MLP's still-unexplained input-layer
behavior tracks task-irrelevant input noise fraction (Experiment 12 — it
does not). Every experiment's code, results, and honest writeup
(including corrections to earlier over-confident readings, e.g. the
Experiment 10 correlation number) was committed and pushed individually;
see `docs/RESEARCH_LOG.md` for the full account and
`docs/NEXT_RESEARCH_DECISION.md` for the current recommendation to pause
autonomous mechanism-hunting and hand prioritization back to the user.

**Second autonomous session (Experiments 15–17), also with explicit
permission to work unattended:** continued the chained-hypothesis
discipline to full resolution on two threads. Transformer mechanism
(Experiments 10, 11, 13, 14, 17): four architectural factors — effective
rank, residual connections, LayerNorm, attention — each isolated to a
specific, tested role. Current picture: cross-token mixing (attention
specifically, confirmed via a parameter-count-matched ablation in
Experiment 17, not just "having an attention-shaped sublayer") drives
gain magnitude and part of the depth gradient; LayerNorm drives
rank-decoupling (confirmed to generalize to the MLP, Experiment 13) and a
smaller, independent magnitude contribution. This thread reached a
natural stopping point — no remaining ambiguity of the kind resolved
along the way, further progress would need new tools (e.g. direct
attention-pattern analysis) rather than more ablations of the same kind.
MLP input-layer question (Experiments 12, 15, 16): three specific
hypotheses tested and ruled out (effective rank, noise fraction,
raw-vs-processed input) at good statistical power (8 seeds); the behavior
itself — no post-training compressibility gain, sometimes a penalty,
unique among all layer types in every architecture tested — remains real,
reproducible, and unexplained. See `docs/RESEARCH_LOG.md` for the full
17-experiment account and `docs/NEXT_RESEARCH_DECISION.md` for current
options (Transformer input-layer analog check, a real pretrained-model
test if network policy allows, or a consolidation pass).

**Stage C (real), Experiment 18:** partway through this session the user
changed the environment's network policy; `huggingface.co` (previously
blocked, see the note above) is now reachable, confirmed directly (`200`,
real API data). `atlas_nn/stage_c_real/` targets distilgpt2 (82M
parameters, a genuinely pretrained HF Hub checkpoint this project did not
train) using the same architecture-agnostic interface as Stage B/C-lite.
The core behavioral-robustness effect reproduces cleanly (4.4–7.6× gain
across every method with room to show a difference), including one
sharper-than-before demonstration (`quantize_4bit_block64`: tensor error
*higher* for pretrained, behavioral error 7.6× *lower*) — the strongest
single piece of evidence in the project that this is a general property
of trained networks, not an artifact of this project's own training
procedure. One divergence from Stage C-lite: the depth pattern here is
non-monotonic (block 3 > block 0 > block 5), not a clean gradient — see
`docs/RESEARCH_LOG.md` Experiment 18.

**Experiment 19 (budget search, completed):** resolved Experiment 18's
odd depth pattern — letting each layer pick its own best method/parameter
(rather than a fixed rank-4 SVD probe) shows a clean, strongly monotonic
gradient after all: block 0 gains ~1.3× from training, block 5 gains
~22.3×, with one layer (`transformer.h.5.mlp.c_proj`) reaching **307.2×
compression at 2.3% behavioral error** post-training (SVD rank 2) vs.
5.33× pre-training — the largest compression number found anywhere in
this project, on a real model this project did not train. Also: every
random-init search's best method was the safe `quantize` fallback;
most pretrained searches were won by a structure-aware method instead
(low-rank, codebook, block-dictionary) — training doesn't just improve
ratios, it makes entire method families viable. Full wall-clock cost:
~3.3 hours on CPU for the 24-search sweep.

**Experiment 20 (cross-model check, completed):** reran Experiment 18's
methodology on `gpt2` (124M, 12 blocks — larger, undistilled). The
effect's magnitude generalizes cleanly (1.4×–8.0× gain range, vs.
distilgpt2's 1.2×–7.6×, tight seed reproducibility on both). The depth
*pattern*, however, is not consistent even within gpt2 itself:
`svd_rank4` peaks at the middle block (echoing distilgpt2's own
non-monotonic Experiment 18 result); `atlas_block_dict` on the same
layers shows the opposite, monotonically-decreasing-with-depth shape —
independent confirmation of Experiment 19's lesson that no single fixed
method's depth reading should be trusted; only a full budget search
(each layer picking its own best method) is reliable. `atlas_nn/
stage_c_real/model.py` is now generalized to accept any GPT-2-family
model by name.

**Experiment 21 (gpt2 budget search, completed) — the depth gradient is
REVERSED, not just differently shaped.** Block 0's mean gain (24.8×) is
over 3× block 11's (7.1×) — distilgpt2 showed the opposite (block 0: 1.3×,
block 5/last: 22.3×). The single largest number found (384× compression
at 5% behavioral error, SVD, `transformer.h.0.attn.c_proj`) is on the
*first* block. This falsifies "gain increases with depth" as a general
Transformer property — it held across Stage C-lite (2 blocks) and
distilgpt2 (6 blocks) before gpt2 (12 blocks) broke it. What still holds
on both real models: the structural finding (training unlocks
structure-aware compression methods entirely, not just better ratios)
and the overall 1.4×–8× fixed-parameter magnitude range (Experiment 20).
Two unverified explanations for the reversal — model scale/depth vs.
distilgpt2's knowledge-distillation training procedure specifically —
were not yet distinguished as of Experiment 21.

**Experiments 22–23 (gpt2-medium, completed overnight) — narrowed to one
better-supported hypothesis.** `atlas_nn/stage_c_real/model.py` and the
budget-search scripts were generalized to any GPT-2-family model.
gpt2-medium (355M, 24 blocks — same non-distilled training recipe as
gpt2, but 2× its depth and ~3× its parameters) shows the *same*
early-block-dominant depth gradient as gpt2, at a strikingly similar
relative margin (~3.5:1 early:late for both). Two very differently-scaled
models sharing gpt2's non-distilled training recipe agree with each
other; the one distilled model (distilgpt2) disagrees with both — a real,
if not conclusive, argument that training procedure (distillation),
not model scale, drives the reversal. The single largest number found
(512× at 5% behavioral error, SVD, on the first block) again matches
gpt2's pattern, not distilgpt2's. The structural finding (training
unlocks structure-aware compression methods entirely) has now held
without exception across 42 total budget searches on 3 models.

**Infrastructure note:** this container restarted unannounced twice
during the overnight session, killing two multi-hour runs with zero
partial results saved. Built `atlas_nn/stage_c_real/parallel_budget_search.py`
(multiprocess, one worker per model instance, ~1.8–4× faster) with
per-layer checkpointing (survives a restart with at most one layer's
work lost) — verified correct (identical results to sequential,
correct resume behavior) before being trusted for the real gpt2-medium
run.

**Experiment 24 (effective rank, completed overnight) — a second
independent confirmation of the same distilled-vs-non-distilled split.**
Cheap, no-restart-risk follow-up (reuses already-computed compression-gain
numbers, just one SVD per already-tested layer): effective-rank
shrinkage correlates strongly with compression gain on gpt2 (r=0.83, the
strongest correlation found anywhere in this project), moderately on
gpt2-medium (r=0.49), and has the *wrong sign* on distilgpt2 (r=−0.29).
Two unrelated analyses (depth-gradient shape, effective-rank correlation)
now split the same three models the same way — by training procedure,
not size — strengthening the case from Experiments 21/23. Labeled
"partial" in `docs/BEST_RESULTS.md` due to a real effective-sample-size
caveat (random-init effective rank barely varies across seeds, so n=18
per model is closer to n≈6).

**Experiment 25 (weight-delta rank analysis) — the first real lead on
the input-layer mystery in four attempts.** A genuinely new tool per the
prior standing recommendation: instead of another property of the
*final* trained matrix (three already ruled out — effective rank, noise
fraction, raw-vs-processed input), measured the effective rank of the
training *update* itself (`W_trained − W_random`). Result: the input
layer's update uses 87.7% of its available rank (diffuse, near
full-rank, range 0.865–0.890 across 8 seeds); the hidden layer's uses
only 34.9% (concentrated, low-rank, range 0.278–0.437) — **zero overlap
across all 8 seeds**. Movement magnitude alone does not separate the
layers the same way — it's specifically the update's *structure*, not
its size.

**Experiment 26 (capacity-sweep cross-check, completed) — confirmed and
strengthened into the strongest mechanism correlation in the project.**
Within the hidden layer, delta-rank fraction correlates with
compression-gain magnitude at r=−0.75 (Pearson) across Experiment 6's
full 6-condition capacity-sweep grid — **stronger than Experiment 7's
original final-matrix effective-rank finding (r≈0.67)**, the first
cross-check in this project to produce a larger effect than the result
it verified. (Sign note: low delta-rank fraction = concentrated update,
so the negative correlation means concentrated updates predict *higher*
gain — same direction as the original cross-layer finding.) A visible
pattern within the hard `parity3` task alone: delta-rank fraction falls
0.60→0.39→0.19 as width rises 16→64→256 while gain rises
≈0.7×→1.0×→2.1× — giving the capacity/slack story (Experiments 5–6) a
mechanistic complement: spare capacity concentrates the training update,
which is what makes the result more compressible. The input layer still
shows no within-layer relationship (r=0.18) — its update stays diffuse
regardless of task or width. Promoted from "partial" to a full VERIFIED
RESULT in `docs/BEST_RESULTS.md`.

**Experiment 27 (Transformer transfer check, completed) — does NOT
replicate, echoing Experiment 10.** Delta-rank fraction on the Stage
C-lite Transformer: pooled r=−0.15, block 0 r=+0.16 (wrong direction),
block 1 r=−0.15 (right direction, weak) — no clean relationship, at 3
seeds (Experiment 9's original power). The classifier head's apparent
r=0.95 is a numerical artifact (degenerate 2-dimensional max rank,
delta-rank fraction constant at 0.500), excluded from interpretation.
This closely mirrors Experiment 10's non-replication of the *final*-
matrix effective-rank finding on this same architecture — both versions
of rank-based mechanism-hunting hit the same Transformer wall. Flagged as
possibly a seed-count artifact (Experiment 11's addendum showed raising
3→8 seeds materially changed a similar weak correlation) rather than
settled.

See `docs/RESEARCH_LOG.md` Experiments 18–27 and
`docs/NEXT_RESEARCH_DECISION.md` for current options (rerunning
Experiment 27 at 8 seeds to check the small-sample-artifact possibility —
the clear next step — a fourth pretrained model isolating scale from
training procedure, a real pretrained-model check of delta-rank fraction,
or folding all of this into the synthesis artifact).

### Design constraints adopted for Track B

- **CPU-only, no heavy ML dependencies for Stage A.** Only `numpy` is
  available in this environment (`torch`, `scipy`, `sklearn` are not
  installed). Stage A (synthetic matrices) needs none of them. Stage
  B/C (trained small networks / pretrained transformers) will need a
  training framework; this is called out explicitly in
  `docs/NEXT_RESEARCH_DECISION.md` as a dependency decision for the user
  rather than added silently.
- **Byte-honest accounting.** Every compression method reports its total
  storage cost as the literal sum of the bytes needed to reconstruct the
  tensor (dictionary/codebook, transform parameters, indices, residuals,
  metadata) — never an estimate. This directly implements mission rule
  "Never call a result novel merely because it compresses" and section 11's
  false-discovery-prevention checklist.
- **Baselines before claims.** Quantization, low-rank (SVD), magnitude
  pruning, scalar/vector codebooks, and a lossless (zlib) baseline are all
  implemented and run before any "Atlas structural method" claim is
  evaluated.
- **Falsification built into the test suite.** A test asserts the first
  structural method does *not* achieve strong compression on pure random
  matrices — operationalizing mission rule 11.7 ("check whether the effect
  also appears on random matrices") as a CI-enforced check rather than a
  one-off manual observation.
