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
