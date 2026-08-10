# Next Research Decision

Written after Phase 1–5 of the immediate execution plan (repo audit, Stage A
baseline sweep, first Atlas structural method, multiseed validation). Covers
Track B (`atlas_nn`) only — Track A (the pre-existing symbolic active-learning
framework) is unaffected and out of scope for this document.

## 1. What we learned

- A real gap exists between "structure a standard baseline already handles"
  and "structure nothing standard handles." Five of six Stage-A synthetic
  matrix types are dominated by an obvious baseline (SVD for low-rank, zlib
  for exact repetition, plain quantization as the lossy floor for pure
  noise). Only the affine-per-block-transform case (`block_transformed`) had
  no good baseline — and that is exactly the structure the mission's section
  5 idea (shared prototype + transform + residual) targets.
- The block-dictionary + affine-transform + residual method, in its current
  fixed-width form, does find that structure: exact reconstruction at 2.27×
  on `block_transformed`, where every baseline is either much lower ratio at
  similar error (zlib) or much higher error at similar/better ratio
  (quantization/SVD/pruning/VQ).
- The same method correctly does *not* fake compression on random data
  (0.072 relative error vs. 0.0000 on structured data, at the identical bit
  budget) — this is now a standing automated test, not a one-off check.
- The method's biggest weakness right now is encoding efficiency, not the
  underlying idea: a fixed per-block bit budget means it can lose to trivial
  lossless compression (zlib) on exactly-repeated data, because it can't
  "pay less" when the residual happens to be zero.

## 2. What failed

- Block-dictionary clustering does not help on globally low-rank structure
  or low-rank-plus-noise data — SVD is simply the right tool there, and nothing
  in the block-local method can see a pattern that spans the whole matrix.
- k-means dictionary fitting at small dictionary sizes (dict_size=8 for 4
  true prototypes) was unreliable in 1/3 seeds — a local-optimum failure
  mode worth fixing (multiple restarts, better init) before trusting small
  dictionaries.
- The current structural method is strictly worse than doing nothing
  (zlib) on exact block repetition, despite reconstructing exactly — a
  format/encoding problem, documented in `docs/BEST_RESULTS.md` rather than
  hidden.

## 3. What worked

- The byte-honest accounting protocol (every component of storage cost
  summed explicitly, no estimates) made the zlib-beats-Atlas-on-repeats
  result visible immediately instead of being missed by only reporting a
  favorable ratio.
- Comparing against a real baseline panel (not just "compresses" in
  isolation) is what revealed both where the new method is genuinely useful
  (`block_transformed`) and where it currently is not (5 of 6 matrix types).
- The falsification test (structural method vs. random matrix, enforced in
  CI) gives a repeatable guard against the single most likely failure mode
  of this whole research direction: claiming structure that isn't there.

## 4. Does the evidence currently support the Atlas hypothesis?

**Partially, and narrowly.** Stage A shows that *when synthetic matrices are
constructed to contain a specific kind of structure* (shared prototypes under
per-block affine transforms), a purpose-built method can find and exploit it
in a way no standard baseline does, without hallucinating similar
performance on unstructured data. That is a real, verified, if modest,
existence proof for the general shape of the Atlas hypothesis — "there can be
structure that simple known methods miss, and a targeted method can extract
it losslessly at a non-trivial compression ratio."

**What Stage A cannot show, and the mission explicitly requires before any
stronger claim:** whether *trained* neural-network weight tensors actually
contain this (or any) kind of exploitable structure, and whether training
creates more of it than random initialization (mission section 9, Stage B).
Stage A was scaffolding for that question, not an answer to it. No claim
about real neural networks is currently supported by any evidence in this
repository.

## 5. The single most informative next experiment

**Stage B: compare block-dictionary-transform (and the baseline panel) on a
small trained MLP's weight matrices vs. the same architecture at random
initialization, same shapes, multiple training seeds.**

Why this one, specifically:
- It directly tests the mission's central open question ("does training
  create additional compressible structure?") rather than extending Stage A
  further (diminishing returns — the current fixed-width encoding weakness
  is a known, understood issue, not a research unknown).
- It's the cheapest possible test of real-network structure: a small MLP
  (e.g. 2-3 hidden layers, low hundreds of units) trained on a toy task
  (MNIST-scale or synthetic regression) can be trained on CPU in minutes,
  keeping with the mission's "don't run expensive experiments before a
  smaller one can test the same hypothesis" rule.
- It reuses 100% of the Stage A infrastructure (metrics, experiment runner,
  baseline panel, structural method) unchanged — only the matrix source
  changes from synthetic generators to extracted weight tensors.

**Blocking dependency decision needed from the user before Stage B can
start:** this environment currently has only `numpy` installed (no `torch`,
`scipy`, or `sklearn`). Stage B needs *some* way to train a small network.
Options, in order of preference given the mission's "avoid unnecessary
dependencies" rule:
1. Hand-write a tiny MLP + manual backprop in pure numpy (no new dependency,
   more implementation work, fine at the sizes Stage B needs).
2. Add `torch` (CPU) as an optional dependency (`pip install torch`) — much
   less implementation work, standard tooling, but a real new dependency to
   the project.

Recommendation: option 1 (pure numpy) for the first Stage B pass, to keep
the dependency footprint unchanged while the hypothesis is still unproven;
revisit adding torch only if/when Stage C (pretrained transformers) becomes
the actual next step, where reimplementing training is no longer reasonable.
