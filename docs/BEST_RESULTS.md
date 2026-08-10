# Best Results — Track B (`atlas_nn`)

Only reproducible results go here. Every entry cites the exact command and
seeds used to reproduce it. Labels are used strictly:

- **OBSERVATION** — something measured once or descriptively, not yet
  stress-tested across seeds/ablations.
- **HYPOTHESIS** — an explanation or prediction, not yet verified.
- **VERIFIED RESULT** — reproduced across multiple seeds, compared against
  baselines under the same protocol, and covered by an automated test.

---

## VERIFIED RESULT: block-dictionary + affine-transform beats all Stage-1 baselines on affine-block-structured synthetic matrices

**Claim.** On a 64×64 matrix built from 4 shared 8×8 prototype blocks, each
tile transformed independently by `sign · scale · prototype + shift`
(`atlas_nn.synthetic.block_transformed`), `atlas_nn.structural.
block_dictionary_transform` (dict_size=16, residual_bits=4, block 8×8)
reconstructs the matrix **exactly** (relative L2 error 0.0000) at a
**2.27× compression ratio**. Every baseline tested (8-bit/4-bit uniform
quantization, rank-4/rank-8 SVD, 50%/90% magnitude pruning, vector-codebook
VQ, zlib) either compresses less (zlib: 1.08×) or compresses more but with
much higher error (quantization/SVD/pruning/VQ: 0.01–0.85 relative L2).

**How verified.**
- Reproduced across 3 seeds (101, 202, 303) — see `results/atlas_nn_stage_a.json`.
- Compared against 9 baseline configurations under the identical protocol
  (`experiments/run_atlas_nn_stage_a.py`).
- Falsification check: the same method/bit-budget applied to
  `random_gaussian` (no structure) gives relative L2 error 0.072 — more than
  5× worse than on the structured matrix, and the "no free lunch on random
  data" property is enforced by an automated test
  (`tests/test_atlas_nn_structural.py::test_structural_method_does_not_fake_compression_on_random_matrix`).
- Reproduce with: `python -m experiments.run_atlas_nn_stage_a` (writes
  `results/atlas_nn_stage_a.json`) or `pytest tests/test_atlas_nn_structural.py`.

**Scope of the claim (what this does NOT show).** This is Stage A only —
controlled synthetic matrices, not trained neural-network weights. It does
not show that real trained weight tensors contain this kind of structure
(that is exactly what Stage B is for — see
`docs/NEXT_RESEARCH_DECISION.md`). It also does not beat baselines on 5 of
the 6 synthetic matrix types tested (see `docs/RESEARCH_LOG.md`, Experiment
2) — low-rank and structured+noise matrices are dominated by SVD, and
exact-repeat matrices (block_repeated, hierarchical_blocks) are dominated by
plain lossless zlib. This is a narrow, verified win on one structure family,
not a general result.

---

## VERIFIED RESULT: the block-dictionary method is currently dominated by plain lossless compression on exact block repetition

**Claim.** On `block_repeated` and `hierarchical_blocks` matrices (exact,
byte-identical repeated blocks, no continuous per-tile transform), zlib
alone reaches 10.15× and 14.15× compression respectively at zero error,
while `block_dictionary_transform` is capped at 2.27–3.18× (also at zero
error) by its fixed-width encoding, which spends a constant number of bits
per block regardless of whether the residual is exactly zero.

**How verified.** Same protocol/seeds as above; see `docs/RESEARCH_LOG.md`
Experiment 2 for the full comparison table.

**Why this matters.** It is a genuine current weakness of the
implementation (fixed-width, no entropy coding), not evidence against the
underlying block-dictionary idea — the method still gets the reconstruction
exactly right, it just doesn't shrink its own encoding when it could. Listed
here, not hidden, per the mission's instruction to report negative results.

---

## OBSERVATION: k-means dictionary fitting is not perfectly reliable at small dictionary sizes

On `block_repeated` with `dict_size=8` (only 4 true unique blocks exist),
one of three seeds gave relative L2 error 0.0448 instead of 0.0000 — the
k-means step must have converged to a local optimum that merged two
distinct prototypes. `dict_size=16` gave exact reconstruction on all 3
seeds tested. Not yet ablated across more seeds/restarts to characterize
how often this happens; flagged as an observation, not a verified failure
rate.

---

## Explicitly not yet claimed

- Nothing about trained neural-network weights (Stage B/C/D of the mission
  ladder) — no such experiment has been run yet.
- Nothing about direct inference without materializing full weight tensors
  (mission section 8) — not attempted yet.
- Nothing about cross-layer/cross-model shared dictionaries (mission section
  7) — not attempted yet.
- No patentability or novelty claim — see `docs/RESEARCH_LOG.md` Experiment
  2's "Novelty discipline" note; this method is a known family (VQ/codebook
  clustering + affine correction), not a new technique.
