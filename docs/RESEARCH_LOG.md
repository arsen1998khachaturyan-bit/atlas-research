# Research Log — Track B (`atlas_nn`)

All entries below are for Stage A (synthetic matrices, mission section 9).
Full per-row data: `results/atlas_nn_stage_a.json` (180 rows = 6 matrices ×
10 methods × 3 seeds `(101, 202, 303)`, shape 64×64, git commit at time of
run recorded per row).

---

## Experiment 1 — Baseline sweep on controlled synthetic matrices

**Hypothesis.** Standard baselines (quantization, low-rank/SVD, magnitude
pruning, vector-codebook VQ, lossless zlib) will each do well on the type of
structure they're suited for and poorly elsewhere, giving a real reference
frame before evaluating anything called "Atlas."

**Method.** `experiments/run_atlas_nn_stage_a.py`. Six matrix types
(`random_gaussian`, `low_rank(rank=4)`, `block_repeated(4 unique 8×8
blocks)`, `block_transformed(4 unique blocks, random per-tile scale/sign/
shift)`, `structured_plus_noise(rank=4, noise_std=0.1)`,
`hierarchical_blocks`), each 64×64 float32, seeds 101/202/303. Baselines:
8-bit/4-bit uniform quantization, rank-4/rank-8 SVD, 50%/90% magnitude
pruning, vector codebook (len-8 vectors, k=16), zlib. All byte costs are
literal component sums (`component_bytes`), not estimates.

**Result (mean over 3 seeds; ratio = original/compressed bytes, rel_l2 =
relative L2 reconstruction error):**

| matrix | best baseline (ratio @ rel_l2) | why |
|---|---|---|
| random_gaussian | zlib 1.08 @ 0.0 (lossless, no compression) OR quantize_4bit 6.4 @ 0.088 | noise is incompressible; only lossy quantization buys ratio, at a real cost |
| low_rank | svd_rank4 **8.00 @ 0.0000** | exact — matrix genuinely has rank 4 |
| block_repeated | zlib **10.15 @ 0.0000** | exact byte-level repeats are DEFLATE's best case |
| block_transformed | *none* clear — best lossy is quantize_8bit 3.99 @ 0.0095; zlib only 1.08 (no exact byte repeats since every tile has a different continuous scale/shift) |
| structured_plus_noise | svd_rank4 **8.00 @ 0.046** | still dominated by the rank-4 signal |
| hierarchical_blocks | zlib **14.15 @ 0.0000** | same reason as block_repeated, more redundancy |

**Interpretation.** Every synthetic category has an obvious, strong,
"boring" baseline that already handles it well — except `block_transformed`,
where structure exists (four shared prototypes under per-tile affine
transforms) but none of the standard baselines can see it, because it's
neither exactly low-rank, exactly repeated, nor bytewise-identical. That's
exactly the gap the mission's "shared prototype + transformation + residual"
idea (section 5) targets, and exactly the condition under which a
purpose-built structural method would be worth having.

**Next experiment.** Implement the block-dictionary + affine-transform +
residual method and test it against this same baseline table on the same
six matrices.

---

## Experiment 2 — First Atlas structural method: block dictionary + affine transform + residual

**Hypothesis.** A method that (a) clusters blocks into a small dictionary,
(b) fits a per-block scalar affine map `a·prototype + b` (trying both signs
of the prototype), and (c) stores a quantized residual, will reconstruct
`block_transformed` and other block-structured matrices far better than any
Stage-1 baseline at a comparable bit budget, while doing no better than
those baselines (and no better than pure noise floor) on `random_gaussian`.

**Method.** `atlas_nn.structural.block_dictionary_transform`, block_shape
(8,8), dict_size ∈ {8, 16}, residual_bits=4, same seeds/matrices as
Experiment 1.

**Result (mean over 3 seeds, dict_size=16, residual_bits=4 ⇒ fixed ratio
2.27 for every matrix by construction — the format doesn't adapt its size to
content):**

| matrix | rel_l2 | reads as |
|---|---|---|
| random_gaussian | 0.0721 | no structure found (expected — see falsification note below) |
| low_rank | 0.0706 | no better than random; block-dict doesn't see global low rank |
| block_repeated | **0.0000** | exact |
| block_transformed | **0.0000** | exact — this is the targeted case |
| structured_plus_noise | 0.0709 | no better than random; same reason as low_rank |
| hierarchical_blocks | **0.0000** | exact |

**Where it wins vs. the Experiment-1 baseline table:**
- `block_transformed`: Atlas gets ratio 2.27 at **0.0000** error; the best
  baseline (8-bit quantization) got ratio 3.99 but at 0.0095 error, and zlib
  (which would be "free" exactness) only reached ratio 1.08. At matched
  near-zero error, Atlas is the only method that compresses this matrix at
  all. This clears the mission's "Interesting" bar (section 15: "2× smaller
  than a strong comparable baseline at similar behavioral quality") relative
  to zlib, the only other near-zero-error method on this matrix.

**Where it loses, badly:**
- `block_repeated` / `hierarchical_blocks`: zlib reaches ratio 10–14 at
  **the same** zero error, because these matrices have byte-identical
  repeated blocks and Atlas's fixed per-block bit budget (dictionary index +
  sign + a + b + quantized residual) doesn't shrink even when the residual
  is exactly zero — it never goes below a hard floor of ~2.27–3.18×. Plain
  lossless compression is strictly better here. This is a real limitation of
  the current encoding, not of the underlying idea: the fix is to add
  entropy coding on top (e.g., only pay for a residual when it's non-zero,
  Huffman/range-code the indices) rather than a fixed-width format.
- `low_rank` / `structured_plus_noise`: SVD wins by a wide margin (ratio 8
  at ≤0.046 error vs. Atlas's ratio 2.27 at 0.07). Block-local dictionary
  matching cannot see a low-rank pattern that spans the whole matrix. Not a
  bug — a scope mismatch between method and structure type, exactly why the
  mission asks for baselines across *all* structure families before any
  claim.

**Falsification check (mission section 11.7, encoded as
`tests/test_atlas_nn_structural.py::test_structural_method_does_not_fake_compression_on_random_matrix`).**
At the identical bit budget, `random_gaussian` gives rel_l2 = 0.072, more
than 5× the error on `block_transformed` (0.0000) and far above the 0.03
threshold the test enforces. The method does not hallucinate structure on
pure noise. This test is part of the regular suite (`pytest`), not a one-off
manual check.

**Novelty discipline (mission section 14).** This method is best described
as **vector quantization / clustering (k-means codebook) generalized with a
per-cluster affine correction**, closely related to product quantization and
to "Deep Compression"-style weight-clustering codebooks; the residual
quantization is standard scalar quantization. Nothing here is claimed as
novel — it's an assembly of known techniques, evaluated honestly, per
mission section 14's requirement to identify the known family before any
novelty discussion.

**Interpretation.** The core mechanism works exactly as designed: it finds
and exploits affine block structure that literally nothing else in the
baseline set can see, and it correctly fails to find structure that isn't
there. But its current fixed-width encoding is dominated by trivial lossless
compression whenever the structure is exact repetition rather than
continuous affine variation, and it's irrelevant for global low-rank
structure. So: real, narrow, verified win — not yet a general-purpose
result.

**Next experiment.** Two candidate next steps (see
`docs/NEXT_RESEARCH_DECISION.md` for the actual choice and reasoning):
(1) fix the encoding so it doesn't lose to zlib on exact-repeat data
(entropy-code the residual/indices, or detect and special-case zero
residuals); (2) move to Stage B (trained vs. randomly-initialized small
neural networks) to test whether *training* produces this kind of
affine-block structure in real weight tensors, which is the actual research
question Stage A was built to prepare for.
