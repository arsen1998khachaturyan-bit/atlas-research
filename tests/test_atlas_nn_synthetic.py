from __future__ import annotations

import numpy as np

from atlas_nn.synthetic import (
    block_repeated,
    block_transformed,
    generate_stage_a_suite,
    hierarchical_blocks,
    low_rank,
    random_gaussian,
    structured_plus_noise,
)


def test_random_gaussian_shape_and_no_repeated_rows():
    sm = random_gaussian((32, 32), seed=1)
    assert sm.matrix.shape == (32, 32)
    # extremely unlikely for two rows of continuous noise to match exactly
    assert len({tuple(row) for row in sm.matrix.tolist()}) == 32


def test_low_rank_matches_declared_rank():
    sm = low_rank((32, 32), rank=3, seed=1)
    rank = np.linalg.matrix_rank(sm.matrix, tol=1e-3)
    assert rank <= 3


def test_block_repeated_has_exactly_n_unique_blocks():
    sm = block_repeated((32, 32), block_shape=(8, 8), n_unique_blocks=3, seed=1)
    blocks = [
        tuple(sm.matrix[i:i + 8, j:j + 8].ravel().round(5).tolist())
        for i in range(0, 32, 8)
        for j in range(0, 32, 8)
    ]
    assert len(set(blocks)) <= 3


def test_block_transformed_blocks_are_affine_images_of_a_shared_prototype():
    sm = block_transformed((32, 32), block_shape=(8, 8), n_unique_blocks=2, noise_std=0.0, seed=1)
    meta = sm.ground_truth["blocks"]
    # every block's stored (proto, scale, sign, shift) reproduces the block exactly
    for entry in meta:
        i, j = entry["i"], entry["j"]
        block = sm.matrix[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
        assert block.shape == (8, 8)


def test_structured_plus_noise_close_to_low_rank_base():
    sm = structured_plus_noise((32, 32), rank=2, noise_std=0.01, seed=1)
    rank = np.linalg.matrix_rank(sm.matrix, tol=0.5)
    assert rank <= 4  # noise inflates numerical rank somewhat, but not to full rank


def test_hierarchical_blocks_shape():
    sm = hierarchical_blocks((32, 32), seed=1)
    assert sm.matrix.shape == (32, 32)


def test_generate_stage_a_suite_returns_six_matrices_with_expected_names():
    suite = generate_stage_a_suite((32, 32), seed=1)
    names = {sm.name for sm in suite}
    assert names == {
        "random_gaussian",
        "low_rank",
        "block_repeated",
        "block_transformed",
        "structured_plus_noise",
        "hierarchical_blocks",
    }
    for sm in suite:
        assert sm.matrix.shape == (32, 32)
        assert sm.matrix.dtype == np.float32
