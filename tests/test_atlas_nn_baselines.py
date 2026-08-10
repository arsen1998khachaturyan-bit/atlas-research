from __future__ import annotations

import numpy as np

from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.lossless import zlib_baseline
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize
from atlas_nn.metrics import reconstruction_metrics
from atlas_nn.synthetic import low_rank, random_gaussian


def test_uniform_quantize_more_bits_reduces_error():
    matrix = random_gaussian((32, 32), seed=2).matrix

    err_2bit = reconstruction_metrics(
        matrix, uniform_quantize(matrix, bits=2).reconstruct()
    ).mse
    err_8bit = reconstruction_metrics(
        matrix, uniform_quantize(matrix, bits=8).reconstruct()
    ).mse

    assert err_8bit < err_2bit


def test_uniform_quantize_byte_accounting():
    matrix = np.zeros((16, 16), dtype=np.float32)
    result = uniform_quantize(matrix, bits=8, block_size=None)
    # 256 elements * 8 bits = 256 bytes of codes + 1 block * 8 bytes overhead
    assert result.component_bytes["codes"] == 256
    assert result.component_bytes["scale_and_zero_point"] == 8
    assert result.total_bytes == 264


def test_svd_low_rank_is_exact_when_rank_covers_true_rank():
    sm = low_rank((24, 24), rank=3, seed=3)
    result = svd_low_rank(sm.matrix, rank=6)
    reconstructed = result.reconstruct()
    metrics = reconstruction_metrics(sm.matrix, reconstructed)
    assert metrics.relative_l2_error < 1e-4


def test_svd_low_rank_smaller_rank_gives_smaller_representation():
    matrix = random_gaussian((32, 32), seed=4).matrix
    small = svd_low_rank(matrix, rank=2)
    large = svd_low_rank(matrix, rank=16)
    assert small.total_bytes < large.total_bytes


def test_magnitude_prune_keeps_declared_fraction_of_entries():
    matrix = random_gaussian((20, 20), seed=5).matrix
    result = magnitude_prune(matrix, sparsity=0.9)
    assert result.params["nnz"] == 40  # 10% of 400
    reconstructed = result.reconstruct()
    assert np.count_nonzero(reconstructed) <= 40


def test_magnitude_prune_keeps_the_largest_magnitude_entries():
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 100.0
    matrix[3, 3] = -50.0
    result = magnitude_prune(matrix, sparsity=0.875)  # keep top 2 of 16
    reconstructed = result.reconstruct()
    assert reconstructed[0, 0] == 100.0
    assert reconstructed[3, 3] == -50.0


def test_vector_codebook_reconstructs_correct_shape_and_reports_bytes():
    matrix = random_gaussian((16, 32), seed=6).matrix
    result = vector_codebook(matrix, vector_len=8, k=4, seed=6)
    reconstructed = result.reconstruct()
    assert reconstructed.shape == matrix.shape
    assert result.total_bytes > 0


def test_zlib_baseline_is_exactly_lossless():
    matrix = random_gaussian((16, 16), seed=7).matrix
    result = zlib_baseline(matrix)
    reconstructed = result.reconstruct()
    assert np.array_equal(matrix, reconstructed)
