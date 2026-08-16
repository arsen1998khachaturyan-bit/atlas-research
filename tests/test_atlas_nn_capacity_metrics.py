from __future__ import annotations

import numpy as np
import pytest

from atlas_nn.stage_b.capacity_metrics import (
    capacity_metrics,
    energy_rank,
    shannon_effective_rank,
    stable_rank,
)


def test_rank_one_matrix_has_effective_rank_near_one():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((20, 1))
    v = rng.standard_normal((1, 15))
    matrix = u @ v

    assert stable_rank(matrix) == pytest.approx(1.0, rel=0.15)
    assert shannon_effective_rank(matrix) == pytest.approx(1.0, rel=0.15)
    assert energy_rank(matrix, 0.95) == 1


def test_random_matrix_has_effective_rank_close_to_max_rank():
    rng = np.random.default_rng(1)
    matrix = rng.standard_normal((40, 40))

    metrics = capacity_metrics(matrix)
    assert metrics["max_rank"] == 40
    # a random square Gaussian matrix's singular values decay gradually
    # (not a sharp cutoff like a low-rank matrix) -- Shannon effective rank
    # (robust to the top singular value) should be a large fraction of
    # max_rank; stable_rank is more sensitive to the top singular value and
    # so runs noticeably lower for a square random matrix, but still well
    # above a low-rank matrix's stable_rank of ~1.
    assert metrics["shannon_effective_rank"] > 0.7 * metrics["max_rank"]
    assert metrics["stable_rank"] > 5.0


def test_effective_rank_increases_with_matrix_rank():
    rng = np.random.default_rng(2)
    shape = (30, 30)

    def make_rank_k(k):
        a = rng.standard_normal((shape[0], k))
        b = rng.standard_normal((k, shape[1]))
        return a @ b

    low = shannon_effective_rank(make_rank_k(2))
    mid = shannon_effective_rank(make_rank_k(10))
    high = shannon_effective_rank(make_rank_k(30))

    assert low < mid < high


def test_zero_matrix_has_zero_effective_rank():
    matrix = np.zeros((10, 10))
    assert stable_rank(matrix) == 0.0
    assert shannon_effective_rank(matrix) == 0.0
    assert energy_rank(matrix) == 0
