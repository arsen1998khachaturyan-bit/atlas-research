from __future__ import annotations

import numpy as np

from atlas_nn.metrics import compression_ratio, reconstruction_metrics, timed_call


def test_reconstruction_metrics_identical_is_zero_error():
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((16, 16)).astype(np.float32)

    metrics = reconstruction_metrics(matrix, matrix.copy())

    assert metrics.mse == 0.0
    assert metrics.rmse == 0.0
    assert metrics.relative_l2_error == 0.0
    assert metrics.max_abs_error == 0.0
    assert metrics.cosine_similarity == 1.0


def test_reconstruction_metrics_shape_mismatch_raises():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 5), dtype=np.float32)

    try:
        reconstruction_metrics(a, b)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def test_compression_ratio_basic():
    assert compression_ratio(1000, 100) == 10.0
    assert compression_ratio(1000, 1000) == 1.0


def test_timed_call_reports_positive_elapsed_time():
    result = timed_call(lambda: sum(range(100_000)))
    assert result.value == sum(range(100_000))
    assert result.seconds >= 0.0
    assert result.peak_memory_bytes >= 0
