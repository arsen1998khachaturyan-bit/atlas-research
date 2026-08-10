from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class ReconstructionMetrics:
    mse: float
    rmse: float
    relative_l2_error: float
    max_abs_error: float
    cosine_similarity: float


def reconstruction_metrics(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> ReconstructionMetrics:
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)

    if original.shape != reconstructed.shape:
        raise ValueError(
            f"shape mismatch: original={original.shape} "
            f"reconstructed={reconstructed.shape}"
        )

    diff = reconstructed - original
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))

    orig_norm = float(np.linalg.norm(original))
    diff_norm = float(np.linalg.norm(diff))
    relative_l2_error = diff_norm / orig_norm if orig_norm > 0 else diff_norm

    max_abs_error = float(np.max(np.abs(diff))) if diff.size else 0.0

    flat_o = original.ravel()
    flat_r = reconstructed.ravel()
    denom = np.linalg.norm(flat_o) * np.linalg.norm(flat_r)
    cosine_similarity = float(np.dot(flat_o, flat_r) / denom) if denom > 0 else 0.0

    return ReconstructionMetrics(
        mse=mse,
        rmse=rmse,
        relative_l2_error=relative_l2_error,
        max_abs_error=max_abs_error,
        cosine_similarity=cosine_similarity,
    )


@dataclass(frozen=True)
class TimedResult:
    value: object
    seconds: float
    peak_memory_bytes: int


def timed_call(fn: Callable[[], T]) -> "TimedResult":
    tracemalloc.start()
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return TimedResult(value=value, seconds=elapsed, peak_memory_bytes=peak)


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes <= 0:
        raise ValueError("compressed_bytes must be positive")
    return original_bytes / compressed_bytes
