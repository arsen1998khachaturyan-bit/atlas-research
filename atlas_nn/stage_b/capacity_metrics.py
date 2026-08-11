from __future__ import annotations

import numpy as np


def singular_values(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)


def stable_rank(matrix: np.ndarray) -> float:
    """||A||_F^2 / ||A||_2^2 -- a smooth, noise-robust rank proxy: 1.0 for a
    perfect rank-1 matrix, min(m, n) for an orthogonal/isometric matrix."""
    s = singular_values(matrix)
    if s.size == 0 or s[0] == 0:
        return 0.0
    return float(np.sum(s ** 2) / (s[0] ** 2))


def shannon_effective_rank(matrix: np.ndarray) -> float:
    """Roy & Vetterli (2007) effective rank: exp(Shannon entropy of the
    normalized singular value distribution). Smoothly interpolates between
    1 (all energy in one direction) and min(m, n) (energy spread evenly
    across every direction, as in a random matrix)."""
    s = singular_values(matrix)
    total = s.sum()
    if total <= 0:
        return 0.0
    p = s / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def energy_rank(matrix: np.ndarray, energy: float = 0.95) -> int:
    """Smallest k such that the top-k singular values capture `energy`
    fraction of total squared Frobenius energy (a discrete "rank at X%
    variance explained" metric, analogous to PCA component selection)."""
    s = singular_values(matrix)
    sq = s ** 2
    total = sq.sum()
    if total <= 0:
        return 0
    cumulative = np.cumsum(sq) / total
    return int(np.searchsorted(cumulative, energy) + 1)


def capacity_metrics(matrix: np.ndarray) -> dict:
    m, n = np.asarray(matrix).shape
    return {
        "max_rank": min(m, n),
        "stable_rank": stable_rank(matrix),
        "shannon_effective_rank": shannon_effective_rank(matrix),
        "energy95_rank": energy_rank(matrix, 0.95),
    }
