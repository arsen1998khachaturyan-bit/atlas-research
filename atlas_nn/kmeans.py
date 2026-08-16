from __future__ import annotations

import numpy as np


def kmeans(
    data: np.ndarray,
    k: int,
    seed: int,
    n_iters: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal Lloyd's-algorithm k-means. No external dependency (sklearn is
    not installed in this environment). Good enough for the block counts used
    in Stage A (tens to low hundreds of vectors); not optimized for scale.
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    k = min(k, n)

    init_idx = rng.choice(n, size=k, replace=False)
    centroids = data[init_idx].astype(np.float64).copy()
    assignments = np.full(n, -1, dtype=np.int64)

    for _ in range(n_iters):
        dists = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = np.argmin(dists, axis=1)

        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        for c in range(k):
            mask = assignments == c
            if mask.any():
                centroids[c] = data[mask].mean(axis=0)

    return centroids.astype(np.float32), assignments
