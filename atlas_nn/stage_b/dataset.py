from __future__ import annotations

import numpy as np


def make_xor_dataset(
    n_samples: int,
    n_features: int = 32,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """XOR-of-two-coordinates label embedded in `n_features` dims (the other
    `n_features - 2` dims are pure noise). Not linearly separable, so a
    trained MLP should clear ~100% while a linear model / random network
    should sit near chance (50%) -- a clean way to check that *training*,
    not just architecture, is what creates any weight structure we find.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int64)
    return x, y
