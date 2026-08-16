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


def make_parity_dataset(
    n_samples: int,
    n_features: int = 32,
    k: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Harder variant of `make_xor_dataset`: label is the parity (k-way XOR)
    of the signs of the first `k` coordinates, rest of `n_features` dims are
    noise. Requires the network to combine more than 2 informative inputs
    nonlinearly, so it needs more representational work than plain 2-XOR --
    used as a second, harder task to check whether the Experiment 4
    post-training-compressibility pattern is specific to the easy 2-XOR task
    or holds more generally (see docs/NEXT_RESEARCH_DECISION.md).
    """
    if k < 2:
        raise ValueError("k must be >= 2 for a non-trivial parity task")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    bits = x[:, :k] > 0
    y = bits[:, 0]
    for i in range(1, k):
        y = y ^ bits[:, i]
    return x, y.astype(np.int64)
