from __future__ import annotations

import numpy as np

from atlas_nn.baselines.common import CompressedResult
from atlas_nn.kmeans import kmeans

FLOAT32_BYTES = 4


def vector_codebook(
    matrix: np.ndarray,
    vector_len: int,
    k: int,
    seed: int = 0,
) -> CompressedResult:
    """Split the flattened matrix into fixed-length vectors and vector-quantize
    them against a k-entry codebook learned by k-means (classic VQ, e.g.
    as used in product quantization)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    shape = matrix.shape
    flat = matrix.ravel()
    n = flat.size

    n_vectors = int(np.ceil(n / vector_len))
    pad = n_vectors * vector_len - n
    padded = np.concatenate([flat, np.zeros(pad, dtype=np.float32)]) if pad else flat
    data = padded.reshape(n_vectors, vector_len)

    centroids, assignments = kmeans(data, k, seed)
    actual_k = centroids.shape[0]
    index_bits = max(1, int(np.ceil(np.log2(max(actual_k, 2)))))

    def reconstruct():
        out = centroids[assignments].reshape(-1)[:n]
        return out.reshape(shape)

    return CompressedResult(
        method="vector_codebook",
        params={"vector_len": vector_len, "k": k, "actual_k": actual_k, "seed": seed},
        component_bytes={
            "codebook": actual_k * vector_len * FLOAT32_BYTES,
            "indices": int(np.ceil(index_bits * n_vectors / 8)),
        },
        reconstruct=reconstruct,
    )
