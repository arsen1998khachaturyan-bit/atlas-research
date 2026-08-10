from __future__ import annotations

import numpy as np

from atlas_nn.baselines.common import CompressedResult

FLOAT32_BYTES = 4
INT32_BYTES = 4


def magnitude_prune(matrix: np.ndarray, sparsity: float) -> CompressedResult:
    """Keep the top (1 - sparsity) fraction of entries by absolute magnitude,
    store the rest implicitly as zero, encoded as sparse COO (values +
    flat indices)."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")

    matrix = np.asarray(matrix, dtype=np.float32)
    shape = matrix.shape
    flat = matrix.ravel()
    n = flat.size
    n_keep = max(1, int(round(n * (1.0 - sparsity))))

    if n_keep >= n:
        keep_idx = np.arange(n)
    else:
        keep_idx = np.argpartition(np.abs(flat), n - n_keep)[-n_keep:]

    values = flat[keep_idx].astype(np.float32)
    indices = keep_idx.astype(np.int32)

    def reconstruct():
        out = np.zeros(n, dtype=np.float32)
        out[indices] = values
        return out.reshape(shape)

    return CompressedResult(
        method="magnitude_pruning",
        params={"sparsity": sparsity, "nnz": int(len(indices))},
        component_bytes={
            "values": len(values) * FLOAT32_BYTES,
            "indices": len(indices) * INT32_BYTES,
        },
        reconstruct=reconstruct,
    )
