from __future__ import annotations

import numpy as np

from atlas_nn.baselines.common import CompressedResult

FLOAT32_BYTES = 4


def svd_low_rank(matrix: np.ndarray, rank: int) -> CompressedResult:
    """Truncated SVD: matrix ~= A @ B, A is (m, rank), B is (rank, n)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = max(1, min(rank, s.size))

    a = (u[:, :rank] * s[np.newaxis, :rank]).astype(np.float32)
    b = vt[:rank, :].astype(np.float32)

    def reconstruct():
        return (a @ b).astype(np.float32)

    return CompressedResult(
        method="svd_low_rank",
        params={"rank": rank},
        component_bytes={
            "factor_a": a.size * FLOAT32_BYTES,
            "factor_b": b.size * FLOAT32_BYTES,
        },
        reconstruct=reconstruct,
    )
