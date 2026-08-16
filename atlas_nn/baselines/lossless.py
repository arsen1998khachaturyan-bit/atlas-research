from __future__ import annotations

import zlib

import numpy as np

from atlas_nn.baselines.common import CompressedResult


def zlib_baseline(matrix: np.ndarray, level: int = 9) -> CompressedResult:
    """General-purpose lossless compression on the raw float32 bytes, with no
    assumption about tensor structure. This is the "did we beat something
    trivial" floor -- any Atlas method that doesn't clear it on structured
    data isn't finding anything zlib can't already find."""
    matrix = np.asarray(matrix, dtype=np.float32)
    shape = matrix.shape
    raw = matrix.tobytes()
    compressed = zlib.compress(raw, level)

    def reconstruct():
        raw_back = zlib.decompress(compressed)
        return np.frombuffer(raw_back, dtype=np.float32).reshape(shape).copy()

    return CompressedResult(
        method="zlib_lossless",
        params={"level": level},
        component_bytes={"compressed_bytes": len(compressed)},
        reconstruct=reconstruct,
    )
