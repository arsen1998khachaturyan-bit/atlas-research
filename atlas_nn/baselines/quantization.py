from __future__ import annotations

import numpy as np

from atlas_nn.baselines.common import CompressedResult

FLOAT32_BYTES = 4


def uniform_quantize(
    matrix: np.ndarray,
    bits: int = 8,
    block_size: int | None = None,
) -> CompressedResult:
    """Uniform affine (min/max) quantization. With `block_size=None` this is
    per-tensor quantization (one scale/zero-point for the whole matrix);
    otherwise the flattened matrix is split into chunks of `block_size`
    elements, each with its own scale/zero-point.

    `component_bytes["codes"]` reports the storage cost as if the codes were
    bit-packed to `bits` bits/element (ceil(bits * n / 8)); reconstruct()
    itself works from the unpacked in-memory uint32 array for simplicity,
    which does not affect the reported byte cost or the reconstruction
    values.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    shape = matrix.shape
    flat = matrix.ravel()
    n = flat.size
    qmax = (1 << bits) - 1

    size = block_size if block_size else n
    n_blocks = int(np.ceil(n / size))

    codes = np.empty(n, dtype=np.uint32)
    scales = np.empty(n_blocks, dtype=np.float32)
    mins = np.empty(n_blocks, dtype=np.float32)

    for bi in range(n_blocks):
        lo_idx, hi_idx = bi * size, min((bi + 1) * size, n)
        block = flat[lo_idx:hi_idx]
        lo, hi = float(block.min()), float(block.max())
        if hi <= lo:
            hi = lo + 1e-8
        scale = (hi - lo) / qmax
        scales[bi] = scale
        mins[bi] = lo
        q = np.clip(np.round((block - lo) / scale), 0, qmax).astype(np.uint32)
        codes[lo_idx:hi_idx] = q

    def reconstruct():
        out = np.empty(n, dtype=np.float32)
        for bi in range(n_blocks):
            lo_idx, hi_idx = bi * size, min((bi + 1) * size, n)
            out[lo_idx:hi_idx] = codes[lo_idx:hi_idx] * scales[bi] + mins[bi]
        return out.reshape(shape)

    packed_code_bytes = int(np.ceil(bits * n / 8))
    overhead_bytes = n_blocks * 2 * FLOAT32_BYTES

    return CompressedResult(
        method=f"quantize_{bits}bit_" + (f"block{block_size}" if block_size else "pertensor"),
        params={"bits": bits, "block_size": block_size, "n_blocks": n_blocks},
        component_bytes={"codes": packed_code_bytes, "scale_and_zero_point": overhead_bytes},
        reconstruct=reconstruct,
    )
