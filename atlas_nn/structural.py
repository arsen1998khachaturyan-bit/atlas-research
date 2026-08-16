from __future__ import annotations

import numpy as np

from atlas_nn.baselines.common import CompressedResult
from atlas_nn.kmeans import kmeans

FLOAT32_BYTES = 4


def _extract_blocks(
    matrix: np.ndarray,
    block_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = matrix.shape
    bh, bw = block_shape
    pad_h, pad_w = (-h) % bh, (-w) % bw
    padded = np.pad(matrix, ((0, pad_h), (0, pad_w)))
    ph, pw = padded.shape
    n_bh, n_bw = ph // bh, pw // bw
    blocks = (
        padded.reshape(n_bh, bh, n_bw, bw)
        .transpose(0, 2, 1, 3)
        .reshape(n_bh * n_bw, bh * bw)
    )
    return blocks, (n_bh, n_bw)


def _assemble_blocks(
    blocks: np.ndarray,
    grid: tuple[int, int],
    block_shape: tuple[int, int],
    orig_shape: tuple[int, int],
) -> np.ndarray:
    n_bh, n_bw = grid
    bh, bw = block_shape
    mat = (
        blocks.reshape(n_bh, n_bw, bh, bw)
        .transpose(0, 2, 1, 3)
        .reshape(n_bh * bh, n_bw * bw)
    )
    return mat[: orig_shape[0], : orig_shape[1]]


def block_dictionary_transform(
    matrix: np.ndarray,
    block_shape: tuple[int, int] = (8, 8),
    dict_size: int = 8,
    allow_sign_flip: bool = True,
    residual_bits: int | None = 4,
    seed: int = 0,
) -> CompressedResult:
    """Mission section 5: represent each block as

        shared_prototype (from a k-means dictionary) -> affine transform -> + residual

    For each block we try every dictionary prototype (and, if
    `allow_sign_flip`, its negation), fit the least-squares-optimal scalar
    affine map `a * prototype + b`, and keep the (prototype, a, b) with
    lowest residual energy. The residual is then optionally quantized to
    `residual_bits` bits/element with a per-block scale/zero-point
    (`residual_bits=None` stores the residual as raw float32, which is only
    useful as a debugging upper bound -- it defeats the point of compression
    and should not be used for real comparisons).

    Every one of these pieces (dictionary, prototype index, sign bit, a, b,
    residual, and residual's own scale/zero-point when quantized) is counted
    in `component_bytes`, per mission section 6 ("no compression if hidden
    storage costs are ignored").
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    orig_shape = matrix.shape
    blocks, grid = _extract_blocks(matrix, block_shape)
    n_blocks, block_len = blocks.shape

    centroids, _ = kmeans(blocks, dict_size, seed)
    actual_k = centroids.shape[0]
    signs = (-1.0, 1.0) if allow_sign_flip else (1.0,)

    proto_idx = np.zeros(n_blocks, dtype=np.int64)
    coef_a = np.zeros(n_blocks, dtype=np.float32)
    coef_b = np.zeros(n_blocks, dtype=np.float32)
    coef_sign = np.ones(n_blocks, dtype=np.float32)
    residual = np.zeros_like(blocks)

    for i in range(n_blocks):
        block = blocks[i]
        block_mean = block.mean()
        block_centered = block - block_mean
        best_cost = None

        for p in range(actual_k):
            proto = centroids[p]
            for sign in signs:
                signed = sign * proto
                signed_mean = signed.mean()
                signed_centered = signed - signed_mean
                denom = float(signed_centered @ signed_centered)
                a = float(signed_centered @ block_centered) / denom if denom > 1e-12 else 0.0
                b = float(block_mean - a * signed_mean)
                pred = a * signed + b
                cost = float(((block - pred) ** 2).sum())

                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    proto_idx[i] = p
                    coef_a[i] = a
                    coef_b[i] = b
                    coef_sign[i] = sign
                    residual[i] = block - pred

    index_bits = max(1, int(np.ceil(np.log2(max(actual_k, 2)))))
    sign_bits = 1 if allow_sign_flip else 0

    if residual_bits is None:
        res_scale = res_min = None
        quantized_residual = residual
        residual_bytes = residual.size * FLOAT32_BYTES
    else:
        qmax = (1 << residual_bits) - 1
        res_min = residual.min(axis=1).astype(np.float32)
        res_max = residual.max(axis=1).astype(np.float32)
        res_scale = np.where(res_max > res_min, (res_max - res_min) / qmax, 1.0).astype(np.float32)
        quantized_residual = np.clip(
            np.round((residual - res_min[:, None]) / res_scale[:, None]), 0, qmax
        )
        residual_bytes = int(np.ceil(residual_bits * residual.size / 8)) + 2 * n_blocks * FLOAT32_BYTES

    def reconstruct():
        recon_blocks = np.empty_like(blocks)
        for i in range(n_blocks):
            signed_proto = coef_sign[i] * centroids[proto_idx[i]]
            pred = coef_a[i] * signed_proto + coef_b[i]
            if residual_bits is None:
                res = quantized_residual[i]
            else:
                res = quantized_residual[i] * res_scale[i] + res_min[i]
            recon_blocks[i] = pred + res
        return _assemble_blocks(recon_blocks, grid, block_shape, orig_shape)

    component_bytes = {
        "dictionary": centroids.size * FLOAT32_BYTES,
        "proto_indices": int(np.ceil(index_bits * n_blocks / 8)),
        "sign_bits": int(np.ceil(sign_bits * n_blocks / 8)),
        "scale_a": n_blocks * FLOAT32_BYTES,
        "shift_b": n_blocks * FLOAT32_BYTES,
        "residual": residual_bytes,
    }

    return CompressedResult(
        method="block_dictionary_transform",
        params={
            "block_shape": block_shape,
            "dict_size": dict_size,
            "actual_dict_size": actual_k,
            "allow_sign_flip": allow_sign_flip,
            "residual_bits": residual_bits,
            "seed": seed,
        },
        component_bytes=component_bytes,
        reconstruct=reconstruct,
    )
