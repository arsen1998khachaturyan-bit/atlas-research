from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SyntheticMatrix:
    name: str
    matrix: np.ndarray
    ground_truth: dict = field(default_factory=dict)


def random_gaussian(shape: tuple[int, int] = (64, 64), seed: int = 0) -> SyntheticMatrix:
    """Pure noise, no structure. Any method claiming strong compression here
    at low error is suspect (mission section 11.7)."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal(shape).astype(np.float32)
    return SyntheticMatrix("random_gaussian", matrix, {"structure": "none"})


def low_rank(shape: tuple[int, int] = (64, 64), rank: int = 4, seed: int = 0) -> SyntheticMatrix:
    rng = np.random.default_rng(seed)
    m, n = shape
    a = rng.standard_normal((m, rank)).astype(np.float32)
    b = rng.standard_normal((rank, n)).astype(np.float32)
    matrix = (a @ b).astype(np.float32)
    return SyntheticMatrix("low_rank", matrix, {"structure": "low_rank", "rank": rank})


def block_repeated(
    shape: tuple[int, int] = (64, 64),
    block_shape: tuple[int, int] = (8, 8),
    n_unique_blocks: int = 4,
    seed: int = 0,
) -> SyntheticMatrix:
    """Tile of a small set of exact prototype blocks, no per-tile transform."""
    rng = np.random.default_rng(seed)
    h, w = shape
    bh, bw = block_shape
    assert h % bh == 0 and w % bw == 0, "shape must divide evenly by block_shape"
    n_bh, n_bw = h // bh, w // bw

    prototypes = rng.standard_normal((n_unique_blocks, bh, bw)).astype(np.float32)
    assign = rng.integers(0, n_unique_blocks, size=(n_bh, n_bw))

    matrix = np.zeros(shape, dtype=np.float32)
    for i in range(n_bh):
        for j in range(n_bw):
            matrix[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = prototypes[assign[i, j]]

    return SyntheticMatrix(
        "block_repeated",
        matrix,
        {
            "structure": "block_repeated",
            "block_shape": block_shape,
            "n_unique_blocks": n_unique_blocks,
            "assignment": assign.tolist(),
        },
    )


def block_transformed(
    shape: tuple[int, int] = (64, 64),
    block_shape: tuple[int, int] = (8, 8),
    n_unique_blocks: int = 4,
    noise_std: float = 0.0,
    seed: int = 0,
) -> SyntheticMatrix:
    """Small set of prototype blocks, each tile = sign * scale * prototype +
    shift (+ optional noise). This is the target structure for the block
    dictionary + transform + residual method (mission section 5)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    bh, bw = block_shape
    assert h % bh == 0 and w % bw == 0, "shape must divide evenly by block_shape"
    n_bh, n_bw = h // bh, w // bw

    prototypes = rng.standard_normal((n_unique_blocks, bh, bw)).astype(np.float32)
    matrix = np.zeros(shape, dtype=np.float32)
    blocks_meta = []

    for i in range(n_bh):
        for j in range(n_bw):
            proto_idx = int(rng.integers(0, n_unique_blocks))
            scale = float(rng.uniform(0.5, 2.0))
            sign = float(rng.choice([-1.0, 1.0]))
            shift = float(rng.uniform(-0.5, 0.5))
            block = sign * scale * prototypes[proto_idx] + shift
            if noise_std > 0:
                block = block + rng.standard_normal(block_shape).astype(np.float32) * noise_std
            matrix[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = block
            blocks_meta.append(
                {"i": i, "j": j, "proto": proto_idx, "scale": scale, "sign": sign, "shift": shift}
            )

    return SyntheticMatrix(
        "block_transformed",
        matrix,
        {
            "structure": "block_transformed",
            "block_shape": block_shape,
            "n_unique_blocks": n_unique_blocks,
            "noise_std": noise_std,
            "blocks": blocks_meta,
        },
    )


def structured_plus_noise(
    shape: tuple[int, int] = (64, 64),
    rank: int = 4,
    noise_std: float = 0.1,
    seed: int = 0,
) -> SyntheticMatrix:
    base = low_rank(shape, rank, seed)
    rng = np.random.default_rng(seed + 777)
    noisy = base.matrix + rng.standard_normal(shape).astype(np.float32) * noise_std
    return SyntheticMatrix(
        "structured_plus_noise",
        noisy.astype(np.float32),
        {"structure": "low_rank_plus_noise", "rank": rank, "noise_std": noise_std},
    )


def hierarchical_blocks(shape: tuple[int, int] = (64, 64), seed: int = 0) -> SyntheticMatrix:
    """2x2 macro-grid of quadrants, each quadrant itself a block_repeated
    matrix built from an independent small prototype set. Tests whether a
    method can exploit structure that repeats at more than one scale."""
    rng = np.random.default_rng(seed)
    h, w = shape
    half_h, half_w = h // 2, w // 2
    fine_block = (max(1, half_h // 4), max(1, half_w // 4))

    quadrants = [
        block_repeated(
            (half_h, half_w),
            fine_block,
            n_unique_blocks=2,
            seed=seed + 1000 + k,
        )
        for k in range(2)
    ]

    assign = rng.integers(0, 2, size=(2, 2))
    matrix = np.zeros(shape, dtype=np.float32)
    for i in range(2):
        for j in range(2):
            matrix[
                i * half_h:(i + 1) * half_h,
                j * half_w:(j + 1) * half_w,
            ] = quadrants[assign[i, j]].matrix

    return SyntheticMatrix(
        "hierarchical_blocks",
        matrix,
        {"structure": "hierarchical", "assignment": assign.tolist()},
    )


REGISTRY: dict[str, Callable[..., SyntheticMatrix]] = {
    "random_gaussian": random_gaussian,
    "low_rank": low_rank,
    "block_repeated": block_repeated,
    "block_transformed": block_transformed,
    "structured_plus_noise": structured_plus_noise,
    "hierarchical_blocks": hierarchical_blocks,
}


def generate_stage_a_suite(shape: tuple[int, int] = (64, 64), seed: int = 0) -> list[SyntheticMatrix]:
    """The full Stage A matrix suite (mission section 9, Stage A)."""
    return [
        random_gaussian(shape, seed=seed),
        low_rank(shape, rank=4, seed=seed),
        block_repeated(shape, block_shape=(8, 8), n_unique_blocks=4, seed=seed),
        block_transformed(shape, block_shape=(8, 8), n_unique_blocks=4, seed=seed),
        structured_plus_noise(shape, rank=4, noise_std=0.1, seed=seed),
        hierarchical_blocks(shape, seed=seed),
    ]
