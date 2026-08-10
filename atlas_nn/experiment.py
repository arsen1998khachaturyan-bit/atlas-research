from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from atlas_nn.baselines.common import CompressedResult
from atlas_nn.metrics import compression_ratio, reconstruction_metrics, timed_call


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def run_single_experiment(
    *,
    method_name: str,
    matrix_name: str,
    matrix: np.ndarray,
    seed: int,
    compress_fn: Callable[[np.ndarray], CompressedResult],
) -> dict:
    """Run one (method, matrix) pair and record every metric the mission
    requires: compression ratio, reconstruction error, encode/decode time,
    peak memory, plus full metadata for reproducibility."""
    original_bytes = int(matrix.nbytes)

    encode_timed = timed_call(lambda: compress_fn(matrix))
    result: CompressedResult = encode_timed.value

    decode_timed = timed_call(result.reconstruct)
    reconstructed = decode_timed.value

    metrics = reconstruction_metrics(matrix, reconstructed)
    total_bytes = int(result.total_bytes)

    return {
        "experiment": "atlas_nn-stage_a",
        "method": method_name,
        "matrix": matrix_name,
        "seed": seed,
        "params": result.params,
        "original_bytes": original_bytes,
        "compressed_bytes": total_bytes,
        "component_bytes": result.component_bytes,
        "compression_ratio": compression_ratio(original_bytes, total_bytes),
        "reconstruction": asdict(metrics),
        "encode_seconds": encode_timed.seconds,
        "decode_seconds": decode_timed.seconds,
        "encode_peak_memory_bytes": encode_timed.peak_memory_bytes,
        "decode_peak_memory_bytes": decode_timed.peak_memory_bytes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }


def save_json(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
