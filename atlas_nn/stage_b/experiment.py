from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from atlas_nn.baselines.common import CompressedResult
from atlas_nn.metrics import compression_ratio, reconstruction_metrics, timed_call
from atlas_nn.stage_b.model import get_weight, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot


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


def run_layer_experiment(
    *,
    model,
    layer_name: str,
    model_state_label: str,
    method_name: str,
    compress_fn: Callable[[np.ndarray], CompressedResult],
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> dict:
    """Compress one Linear layer's weight matrix, reconstruct it, put the
    reconstruction back into a *copy* of the model's current weights, and
    measure both tensor-level and behavioral degradation vs. the original
    (uncompressed) model in its current state. The model's own weights are
    restored to their original value before returning, regardless of
    outcome, so this is safe to call repeatedly on the same model instance.
    """
    original_weight = get_weight(model, layer_name).detach().numpy().copy()
    original_state = snapshot(model)

    baseline_eval = evaluate(model, x_eval, y_eval)

    original_bytes = int(original_weight.nbytes)
    encode_timed = timed_call(lambda: compress_fn(original_weight))
    result: CompressedResult = encode_timed.value

    decode_timed = timed_call(result.reconstruct)
    reconstructed_weight = decode_timed.value

    tensor_metrics = reconstruction_metrics(original_weight, reconstructed_weight)

    try:
        set_weight(model, layer_name, reconstructed_weight)
        substituted_eval = evaluate(model, x_eval, y_eval)
    finally:
        load_snapshot(model, original_state)

    logits_orig = baseline_eval["logits"]
    logits_sub = substituted_eval["logits"]
    logit_diff_norm = float(np.linalg.norm(logits_sub - logits_orig))
    logit_orig_norm = float(np.linalg.norm(logits_orig))
    relative_logit_error = logit_diff_norm / logit_orig_norm if logit_orig_norm > 0 else logit_diff_norm

    total_bytes = int(result.total_bytes)

    return {
        "experiment": "atlas_nn-stage_b",
        "model_state": model_state_label,
        "layer": layer_name,
        "layer_shape": list(original_weight.shape),
        "method": method_name,
        "seed": seed,
        "params": result.params,
        "original_bytes": original_bytes,
        "compressed_bytes": total_bytes,
        "component_bytes": result.component_bytes,
        "compression_ratio": compression_ratio(original_bytes, total_bytes),
        "reconstruction": asdict(tensor_metrics),
        "original_accuracy": baseline_eval["accuracy"],
        "substituted_accuracy": substituted_eval["accuracy"],
        "accuracy_drop": baseline_eval["accuracy"] - substituted_eval["accuracy"],
        "relative_logit_error": relative_logit_error,
        "encode_seconds": encode_timed.seconds,
        "decode_seconds": decode_timed.seconds,
        "encode_peak_memory_bytes": encode_timed.peak_memory_bytes,
        "decode_peak_memory_bytes": decode_timed.peak_memory_bytes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
    }


def save_json(payload: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
