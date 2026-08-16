from __future__ import annotations

from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.experiment import run_single_experiment
from atlas_nn.synthetic import low_rank


def test_run_single_experiment_records_required_fields():
    sm = low_rank((16, 16), rank=2, seed=9)

    row = run_single_experiment(
        method_name="svd_rank2",
        matrix_name=sm.name,
        matrix=sm.matrix,
        seed=9,
        compress_fn=lambda m: svd_low_rank(m, rank=2),
    )

    required_fields = {
        "method", "matrix", "seed", "params", "original_bytes",
        "compressed_bytes", "component_bytes", "compression_ratio",
        "reconstruction", "encode_seconds", "decode_seconds",
        "encode_peak_memory_bytes", "decode_peak_memory_bytes",
        "timestamp", "git_commit",
    }
    assert required_fields.issubset(row.keys())
    assert row["original_bytes"] == sm.matrix.nbytes
    assert row["compression_ratio"] == row["original_bytes"] / row["compressed_bytes"]
    assert row["reconstruction"]["relative_l2_error"] < 1e-3
