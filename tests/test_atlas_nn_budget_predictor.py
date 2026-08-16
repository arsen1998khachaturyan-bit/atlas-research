from __future__ import annotations

from pathlib import Path

import pytest

from atlas_nn.stage_c_real.budget_predictor import (
    REAL_MODEL_FILES,
    TOTAL_CONFIGS,
    evaluate_predictor,
    predict_family,
)

# This module analyzes already-computed budget-search result JSONs (no
# network, no torch, no new compression sweeps) -- skip cleanly if a
# result file is missing rather than failing, since these are large
# generated artifacts not tracked in git (see results/.gitignore).
_ALL_FILES_PRESENT = all(Path(p).exists() for p in REAL_MODEL_FILES.values())
pytestmark = pytest.mark.skipif(not _ALL_FILES_PRESENT, reason="budget search result JSONs not present")


def test_predict_family_falls_back_to_quantize_with_no_votes():
    assert predict_family([], "some_model", "early", "attn.c_proj", "from_scratch") == "quantize"


def test_total_configs_matches_documented_31_config_sweep():
    assert TOTAL_CONFIGS == 31


def test_evaluate_predictor_returns_sane_rows():
    result = evaluate_predictor()
    rows = result["per_row"]
    assert len(rows) == 30  # 5 models x 6 pretrained layer-rows each

    for row in rows:
        assert 0.0 <= row["recovery_fraction"] <= 1.0 + 1e-9
        assert 0.0 < row["compute_fraction"] <= 1.0
        assert isinstance(row["exact_match"], bool)
        assert row["predicted_compressed_bytes"] <= row["original_bytes"]
        assert row["true_compressed_bytes"] <= row["original_bytes"]


def test_leave_one_model_out_never_votes_using_the_held_out_models_own_row():
    # A row's predicted family must not depend on that exact row being in
    # the voting pool -- verified indirectly: predicting for a model with
    # only itself in its origin category (impossible to leave one out
    # from an empty remainder) must not crash and must fall back safely.
    from atlas_nn.stage_c_real.budget_predictor import build_pretrained_records

    records = build_pretrained_records("gpt2", "from_scratch", REAL_MODEL_FILES["gpt2"])
    prediction = predict_family(records, "gpt2", "early", "attn.c_proj", "from_scratch")
    assert prediction == "quantize"  # no other from_scratch model in this single-model pool
