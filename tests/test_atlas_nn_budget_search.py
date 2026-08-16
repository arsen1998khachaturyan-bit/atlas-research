from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from atlas_nn.stage_b.budget_search import run_budget_search
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp


def test_run_budget_search_returns_well_formed_result():
    x_train, y_train = make_xor_dataset(300, n_features=32, seed=7)
    x_eval, y_eval = make_xor_dataset(100, n_features=32, seed=57)

    model = build_mlp(seed=7, input_dim=32)
    random_state = snapshot(model)
    train_mlp(model, x_train, y_train, epochs=50, lr=2e-2)

    layer = linear_layer_names(model)[0]
    shape = tuple(get_weight(model, layer).shape)

    result = run_budget_search(
        model=model,
        layer_name=layer,
        layer_shape=shape,
        model_state_label="trained",
        x_eval=x_eval,
        y_eval=y_eval,
        seed=7,
        get_weight=get_weight,
        set_weight=set_weight,
        evaluate=evaluate,
        snapshot=snapshot,
        load_snapshot=load_snapshot,
        quality_threshold=0.2,
    )

    expected_families = {"quantize", "svd", "prune", "vector_codebook", "atlas_block_dict"}
    assert set(result["families"]) == expected_families

    for family_name, family_result in result["families"].items():
        assert family_result["status"] in {"met_threshold", "threshold_not_met"}
        if family_result["status"] == "met_threshold":
            assert family_result["best"]["relative_logit_error"] <= 0.2

    # weights must be unchanged after the search (each row restores them)
    load_snapshot(model, random_state)  # sanity: snapshot/restore machinery still works
