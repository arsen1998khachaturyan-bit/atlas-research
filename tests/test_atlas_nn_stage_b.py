from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import run_layer_experiment
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp


def test_xor_dataset_is_balanced_and_not_linearly_trivial():
    x, y = make_xor_dataset(1000, n_features=32, seed=1)
    assert x.shape == (1000, 32)
    assert 0.3 < y.mean() < 0.7


def test_build_mlp_has_three_linear_layers():
    model = build_mlp(seed=1, input_dim=32)
    names = linear_layer_names(model)
    assert len(names) == 3


def test_training_improves_accuracy_over_random_init():
    x_train, y_train = make_xor_dataset(2000, n_features=32, seed=2)
    x_eval, y_eval = make_xor_dataset(300, n_features=32, seed=52)

    model = build_mlp(seed=2, input_dim=32)
    random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

    train_mlp(model, x_train, y_train, epochs=500, lr=2e-2)
    trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

    assert trained_acc > random_acc + 0.2
    assert trained_acc > 0.85


def test_snapshot_and_load_snapshot_round_trip():
    model = build_mlp(seed=3, input_dim=32)
    layer = linear_layer_names(model)[0]
    original = get_weight(model, layer).detach().numpy().copy()

    state = snapshot(model)
    set_weight(model, layer, np.zeros_like(original))
    assert not np.allclose(get_weight(model, layer).detach().numpy(), original)

    load_snapshot(model, state)
    assert np.allclose(get_weight(model, layer).detach().numpy(), original)


def test_run_layer_experiment_restores_model_weights_and_records_behavioral_metrics():
    x_train, y_train = make_xor_dataset(500, n_features=32, seed=4)
    x_eval, y_eval = make_xor_dataset(200, n_features=32, seed=54)

    model = build_mlp(seed=4, input_dim=32)
    train_mlp(model, x_train, y_train, epochs=100, lr=1e-2)
    layer = linear_layer_names(model)[0]
    weight_before = get_weight(model, layer).detach().numpy().copy()

    row = run_layer_experiment(
        model=model,
        layer_name=layer,
        model_state_label="trained",
        method_name="svd_rank4",
        compress_fn=lambda m: svd_low_rank(m, rank=4),
        x_eval=x_eval,
        y_eval=y_eval,
        seed=4,
    )

    # weights must be restored after the experiment
    assert np.allclose(get_weight(model, layer).detach().numpy(), weight_before)

    required_fields = {
        "model_state", "layer", "method", "compression_ratio", "reconstruction",
        "original_accuracy", "substituted_accuracy", "accuracy_drop",
        "relative_logit_error", "encode_seconds", "decode_seconds",
    }
    assert required_fields.issubset(row.keys())
    assert 0.0 <= row["original_accuracy"] <= 1.0
    assert 0.0 <= row["substituted_accuracy"] <= 1.0
