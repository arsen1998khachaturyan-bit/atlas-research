from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import numpy as np

from atlas_nn.stage_c_real.dataset import build_eval_batch
from atlas_nn.stage_c_real.model import (
    evaluate,
    get_weight,
    linear_layer_names,
    load_pretrained,
    load_random_init,
    load_snapshot,
    load_tokenizer,
    set_weight,
    snapshot,
    tested_layer_names as get_tested_layer_names,
)

# These tests need the real distilgpt2 checkpoint from huggingface.co --
# skip cleanly (rather than fail) if it can't be reached, since Stage C
# (real) is an opt-in extra whose whole premise is network availability
# (see docs/RESEARCH_LOG.md Experiment 8's network-policy note).
try:
    _tokenizer = load_tokenizer()
    _pretrained_model = load_pretrained()
    _NETWORK_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on environment
    _NETWORK_AVAILABLE = False
    _SKIP_REASON = f"distilgpt2 unreachable: {exc}"

pytestmark = pytest.mark.skipif(not _NETWORK_AVAILABLE, reason="distilgpt2 unreachable")


def test_tested_layer_names_are_a_fixed_depth_balanced_subset():
    names = get_tested_layer_names()
    assert len(names) == 12
    assert names == linear_layer_names(None)
    for block in (0, 3, 5):
        for suffix in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            assert f"transformer.h.{block}.{suffix}" in names


def test_get_weight_shapes_match_gpt2_conv1d_layout():
    expected_shapes = {
        "attn.c_attn": (768, 2304),
        "attn.c_proj": (768, 768),
        "mlp.c_fc": (768, 3072),
        "mlp.c_proj": (3072, 768),
    }
    for name in get_tested_layer_names():
        suffix = name.split(".", 3)[-1]
        weight = get_weight(_pretrained_model, name)
        assert tuple(weight.shape) == expected_shapes[suffix]


def test_set_weight_then_get_weight_roundtrips():
    name = "transformer.h.0.attn.c_proj"
    original = get_weight(_pretrained_model, name).detach().numpy().copy()
    replacement = np.random.RandomState(0).randn(*original.shape).astype(np.float32)
    try:
        set_weight(_pretrained_model, name, replacement)
        updated = get_weight(_pretrained_model, name).detach().numpy()
        np.testing.assert_allclose(updated, replacement, atol=1e-6)
    finally:
        set_weight(_pretrained_model, name, original)


def test_snapshot_and_load_snapshot_restore_tested_layers_exactly():
    name = "transformer.h.5.mlp.c_proj"
    original = get_weight(_pretrained_model, name).detach().numpy().copy()
    state = snapshot(_pretrained_model)

    replacement = np.zeros_like(original)
    set_weight(_pretrained_model, name, replacement)
    assert np.allclose(get_weight(_pretrained_model, name).detach().numpy(), 0.0)

    load_snapshot(_pretrained_model, state)
    restored = get_weight(_pretrained_model, name).detach().numpy()
    np.testing.assert_allclose(restored, original, atol=1e-6)


def test_random_init_is_reproducible_per_seed_and_differs_across_seeds():
    model_a = load_random_init(seed=101)
    model_b = load_random_init(seed=101)
    model_c = load_random_init(seed=202)

    name = "transformer.h.0.mlp.c_fc"
    weight_a = get_weight(model_a, name).detach().numpy()
    weight_b = get_weight(model_b, name).detach().numpy()
    weight_c = get_weight(model_c, name).detach().numpy()

    np.testing.assert_allclose(weight_a, weight_b, atol=1e-6)
    assert not np.allclose(weight_a, weight_c, atol=1e-3)


def test_random_init_differs_from_pretrained():
    random_model = load_random_init(seed=11)
    name = "transformer.h.0.mlp.c_fc"
    pretrained_weight = get_weight(_pretrained_model, name).detach().numpy()
    random_weight = get_weight(random_model, name).detach().numpy()
    assert not np.allclose(pretrained_weight, random_weight, atol=1e-3)


def test_evaluate_returns_well_shaped_logits_and_bounded_accuracy():
    x_eval = build_eval_batch(_tokenizer)
    result = evaluate(_pretrained_model, x_eval, None)
    assert result["logits"].shape == (x_eval.shape[0], x_eval.shape[1], _pretrained_model.config.vocab_size)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_pretrained_next_token_accuracy_beats_random_init():
    """A trained language model should be a far better next-token predictor
    on ordinary English sentences than an untrained one of the same
    architecture -- the language-model analogue of the accuracy checks used
    throughout Stage B/C-lite to confirm training actually happened before
    trusting any compression comparison. Here there is no training step to
    verify (the pretrained checkpoint's training already happened upstream
    and is not reproduced in this repo), so this is instead a sanity check
    that the loaded pretrained weights are the real, trained checkpoint and
    not some accidentally-untrained substitute."""
    x_eval = build_eval_batch(_tokenizer)
    pretrained_acc = evaluate(_pretrained_model, x_eval, None)["accuracy"]
    random_model = load_random_init(seed=11)
    random_acc = evaluate(random_model, x_eval, None)["accuracy"]
    assert pretrained_acc > random_acc + 0.1
