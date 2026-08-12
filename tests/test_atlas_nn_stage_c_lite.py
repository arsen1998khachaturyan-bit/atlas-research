from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.stage_b.experiment import run_layer_experiment
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from atlas_nn.stage_c_lite.dataset import (
    build_vocab,
    encode,
    generate_sentiment_dataset,
    make_sentiment_arrays,
    tokenize,
)
from atlas_nn.stage_c_lite.model import (
    build_ablation_transformer_classifier,
    build_transformer_classifier,
    get_weight,
    linear_layer_names,
    set_weight,
)


def test_generate_sentiment_dataset_is_balanced_and_deduplicated():
    examples = generate_sentiment_dataset(100, seed=1)
    assert len(examples) == 100
    texts = [text for text, _label in examples]
    assert len(set(texts)) == len(texts)
    positive_frac = sum(label for _text, label in examples) / len(examples)
    assert 0.3 < positive_frac < 0.7


def test_tokenize_and_vocab_roundtrip():
    tokens = tokenize("The movie was Wonderful!")
    assert tokens == ["the", "movie", "was", "wonderful"]

    vocab = build_vocab(["the movie was great", "the food was bad"])
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert "the" in vocab and "was" in vocab


def test_encode_pads_and_truncates():
    vocab = {"<pad>": 0, "<unk>": 1, "great": 2, "movie": 3}
    ids = encode("great movie", vocab, max_len=5)
    assert ids == [2, 3, 0, 0, 0]

    ids_truncated = encode("great movie great movie great movie", vocab, max_len=3)
    assert len(ids_truncated) == 3


def test_make_sentiment_arrays_shapes_and_dtypes():
    examples = generate_sentiment_dataset(20, seed=2)
    vocab = build_vocab([text for text, _label in examples])
    x, y = make_sentiment_arrays(examples, vocab, max_len=10)
    assert x.shape == (20, 10)
    assert y.shape == (20,)
    assert x.dtype == np.int64
    assert y.dtype == np.int64


def test_build_transformer_classifier_exposes_expected_linear_layers():
    model = build_transformer_classifier(seed=1, vocab_size=50, n_layers=2)
    names = linear_layer_names(model)
    assert names == [
        "encoder.layers.0.self_attn.out_proj",
        "encoder.layers.0.linear1",
        "encoder.layers.0.linear2",
        "encoder.layers.1.self_attn.out_proj",
        "encoder.layers.1.linear1",
        "encoder.layers.1.linear2",
        "classifier",
    ]
    assert tuple(get_weight(model, "classifier").shape) == (2, 64)


def test_transformer_training_beats_random_init_on_sentiment_task():
    examples = generate_sentiment_dataset(150, seed=5)
    train_examples, eval_examples = examples[:110], examples[110:]
    vocab = build_vocab([text for text, _label in train_examples])

    x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=16)
    x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=16)

    model = build_transformer_classifier(seed=5, vocab_size=len(vocab), max_len=16)
    random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

    train_mlp(model, x_train, y_train, epochs=100, lr=3e-3)
    trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

    assert trained_acc > random_acc + 0.15
    assert trained_acc > 0.75


def test_run_layer_experiment_works_on_transformer_layer():
    examples = generate_sentiment_dataset(120, seed=6)
    train_examples, eval_examples = examples[:90], examples[90:]
    vocab = build_vocab([text for text, _label in train_examples])

    x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=16)
    x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=16)

    model = build_transformer_classifier(seed=6, vocab_size=len(vocab), max_len=16)
    train_mlp(model, x_train, y_train, epochs=60, lr=3e-3)

    layer = "classifier"
    weight_before = get_weight(model, layer).detach().numpy().copy()

    row = run_layer_experiment(
        model=model,
        layer_name=layer,
        model_state_label="trained",
        method_name="svd_rank1",
        compress_fn=lambda m: svd_low_rank(m, rank=1),
        x_eval=x_eval,
        y_eval=y_eval,
        seed=6,
        get_weight=get_weight,
        set_weight=set_weight,
        evaluate=evaluate,
        snapshot=snapshot,
        load_snapshot=load_snapshot,
    )

    assert np.allclose(get_weight(model, layer).detach().numpy(), weight_before)
    assert 0.0 <= row["original_accuracy"] <= 1.0
    assert row["layer"] == layer


@pytest.mark.parametrize("use_residual,use_layernorm", [(True, True), (False, False), (True, False), (False, True)])
def test_ablation_transformer_has_same_layer_names_as_baseline(use_residual, use_layernorm):
    model = build_ablation_transformer_classifier(
        seed=1, vocab_size=50, n_layers=2, use_residual=use_residual, use_layernorm=use_layernorm
    )
    names = linear_layer_names(model)
    assert names == [
        "encoder.layers.0.self_attn.out_proj",
        "encoder.layers.0.linear1",
        "encoder.layers.0.linear2",
        "encoder.layers.1.self_attn.out_proj",
        "encoder.layers.1.linear1",
        "encoder.layers.1.linear2",
        "classifier",
    ]


def test_ablation_transformer_trains_without_residual_or_layernorm():
    examples = generate_sentiment_dataset(150, seed=8)
    train_examples, eval_examples = examples[:110], examples[110:]
    vocab = build_vocab([text for text, _label in train_examples])

    x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=16)
    x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=16)

    model = build_ablation_transformer_classifier(
        seed=8, vocab_size=len(vocab), max_len=16, use_residual=False, use_layernorm=False
    )
    random_acc = evaluate(model, x_eval, y_eval)["accuracy"]
    train_mlp(model, x_train, y_train, epochs=100, lr=3e-3)
    trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

    assert trained_acc > random_acc + 0.15


@pytest.mark.parametrize("use_attention", [True, False])
def test_ablation_transformer_has_same_layer_names_regardless_of_attention(use_attention):
    model = build_ablation_transformer_classifier(
        seed=1, vocab_size=50, n_layers=2, use_attention=use_attention
    )
    names = linear_layer_names(model)
    assert names == [
        "encoder.layers.0.self_attn.out_proj",
        "encoder.layers.0.linear1",
        "encoder.layers.0.linear2",
        "encoder.layers.1.self_attn.out_proj",
        "encoder.layers.1.linear1",
        "encoder.layers.1.linear2",
        "classifier",
    ]
    assert tuple(get_weight(model, "encoder.layers.0.self_attn.out_proj").shape) == (64, 64)


def test_ablation_transformer_trains_without_attention():
    examples = generate_sentiment_dataset(150, seed=9)
    train_examples, eval_examples = examples[:110], examples[110:]
    vocab = build_vocab([text for text, _label in train_examples])

    x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=16)
    x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=16)

    model = build_ablation_transformer_classifier(
        seed=9, vocab_size=len(vocab), max_len=16, use_attention=False
    )
    random_acc = evaluate(model, x_eval, y_eval)["accuracy"]
    train_mlp(model, x_train, y_train, epochs=100, lr=3e-3)
    trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

    assert trained_acc > random_acc + 0.15
