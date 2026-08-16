"""Stage C-lite smoke test (see atlas_nn/stage_c_lite/__init__.py for why
this exists instead of a literal pretrained-transformer Stage C).

Mirrors the original Stage B smoke experiment
(experiments/run_atlas_nn_stage_b.py) but on a small Transformer classifier
trained from scratch on a real (self-authored) sentiment task instead of an
MLP on synthetic XOR: for each of the model's 7 Linear layers (attention
output projections, FFN layers, and the classification head, across 2
Transformer blocks), in both the random-init and trained state, runs the
Stage-A baseline panel + the Atlas structural method at fixed parameters,
measuring tensor reconstruction error, compression ratio, and behavioral
degradation (accuracy drop, relative logit error).
"""
from __future__ import annotations

from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.lossless import zlib_baseline
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize
from atlas_nn.stage_b.experiment import run_layer_experiment, save_json
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from atlas_nn.stage_c_lite.dataset import (
    build_vocab,
    generate_sentiment_dataset,
    make_sentiment_arrays,
)
from atlas_nn.stage_c_lite.model import (
    build_transformer_classifier,
    get_weight,
    linear_layer_names,
    set_weight,
)
from atlas_nn.structural import block_dictionary_transform

SEEDS = (11, 22, 33)
N_EXAMPLES = 200
N_TRAIN = 150
MAX_LEN = 16
EPOCHS = 150
LR = 3e-3


def method_registry(seed: int) -> dict:
    return {
        "quantize_8bit_pertensor": lambda m: uniform_quantize(m, bits=8, block_size=None),
        "quantize_4bit_block64": lambda m: uniform_quantize(m, bits=4, block_size=64),
        "svd_rank4": lambda m: svd_low_rank(m, rank=4),
        "prune_50pct": lambda m: magnitude_prune(m, sparsity=0.5),
        "vector_codebook_len8_k16": lambda m: vector_codebook(m, vector_len=8, k=16, seed=seed),
        "zlib_lossless": lambda m: zlib_baseline(m),
        "atlas_block_dict16_res4bit": lambda m: block_dictionary_transform(
            m, block_shape=(8, 8), dict_size=16, residual_bits=4, seed=seed
        ),
    }


def main() -> None:
    all_rows = []
    training_summaries = []

    for seed in SEEDS:
        all_examples = generate_sentiment_dataset(N_EXAMPLES, seed=seed)
        train_examples, eval_examples = all_examples[:N_TRAIN], all_examples[N_TRAIN:]
        vocab = build_vocab([text for text, _label in train_examples])

        x_train, y_train = make_sentiment_arrays(train_examples, vocab, max_len=MAX_LEN)
        x_eval, y_eval = make_sentiment_arrays(eval_examples, vocab, max_len=MAX_LEN)

        model = build_transformer_classifier(seed=seed, vocab_size=len(vocab), max_len=MAX_LEN)
        random_init_state = snapshot(model)
        random_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        history = train_mlp(model, x_train, y_train, epochs=EPOCHS, lr=LR)
        trained_state = snapshot(model)
        trained_acc = evaluate(model, x_eval, y_eval)["accuracy"]

        training_summaries.append({
            "seed": seed, "vocab_size": len(vocab), "random_accuracy": random_acc,
            "trained_accuracy": trained_acc, **history,
        })
        print(
            f"seed={seed} vocab={len(vocab)} random_acc={random_acc:.3f} "
            f"train_acc={history['final_train_accuracy']:.3f} eval_acc={trained_acc:.3f}"
        )

        layers = linear_layer_names(model)
        methods = method_registry(seed)

        for state_label, state in (("random_init", random_init_state), ("trained", trained_state)):
            load_snapshot(model, state)
            for layer_name in layers:
                for method_name, fn in methods.items():
                    row = run_layer_experiment(
                        model=model,
                        layer_name=layer_name,
                        model_state_label=state_label,
                        method_name=method_name,
                        compress_fn=fn,
                        x_eval=x_eval,
                        y_eval=y_eval,
                        seed=seed,
                        get_weight=get_weight,
                        set_weight=set_weight,
                        evaluate=evaluate,
                        snapshot=snapshot,
                        load_snapshot=load_snapshot,
                        experiment_name="atlas_nn-stage_c_lite_smoke",
                    )
                    all_rows.append(row)
                    print(
                        f"seed={seed:>3} state={state_label:<11} layer={layer_name:<32} "
                        f"method={method_name:<28} ratio={row['compression_ratio']:6.2f} "
                        f"rel_l2={row['reconstruction']['relative_l2_error']:.4f} "
                        f"acc_drop={row['accuracy_drop']:+.3f}"
                    )

    save_json(
        {
            "experiment": "atlas_nn-stage_c_lite_smoke",
            "seeds": list(SEEDS),
            "n_examples": N_EXAMPLES,
            "n_train": N_TRAIN,
            "max_len": MAX_LEN,
            "training_summaries": training_summaries,
            "rows": all_rows,
        },
        "results/atlas_nn_stage_c_lite_smoke.json",
    )


if __name__ == "__main__":
    main()
