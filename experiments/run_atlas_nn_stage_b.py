"""Stage B (mission section 9): does training create compressible structure
in weight tensors, compared to the same architecture at random init?

For each seed: build an MLP, snapshot its random-init weights, train it on
a synthetic XOR-in-noise task, then run the full Stage A method panel
(baselines + Atlas structural method) on each Linear layer's weight matrix
in BOTH the random-init state and the trained state, measuring tensor
reconstruction error, compression ratio, AND behavioral degradation
(accuracy drop, relative logit error) from substituting the reconstructed
weight back into the network.
"""
from __future__ import annotations

from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.lossless import zlib_baseline
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize
from atlas_nn.stage_b.dataset import make_xor_dataset
from atlas_nn.stage_b.experiment import run_layer_experiment, save_json
from atlas_nn.stage_b.model import build_mlp, get_weight, linear_layer_names, set_weight
from atlas_nn.stage_b.train import evaluate, load_snapshot, snapshot, train_mlp
from atlas_nn.structural import block_dictionary_transform

SEEDS = (11, 22, 33)
N_TRAIN = 2000
N_EVAL = 500
N_FEATURES = 32


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
        x_train, y_train = make_xor_dataset(N_TRAIN, n_features=N_FEATURES, seed=seed)
        x_eval, y_eval = make_xor_dataset(N_EVAL, n_features=N_FEATURES, seed=seed + 50_000)

        model = build_mlp(seed, input_dim=N_FEATURES)
        random_init_state = snapshot(model)

        history = train_mlp(model, x_train, y_train, epochs=500, lr=2e-2)
        trained_state = snapshot(model)

        training_summaries.append({"seed": seed, **history})
        print(f"seed={seed} trained: loss={history['final_train_loss']:.4f} "
              f"acc={history['final_train_accuracy']:.3f}")

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
                    )
                    all_rows.append(row)
                    print(
                        f"seed={seed:>3} state={state_label:<11} layer={layer_name:<3} "
                        f"method={method_name:<28} ratio={row['compression_ratio']:6.2f} "
                        f"rel_l2={row['reconstruction']['relative_l2_error']:.4f} "
                        f"acc_drop={row['accuracy_drop']:+.3f}"
                    )

    save_json(
        {
            "experiment": "atlas_nn-stage_b",
            "seeds": list(SEEDS),
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "n_features": N_FEATURES,
            "training_summaries": training_summaries,
            "rows": all_rows,
        },
        "results/atlas_nn_stage_b.json",
    )


if __name__ == "__main__":
    main()
