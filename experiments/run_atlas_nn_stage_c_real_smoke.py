"""Stage C (real): does the behavioral-robustness effect appear on a
literal pretrained model? (see atlas_nn/stage_c_real/model.py for why this
is now possible -- huggingface.co was blocked earlier this session and is
no longer, per the user's own network-policy change.)

Mirrors Experiment 3/8's smoke pattern: for a fixed set of compression
methods, on each of distilgpt2's 12 tested Conv1D layers (3 depths x 4
sublayer types), compares tensor reconstruction error and behavioral error
(relative_logit_error, teacher-forced next-token accuracy) between the real
pretrained checkpoint and freshly-initialized (untrained) copies of the
same architecture.

Unlike Stage B/C-lite, there is no training step here and therefore no
training-success confound to check: the pretrained arm is one fixed,
already-trained checkpoint (no seed); the random-init arm is instantiated
fresh, with zero optimization performed, across 3 seeds.
"""
from __future__ import annotations

from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.lossless import zlib_baseline
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize
from atlas_nn.stage_b.experiment import run_layer_experiment, save_json
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
    tested_layer_names,
)
from atlas_nn.structural import block_dictionary_transform

RANDOM_SEEDS = (11, 22, 33)


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


def run_state(model, state_label: str, seed: int, x_eval, all_rows: list) -> None:
    layers = linear_layer_names(model)
    methods = method_registry(seed)
    for layer_name in layers:
        for method_name, fn in methods.items():
            row = run_layer_experiment(
                model=model,
                layer_name=layer_name,
                model_state_label=state_label,
                method_name=method_name,
                compress_fn=fn,
                x_eval=x_eval,
                y_eval=None,
                seed=seed,
                get_weight=get_weight,
                set_weight=set_weight,
                evaluate=evaluate,
                snapshot=snapshot,
                load_snapshot=load_snapshot,
                experiment_name="atlas_nn-stage_c_real_smoke",
            )
            all_rows.append(row)
            print(
                f"state={state_label:<11} seed={seed:>3} layer={layer_name:<32} "
                f"method={method_name:<28} ratio={row['compression_ratio']:6.2f} "
                f"rel_l2={row['reconstruction']['relative_l2_error']:.4f} "
                f"rel_logit={row['relative_logit_error']:.4f}",
                flush=True,
            )


def main(model_name: str = "distilgpt2", output_path: str = "results/atlas_nn_stage_c_real_smoke.json") -> None:
    tokenizer = load_tokenizer(model_name)
    x_eval = build_eval_batch(tokenizer)

    all_rows: list = []

    pretrained_model = load_pretrained(model_name)
    pretrained_acc = evaluate(pretrained_model, x_eval, None)["accuracy"]
    print(f"pretrained next-token accuracy on eval batch: {pretrained_acc:.3f}", flush=True)
    run_state(pretrained_model, "pretrained", seed=0, x_eval=x_eval, all_rows=all_rows)
    del pretrained_model

    random_accuracies = {}
    for seed in RANDOM_SEEDS:
        random_model = load_random_init(seed, model_name)
        random_acc = evaluate(random_model, x_eval, None)["accuracy"]
        random_accuracies[seed] = random_acc
        print(f"seed={seed} random-init next-token accuracy: {random_acc:.3f}", flush=True)
        run_state(random_model, "random_init", seed=seed, x_eval=x_eval, all_rows=all_rows)
        del random_model

    save_json(
        {
            "experiment": "atlas_nn-stage_c_real_smoke",
            "model_name": model_name,
            "random_seeds": list(RANDOM_SEEDS),
            "pretrained_accuracy": pretrained_acc,
            "random_init_accuracies": random_accuracies,
            "tested_layers": linear_layer_names(None) if model_name == "distilgpt2" else tested_layer_names(model_name),
            "rows": all_rows,
        },
        output_path,
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
