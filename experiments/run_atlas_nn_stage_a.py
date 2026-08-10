"""Stage A benchmark (mission section 9): compare Atlas structural method
against baselines on controlled synthetic matrices -- random, low-rank,
block-repeated, block-transformed, structured+noise, hierarchical.

A good method should exploit structure where it exists and should NOT claim
strong compression on the random matrix (see tests/test_atlas_nn_structural.py
for the corresponding falsification test).
"""
from __future__ import annotations

from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.lossless import zlib_baseline
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize
from atlas_nn.experiment import run_single_experiment, save_json
from atlas_nn.structural import block_dictionary_transform
from atlas_nn.synthetic import generate_stage_a_suite

SEEDS = (101, 202, 303)
SHAPE = (64, 64)


def method_registry(seed: int) -> dict:
    return {
        "quantize_8bit_pertensor": lambda m: uniform_quantize(m, bits=8, block_size=None),
        "quantize_4bit_block64": lambda m: uniform_quantize(m, bits=4, block_size=64),
        "svd_rank4": lambda m: svd_low_rank(m, rank=4),
        "svd_rank8": lambda m: svd_low_rank(m, rank=8),
        "prune_50pct": lambda m: magnitude_prune(m, sparsity=0.5),
        "prune_90pct": lambda m: magnitude_prune(m, sparsity=0.9),
        "vector_codebook_len8_k16": lambda m: vector_codebook(m, vector_len=8, k=16, seed=seed),
        "zlib_lossless": lambda m: zlib_baseline(m),
        "atlas_block_dict8_res4bit": lambda m: block_dictionary_transform(
            m, block_shape=(8, 8), dict_size=8, residual_bits=4, seed=seed
        ),
        "atlas_block_dict16_res4bit": lambda m: block_dictionary_transform(
            m, block_shape=(8, 8), dict_size=16, residual_bits=4, seed=seed
        ),
    }


def main() -> None:
    all_rows = []

    for seed in SEEDS:
        matrices = generate_stage_a_suite(SHAPE, seed=seed)
        methods = method_registry(seed)

        for sm in matrices:
            for method_name, fn in methods.items():
                row = run_single_experiment(
                    method_name=method_name,
                    matrix_name=sm.name,
                    matrix=sm.matrix,
                    seed=seed,
                    compress_fn=fn,
                )
                all_rows.append(row)
                print(
                    f"seed={seed:>4} matrix={sm.name:<22} method={method_name:<28} "
                    f"ratio={row['compression_ratio']:6.2f} "
                    f"rel_l2={row['reconstruction']['relative_l2_error']:.4f}"
                )

    save_json({"experiment": "atlas_nn-stage_a", "shape": list(SHAPE), "seeds": list(SEEDS), "rows": all_rows},
               "results/atlas_nn_stage_a.json")


if __name__ == "__main__":
    main()
