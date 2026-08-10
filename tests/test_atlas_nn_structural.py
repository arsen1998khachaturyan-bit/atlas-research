from __future__ import annotations

from atlas_nn.metrics import reconstruction_metrics
from atlas_nn.structural import block_dictionary_transform
from atlas_nn.synthetic import (
    block_repeated,
    block_transformed,
    hierarchical_blocks,
    random_gaussian,
)

# Fixed method configuration used throughout: same bit budget (and therefore
# the same compression ratio) is applied to every matrix, so any difference
# in reconstruction error reflects whether real structure was found -- not a
# knob we tuned per matrix.
METHOD_KWARGS = dict(block_shape=(8, 8), dict_size=16, residual_bits=4, seed=101)


def _rel_l2(matrix) -> float:
    result = block_dictionary_transform(matrix, **METHOD_KWARGS)
    reconstructed = result.reconstruct()
    return reconstruction_metrics(matrix, reconstructed).relative_l2_error


def test_structural_method_is_near_exact_on_block_repeated_structure():
    sm = block_repeated((64, 64), block_shape=(8, 8), n_unique_blocks=4, seed=101)
    assert _rel_l2(sm.matrix) < 0.01


def test_structural_method_is_near_exact_on_block_transformed_structure():
    sm = block_transformed(
        (64, 64), block_shape=(8, 8), n_unique_blocks=4, noise_std=0.0, seed=101
    )
    assert _rel_l2(sm.matrix) < 0.01


def test_structural_method_is_near_exact_on_hierarchical_structure():
    sm = hierarchical_blocks((64, 64), seed=101)
    assert _rel_l2(sm.matrix) < 0.01


def test_structural_method_does_not_fake_compression_on_random_matrix():
    """Falsification check (mission section 11.7): at the identical bit
    budget that reconstructs structured matrices almost exactly, the method
    must NOT also achieve near-zero error on pure noise -- otherwise it
    would be silently discarding information rather than finding structure.
    """
    sm = random_gaussian((64, 64), seed=101)
    rel_l2_random = _rel_l2(sm.matrix)

    structured = block_transformed(
        (64, 64), block_shape=(8, 8), n_unique_blocks=4, noise_std=0.0, seed=101
    )
    rel_l2_structured = _rel_l2(structured.matrix)

    assert rel_l2_random > 0.03
    assert rel_l2_random > 5 * rel_l2_structured + 1e-9


def test_structural_method_reports_all_storage_components():
    sm = block_repeated((64, 64), block_shape=(8, 8), n_unique_blocks=4, seed=101)
    result = block_dictionary_transform(sm.matrix, **METHOD_KWARGS)
    expected_keys = {
        "dictionary",
        "proto_indices",
        "sign_bits",
        "scale_a",
        "shift_b",
        "residual",
    }
    assert set(result.component_bytes) == expected_keys
    assert all(v >= 0 for v in result.component_bytes.values())
    assert result.total_bytes == sum(result.component_bytes.values())
