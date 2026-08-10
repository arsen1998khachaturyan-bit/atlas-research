from atlas_nn.baselines.codebook import vector_codebook
from atlas_nn.baselines.common import CompressedResult
from atlas_nn.baselines.lossless import zlib_baseline
from atlas_nn.baselines.low_rank import svd_low_rank
from atlas_nn.baselines.pruning import magnitude_prune
from atlas_nn.baselines.quantization import uniform_quantize

__all__ = [
    "CompressedResult",
    "uniform_quantize",
    "svd_low_rank",
    "magnitude_prune",
    "vector_codebook",
    "zlib_baseline",
]
