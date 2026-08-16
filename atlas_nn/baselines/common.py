from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class CompressedResult:
    """Common output of every compression method.

    `component_bytes` MUST include every piece of information required to
    reconstruct the tensor (codebooks, transform parameters, indices,
    residuals, metadata) -- never just the "interesting" part. This is what
    makes `total_bytes` an honest storage cost rather than an estimate.
    """

    method: str
    params: dict
    component_bytes: dict[str, int]
    reconstruct: Callable[[], np.ndarray]

    @property
    def total_bytes(self) -> int:
        return sum(self.component_bytes.values())
