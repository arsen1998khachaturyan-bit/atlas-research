from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_interval_95_low: float
    confidence_interval_95_high: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarize(values: Iterable[float | int]) -> MetricSummary:
    samples = [float(value) for value in values]

    if not samples:
        raise ValueError("At least one value is required.")

    count = len(samples)
    mean = statistics.fmean(samples)

    if count == 1:
        standard_deviation = 0.0
        standard_error = 0.0
        margin = 0.0
    else:
        standard_deviation = statistics.stdev(samples)
        standard_error = standard_deviation / math.sqrt(count)
        margin = 1.96 * standard_error

    return MetricSummary(
        count=count,
        mean=mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        confidence_interval_95_low=mean - margin,
        confidence_interval_95_high=mean + margin,
        minimum=min(samples),
        maximum=max(samples),
    )