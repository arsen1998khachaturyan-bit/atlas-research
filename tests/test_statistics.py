import pytest

from atlas.benchmarks.statistics import summarize


def test_summarize_multiple_values() -> None:
    result = summarize([10, 12, 14, 16, 18])

    assert result.count == 5
    assert result.mean == pytest.approx(14.0)
    assert result.minimum == 10.0
    assert result.maximum == 18.0
    assert result.standard_deviation > 0
    assert result.confidence_interval_95_low < result.mean
    assert result.confidence_interval_95_high > result.mean


def test_summarize_single_value() -> None:
    result = summarize([11.2])

    assert result.count == 1
    assert result.mean == pytest.approx(11.2)
    assert result.standard_deviation == 0.0
    assert result.confidence_interval_95_low == pytest.approx(11.2)
    assert result.confidence_interval_95_high == pytest.approx(11.2)


def test_summarize_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        summarize([])