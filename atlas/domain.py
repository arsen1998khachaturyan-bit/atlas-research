from __future__ import annotations

from typing import Sequence


def all_sequences(
    alphabet: Sequence[int],
    max_length: int,
    *,
    include_empty: bool = True,
) -> list[tuple[int, ...]]:
    result = [tuple()] if include_empty else []
    frontier = [tuple()]

    for _ in range(max_length):
        next_frontier = []

        for prefix in frontier:
            for symbol in alphabet:
                sequence = prefix + (int(symbol),)
                result.append(sequence)
                next_frontier.append(sequence)

        frontier = next_frontier

    return result


def all_binary_sequences(
    max_length: int,
    *,
    include_empty: bool = True,
) -> list[tuple[int, ...]]:
    return all_sequences((0, 1), max_length, include_empty=include_empty)
