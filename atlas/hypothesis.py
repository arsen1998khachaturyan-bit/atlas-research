from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from atlas.dsl import (
    build_atoms,
    compose_and,
    compose_not,
    compose_or,
    compose_xor,
)
from atlas.programs import Program


def behavior_matrix(
    programs: Sequence[Program],
    domain: Sequence[tuple[int, ...]],
) -> np.ndarray:
    return np.asarray(
        [
            [program.run(sequence) for sequence in domain]
            for program in programs
        ],
        dtype=np.uint8,
    )


def behavior_signature(
    program: Program,
    domain: Sequence[tuple[int, ...]],
) -> bytes:
    return bytes(program.run(sequence) for sequence in domain)


def generate_hypothesis_pool(
    *,
    target_size: int,
    domain: Sequence[tuple[int, ...]],
    seed: int,
    max_modulus: int = 6,
    max_pattern_length: int = 4,
) -> list[Program]:
    rng = random.Random(seed)
    atoms = build_atoms(
        max_modulus=max_modulus,
        max_pattern_length=max_pattern_length,
    )

    unique: dict[bytes, Program] = {}

    for atom in atoms:
        unique.setdefault(behavior_signature(atom, domain), atom)

    attempts = 0
    max_attempts = target_size * 100

    while len(unique) < target_size and attempts < max_attempts:
        attempts += 1
        depth = rng.choices([1, 2, 3], weights=[2, 5, 3], k=1)[0]

        if depth == 1:
            candidate = compose_not(rng.choice(atoms))
        elif depth == 2:
            left, right = rng.sample(atoms, 2)
            candidate = rng.choice(
                [compose_and, compose_or, compose_xor]
            )(left, right)
        else:
            first, second, third = rng.sample(atoms, 3)
            inner = rng.choice(
                [compose_and, compose_or, compose_xor]
            )(first, second)
            candidate = rng.choice(
                [compose_and, compose_or, compose_xor]
            )(inner, third)

        unique.setdefault(
            behavior_signature(candidate, domain),
            candidate,
        )

    programs = list(unique.values())
    programs.sort(key=lambda p: (p.complexity_bits, p.name))
    return programs[:target_size]
