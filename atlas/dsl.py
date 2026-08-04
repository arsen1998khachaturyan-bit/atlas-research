from __future__ import annotations

from atlas.programs import Program


def build_atoms(
    *,
    max_modulus: int = 6,
    max_pattern_length: int = 4,
) -> list[Program]:
    atoms: list[Program] = [
        Program("palindrome", 7, lambda s: s == s[::-1]),
        Program(
            "majority_ones",
            7,
            lambda s: len(s) > 0 and sum(s) * 2 > len(s),
        ),
        Program(
            "first_eq_last",
            7,
            lambda s: len(s) > 0 and s[0] == s[-1],
        ),
        Program(
            "alternating",
            7,
            lambda s: all(s[i] != s[i - 1] for i in range(1, len(s))),
        ),
    ]

    for modulus in range(2, max_modulus + 1):
        for remainder in range(modulus):
            atoms.append(
                Program(
                    f"count1_mod_{modulus}_eq_{remainder}",
                    10,
                    lambda s, m=modulus, r=remainder: sum(s) % m == r,
                )
            )
            atoms.append(
                Program(
                    f"length_mod_{modulus}_eq_{remainder}",
                    10,
                    lambda s, m=modulus, r=remainder: len(s) % m == r,
                )
            )

    for length in range(1, max_pattern_length + 1):
        for value in range(2**length):
            pattern = tuple(
                (value >> bit) & 1
                for bit in reversed(range(length))
            )
            text = "".join(str(x) for x in pattern)

            atoms.extend([
                Program(
                    f"starts_with_{text}",
                    8 + length,
                    lambda s, p=pattern: len(s) >= len(p) and s[:len(p)] == p,
                ),
                Program(
                    f"ends_with_{text}",
                    8 + length,
                    lambda s, p=pattern: len(s) >= len(p) and s[-len(p):] == p,
                ),
                Program(
                    f"contains_{text}",
                    9 + length,
                    lambda s, p=pattern: any(
                        s[i:i + len(p)] == p
                        for i in range(len(s) - len(p) + 1)
                    ),
                ),
            ])

    return atoms


def compose_not(program: Program) -> Program:
    return Program(
        f"NOT({program.name})",
        program.complexity_bits + 2,
        lambda s, p=program: not p.run(s),
    )


def compose_and(left: Program, right: Program) -> Program:
    return Program(
        f"({left.name}) AND ({right.name})",
        left.complexity_bits + right.complexity_bits + 3,
        lambda s, a=left, b=right: a.run(s) and b.run(s),
    )


def compose_or(left: Program, right: Program) -> Program:
    return Program(
        f"({left.name}) OR ({right.name})",
        left.complexity_bits + right.complexity_bits + 3,
        lambda s, a=left, b=right: a.run(s) or b.run(s),
    )


def compose_xor(left: Program, right: Program) -> Program:
    return Program(
        f"({left.name}) XOR ({right.name})",
        left.complexity_bits + right.complexity_bits + 3,
        lambda s, a=left, b=right: bool(a.run(s)) ^ bool(b.run(s)),
    )
