from __future__ import annotations

import argparse
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas Research Framework",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "status",
        help="Show framework status.",
    )

    subparsers.add_parser(
        "smoke",
        help="Run the quick smoke benchmark.",
    )

    subparsers.add_parser(
        "test",
        help="Run the test suite.",
    )

    subparsers.add_parser(
        "benchmark",
        help="Run the multiseed validation benchmark.",
    )

    subparsers.add_parser(
        "plot",
        help="Generate the smoke benchmark chart.",
    )

    return parser


def run_module(module_name: str) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", module_name],
        check=False,
    )
    return completed.returncode


def run_tests() -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
    )
    return completed.returncode


def show_status() -> int:
    print("=" * 52)
    print("Atlas Research Framework v2")
    print("=" * 52)
    print("Status: READY")
    print("Branch: framework-v2")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        return show_status()

    if args.command == "smoke":
        return run_module("experiments.run_smoke")

    if args.command == "test":
        return run_tests()

    if args.command == "benchmark":
        return run_module("experiments.run_multiseed_validation")

    if args.command == "plot":
        return run_module("experiments.plot_smoke")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())