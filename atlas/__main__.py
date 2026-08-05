from __future__ import annotations

import argparse
import subprocess
import sys


def run_module(module_name: str) -> int:
    """Run a Python module using the current interpreter."""
    completed = subprocess.run(
        [sys.executable, "-m", module_name],
        check=False,
    )
    return completed.returncode


def run_tests() -> int:
    """Run the Atlas test suite."""
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
    print()
    print("Available commands:")
    print("  python -m atlas status")
    print("  python -m atlas test")
    print("  python -m atlas smoke")
    print("  python -m atlas benchmark")
    print("  python -m atlas plot")
    print("=" * 52)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description=(
            "Atlas Research Framework for query-efficient "
            "symbolic program extraction."
        ),
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=[
            "status",
            "test",
            "smoke",
            "benchmark",
            "plot",
        ],
        help="Command to execute.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        return show_status()

    if args.command == "test":
        return run_tests()

    if args.command == "smoke":
        return run_module("experiments.run_smoke")

    if args.command == "benchmark":
        return run_module("experiments.run_multiseed_validation")

    if args.command == "plot":
        return run_module("experiments.plot_smoke")

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())