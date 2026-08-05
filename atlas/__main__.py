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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "status":
        print("=" * 50)
        print("Atlas Research Framework v2")
        print("=" * 50)
        print("Status: READY")
        print("Branch: framework-v2")
        return 0

    if args.command == "smoke":
        return run_module("experiments.run_smoke")

    if args.command == "test":
        return run_tests()

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())