from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/smoke_results.json")
OUTPUT_PATH = Path("results/smoke_query_comparison.png")


def load_results(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}. "
            "Run `python -m atlas smoke` first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_chart(payload: dict) -> None:
    algorithms = ["Active", "Random", "Passive"]

    mean_queries = [
        payload["mean_active_queries"],
        payload["mean_random_queries"],
        payload["mean_passive_queries"],
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        algorithms,
        mean_queries,
    )

    ax.set_title("Atlas Smoke Benchmark")
    ax.set_xlabel("Learning strategy")
    ax.set_ylabel("Average membership queries")
    ax.set_ylim(0, max(mean_queries) * 1.2)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, mean_queries):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_queries) * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    print(f"Chart saved to: {OUTPUT_PATH}")


def main() -> None:
    payload = load_results(RESULTS_PATH)
    build_chart(payload)


if __name__ == "__main__":
    main()