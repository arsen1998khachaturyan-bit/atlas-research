from atlas.benchmarks.runner import (
    run_multiseed_validation,
    save_json,
)
from atlas.reporting import format_validation_report


def main() -> None:
    payload = run_multiseed_validation(
        pool_size=250,
        max_length=6,
        tasks_per_seed=5,
        seeds=(101,),
    )

    save_json(payload, "results/smoke_results.json")

    report = format_validation_report(payload)
    print(report)


if __name__ == "__main__":
    main()