from atlas.benchmarks.protocol import ValidationProtocol
from atlas.benchmarks.runner import run_multiseed_validation, save_json


def main() -> None:
    protocol = ValidationProtocol()
    payload = run_multiseed_validation(
        pool_size=protocol.pool_size,
        max_length=protocol.max_length,
        tasks_per_seed=protocol.tasks_per_seed,
        seeds=protocol.seeds,
    )
    save_json(payload, "results/multiseed_validation_results.json")
    print(payload)


if __name__ == "__main__":
    main()
