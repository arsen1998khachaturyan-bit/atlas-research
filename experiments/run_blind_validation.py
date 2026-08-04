from atlas.benchmarks.runner import run_multiseed_validation, save_json


def main() -> None:
    payload = run_multiseed_validation(
        pool_size=1500,
        max_length=8,
        tasks_per_seed=60,
        seeds=(20268111,),
    )
    save_json(payload, "results/blind_validation_results.json")
    print(payload)


if __name__ == "__main__":
    main()
