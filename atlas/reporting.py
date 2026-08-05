from __future__ import annotations


def _improvement(baseline: float, active: float) -> float:
    if baseline == 0:
        return 0.0

    return (baseline - active) / baseline * 100


def format_validation_report(payload: dict) -> str:
    passive = payload["mean_passive_queries"]
    random = payload["mean_random_queries"]
    active = payload["mean_active_queries"]
    total_tasks = payload["total_tasks"]

    active_vs_passive = _improvement(passive, active)
    active_vs_random = _improvement(random, active)

    lines = [
        "",
        "=" * 52,
        "ATLAS RESEARCH REPORT",
        "=" * 52,
        "",
        f'Experiment: {payload["experiment"]}',
        f'Pool size:  {payload["pool_size"]}',
        f'Max length: {payload["max_length"]}',
        f'Seeds:      {len(payload["seeds"])}',
        f'Tasks:      {total_tasks}',
        "",
        "AVERAGE QUERY COUNT",
        "-" * 52,
        f"Active:     {active:.2f}",
        f"Random:     {random:.2f}",
        f"Passive:    {passive:.2f}",
        "",
        "ACTIVE LEARNER IMPROVEMENT",
        "-" * 52,
        f"vs Random:  {active_vs_random:.1f}%",
        f"vs Passive: {active_vs_passive:.1f}%",
        "",
        "WIN RATE",
        "-" * 52,
        (
            "vs Random:  "
            f'{payload["active_wins_vs_random"]}/{total_tasks}'
        ),
        (
            "vs Passive: "
            f'{payload["active_wins_vs_passive"]}/{total_tasks}'
        ),
        "",
        "FIDELITY",
        "-" * 52,
        (
            "Full fidelity on every active-learning task: "
            f'{payload["all_active_full_fidelity"]}'
        ),
        "",
        "=" * 52,
    ]

    return "\n".join(lines)