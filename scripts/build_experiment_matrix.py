"""Generate reproducible main and optional ablation experiment matrices."""

from __future__ import annotations

import argparse
from pathlib import Path


ALGORITHMS = [
    "fedavg",
    "fedprox",
    "fedavg_classweight",
    "fedrba_classagg",
    "fedrba",
]
SEEDS = [1, 2, 3]


def command(
    config: str,
    algorithm: str,
    seed: int,
    alpha: float,
    extras: list[str] | None = None,
) -> str:
    overrides = [
        f"experiment.seed={seed}",
        f"partition.dirichlet_alpha={alpha}",
        "runtime.device=auto",
    ]
    overrides.extend(extras or [])
    return (
        f"python train.py --config {config} --algorithm {algorithm} "
        f"--override {' '.join(overrides)}"
    )


def build(
    profile: str,
    *,
    include_ablations: bool = False,
    only_ablations: bool = False,
    seeds: list[int] | tuple[int, ...] = SEEDS,
) -> list[str]:
    """Return 90 full or 75 reduced main jobs, plus 9 optional ablations."""

    if include_ablations and only_ablations:
        raise ValueError(
            "include_ablations and only_ablations are mutually exclusive."
        )
    selected_seeds = [int(seed) for seed in seeds]
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("seeds must be a non-empty list without duplicates.")
    dataset_grid = {
        "configs/fashion_mnist.yaml": (
            [1.0, 0.3, 0.1] if profile == "full" else [0.3, 0.1]
        ),
        "configs/default_credit.yaml": [0.3, 0.1],
        "configs/give_me_some_credit.yaml": [0.3],
    }
    commands: list[str] = []
    if not only_ablations:
        for config, alphas in dataset_grid.items():
            for alpha in alphas:
                for algorithm in ALGORITHMS:
                    for seed in selected_seeds:
                        commands.append(command(config, algorithm, seed, alpha))

    if include_ablations or only_ablations:
        ablations = {
            "ablation_no_blindspot_loss": [
                "experiment.variant=ablation_no_blindspot_loss",
                "fedrba.use_blindspot_loss=false",
            ],
            "ablation_no_classwise_reliability": [
                "experiment.variant=ablation_no_classwise_reliability",
                "fedrba.use_classwise_aggregation=false",
            ],
            "ablation_no_update_alignment": [
                "experiment.variant=ablation_no_update_alignment",
                "fedrba.use_update_alignment=false",
            ],
        }
        for extras in ablations.values():
            for seed in selected_seeds:
                commands.append(
                    command(
                        "configs/default_credit.yaml",
                        "fedrba",
                        seed,
                        0.3,
                        extras,
                    )
                )
    if len(commands) != len(set(commands)):
        raise RuntimeError("Generated experiment matrix contains duplicate commands.")
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full", "reduced"], default="full")
    ablation_mode = parser.add_mutually_exclusive_group()
    ablation_mode.add_argument(
        "--include-ablations",
        action="store_true",
        help="Append the 9 non-duplicate Default Credit ablation jobs.",
    )
    ablation_mode.add_argument(
        "--only-ablations",
        action="store_true",
        help="Generate only the 9 non-duplicate Default Credit ablation jobs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seed subset to generate, for example --seeds 1 or --seeds 2 3.",
    )
    parser.add_argument("--output", default="scripts/experiments.txt")
    args = parser.parse_args()
    commands = build(
        args.profile,
        include_ablations=bool(args.include_ablations),
        only_ablations=bool(args.only_ablations),
        seeds=args.seeds,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {destination}")
    if args.include_ablations or args.only_ablations:
        print(
            "The 9 appended ablation jobs exclude A1 FedAvg and A5 Full "
            "FedRBA because those are reused from the main matrix."
        )


if __name__ == "__main__":
    main()
