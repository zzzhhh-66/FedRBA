"""Build the 45-command credit operating-point evaluation matrix."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


DATASET_SCENARIOS = {
    "default_credit": {"alpha_0p1", "alpha_0p3"},
    "give_me_some_credit": {"alpha_0p3"},
}
METHODS = {
    "fedavg",
    "fedavg_classweight",
    "fedprox",
    "fedrba",
    "fedrba_classagg",
}


def discover_run_dirs(root: str | Path, seeds: list[int]) -> list[Path]:
    source = Path(root)
    discovered: list[Path] = []
    for dataset, scenarios in DATASET_SCENARIOS.items():
        for method in sorted(METHODS):
            for scenario in sorted(scenarios):
                for seed in sorted(seeds):
                    run_dir = source / dataset / method / scenario / f"seed_{seed}"
                    required = [
                        run_dir / "config.yaml",
                        run_dir / "client_partitions.json",
                        run_dir / "checkpoints",
                    ]
                    if not all(path.exists() for path in required):
                        raise FileNotFoundError(
                            f"Incomplete source run {run_dir}; missing one of "
                            "config.yaml, client_partitions.json, checkpoints/."
                        )
                    if not list((run_dir / "checkpoints").glob("round_*.pt")):
                        raise FileNotFoundError(
                            f"No periodic checkpoints under {run_dir}."
                        )
                    discovered.append(run_dir)
    return discovered


def build_commands(
    run_dirs: list[Path],
    *,
    output_root: str | Path,
    device: str,
    fpr_targets: list[float],
    recall_targets: list[float],
) -> list[str]:
    commands: list[str] = []
    for run_dir in run_dirs:
        parts = [
            "python",
            "scripts/evaluate_credit_operating_points.py",
            "--run-dir",
            str(run_dir),
            "--output-root",
            str(output_root),
            "--device",
            str(device),
            "--fpr-targets",
            *(format(value, "g") for value in fpr_targets),
            "--recall-targets",
            *(format(value, "g") for value in recall_targets),
        ]
        commands.append(" ".join(shlex.quote(value) for value in parts))
    return commands


def remove_completed(
    run_dirs: list[Path], *, output_root: str | Path
) -> list[Path]:
    """Return runs without an existing audited result."""

    destination = Path(output_root)
    pending: list[Path] = []
    for run_dir in run_dirs:
        result = (
            destination
            / run_dir.parents[2].name
            / run_dir.parents[1].name
            / run_dir.parents[0].name
            / run_dir.name
            / "operating_point_metrics.json"
        )
        if not result.is_file():
            pending.append(run_dir)
    return pending


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output-root", default="operating_point_outputs")
    parser.add_argument(
        "--output", default="scripts/operating_point_experiments.txt"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Generate only runs without operating_point_metrics.json.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--fpr-targets",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10],
    )
    parser.add_argument(
        "--recall-targets",
        nargs="+",
        type=float,
        default=[0.50, 0.70, 0.80],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = sorted(set(args.seeds))
    if seeds != sorted(args.seeds):
        raise SystemExit("--seeds must not contain duplicates.")
    run_dirs = discover_run_dirs(args.root, seeds)
    expected = 15 * len(seeds)
    if len(run_dirs) != expected:
        raise RuntimeError(f"Expected {expected} source runs, found {len(run_dirs)}.")
    if args.skip_completed:
        run_dirs = remove_completed(run_dirs, output_root=args.output_root)
    commands = build_commands(
        run_dirs,
        output_root=args.output_root,
        device=args.device,
        fpr_targets=args.fpr_targets,
        recall_targets=args.recall_targets,
    )
    if len(commands) != len(set(commands)):
        raise RuntimeError(
            f"Generated {len(commands)} commands with duplicates."
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(commands)} of {expected} commands to {destination}"
        + (" (completed results skipped)" if args.skip_completed else "")
    )


if __name__ == "__main__":
    main()
