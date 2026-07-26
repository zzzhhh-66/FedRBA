"""Build a matrix containing only experiments without valid final results."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_experiment_matrix import SEEDS, build
from src.config import load_config


def command_metadata(
    command: str,
) -> tuple[Path, int, str, str, float, int]:
    """Return expected result path and identifying metadata for one command."""

    tokens = shlex.split(command)
    config_path = tokens[tokens.index("--config") + 1]
    algorithm = tokens[tokens.index("--algorithm") + 1]
    override_index = tokens.index("--override") + 1
    config = load_config(config_path, tokens[override_index:])
    seed = int(config["experiment"]["seed"])
    dataset = str(config["dataset"]["name"])
    method = str(config["experiment"].get("variant", algorithm))
    alpha = float(config["partition"]["dirichlet_alpha"])
    scenario = str(
        config["experiment"].get(
            "scenario", f"alpha_{format(alpha, 'g').replace('.', 'p')}"
        )
    )
    result_path = (
        Path(str(config["experiment"]["output_dir"]))
        / dataset
        / method
        / scenario
        / f"seed_{seed}"
        / "final_metrics.json"
    )
    expected_round = int(config["federated"]["rounds"])
    return result_path, expected_round, algorithm, dataset, alpha, seed


def is_valid_result(
    path: Path,
    *,
    expected_round: int,
    algorithm: str,
    dataset: str,
    alpha: float,
    seed: int,
) -> bool:
    """Validate that a final result belongs to and completes the command."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(value.get("round", -1)) == expected_round
        and str(value.get("algorithm")) == algorithm
        and str(value.get("dataset")) == dataset
        and int(value.get("seed", -1)) == seed
        and abs(float(value.get("dirichlet_alpha", float("nan"))) - alpha)
        < 1e-12
    )


def pending_commands(
    profile: str,
    *,
    seeds: list[int],
    include_ablations: bool,
) -> tuple[list[str], list[Path]]:
    """Return missing commands and partial/invalid run directories."""

    commands = build(
        profile,
        include_ablations=include_ablations,
        seeds=seeds,
    )
    pending: list[str] = []
    partial: list[Path] = []
    for command in commands:
        (
            final_path,
            expected_round,
            algorithm,
            dataset,
            alpha,
            seed,
        ) = command_metadata(command)
        if final_path.is_file() and is_valid_result(
            final_path,
            expected_round=expected_round,
            algorithm=algorithm,
            dataset=dataset,
            alpha=alpha,
            seed=seed,
        ):
            continue
        run_dir = final_path.parent
        if (run_dir / "metrics.csv").exists():
            partial.append(run_dir)
        else:
            pending.append(command)
    return pending, partial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["full", "reduced"], default="full")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument(
        "--output", default="scripts/experiments_pending.txt"
    )
    args = parser.parse_args()
    pending, partial = pending_commands(
        args.profile,
        seeds=args.seeds,
        include_ablations=bool(args.include_ablations),
    )
    if partial:
        print("Partial or invalid runs must be resolved before submission:")
        for path in partial:
            print(path)
        raise SystemExit(2)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(f"{command}\n" for command in pending), encoding="utf-8"
    )
    total = len(
        build(
            args.profile,
            include_ablations=bool(args.include_ablations),
            seeds=args.seeds,
        )
    )
    print(f"Valid completed experiments: {total - len(pending)}/{total}")
    print(f"Wrote {len(pending)} pending commands to {destination}")


if __name__ == "__main__":
    main()
