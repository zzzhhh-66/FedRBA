"""Build the 18 new Default Credit ablation commands for two alphas."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path
from typing import Iterable


ABLATIONS = {
    "ablation_no_blindspot_loss": (
        "fedrba.use_blindspot_loss=false",
    ),
    "ablation_no_classwise_reliability": (
        "fedrba.use_classwise_aggregation=false",
    ),
    "ablation_no_update_alignment": (
        "fedrba.use_update_alignment=false",
    ),
}
EXPECTED_ROUNDS = 150


def _scenario(alpha: float) -> str:
    return f"alpha_{format(float(alpha), 'g').replace('.', 'p')}"


def _is_complete(metrics_path: Path) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROUNDS:
        return False
    try:
        rounds = [int(row["round"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return rounds == list(range(1, EXPECTED_ROUNDS + 1))


def build_commands(
    *,
    output_root: str | Path = "outputs",
    alphas: Iterable[float] = (0.1, 0.3),
    seeds: Iterable[int] = (1, 2, 3),
    skip_completed: bool = False,
) -> tuple[list[str], int]:
    """Return commands and the total expected design size."""

    selected_alphas = sorted(float(value) for value in alphas)
    selected_seeds = sorted(int(value) for value in seeds)
    if (
        not selected_alphas
        or selected_alphas != sorted(set(selected_alphas))
        or any(value <= 0.0 for value in selected_alphas)
    ):
        raise ValueError("alphas must be unique positive values.")
    if (
        not selected_seeds
        or selected_seeds != sorted(set(selected_seeds))
        or any(value < 0 for value in selected_seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers.")

    root = Path(output_root)
    commands: list[str] = []
    expected = len(ABLATIONS) * len(selected_alphas) * len(selected_seeds)
    for alpha in selected_alphas:
        scenario = _scenario(alpha)
        for variant, variant_overrides in ABLATIONS.items():
            for seed in selected_seeds:
                run_dir = (
                    root
                    / "default_credit"
                    / variant
                    / scenario
                    / f"seed_{seed}"
                )
                metrics_path = run_dir / "metrics.csv"
                if _is_complete(metrics_path):
                    if skip_completed:
                        continue
                    raise FileExistsError(
                        f"Completed ablation already exists: {run_dir}. "
                        "Use --skip-completed."
                    )
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise RuntimeError(
                        f"Incomplete non-empty ablation directory requires "
                        f"manual inspection before resubmission: {run_dir}"
                    )
                overrides = [
                    f"experiment.seed={seed}",
                    f"experiment.output_dir={root}",
                    f"experiment.variant={variant}",
                    f"partition.dirichlet_alpha={format(alpha, 'g')}",
                    "runtime.device=auto",
                    *variant_overrides,
                ]
                parts = [
                    "python",
                    "train.py",
                    "--config",
                    "configs/default_credit.yaml",
                    "--algorithm",
                    "fedrba",
                    "--override",
                    *overrides,
                ]
                commands.append(
                    " ".join(shlex.quote(value) for value in parts)
                )
    if len(commands) != len(set(commands)):
        raise RuntimeError("Generated ablation matrix contains duplicates.")
    return commands, expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--output",
        default="scripts/experiments_credit_ablations_alpha01_alpha03.txt",
    )
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands, expected = build_commands(
        output_root=args.output_root,
        alphas=args.alphas,
        seeds=args.seeds,
        skip_completed=bool(args.skip_completed),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(commands) + ("\n" if commands else ""),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(commands)} of {expected} ablation commands to {destination}"
    )


if __name__ == "__main__":
    main()
