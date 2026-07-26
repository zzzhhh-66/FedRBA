"""Build the auditable Default Credit log-normal quantity-skew matrix."""

from __future__ import annotations

import argparse
import csv
import shlex
from pathlib import Path
from typing import Iterable


METHODS = (
    "fedavg",
    "fedavg_classweight",
    "fedprox",
    "fedrba_classagg",
    "fedrba",
)
SMOKE_METHODS = ("fedavg", "fedrba")
FULL_ROUNDS = 150
SMOKE_ROUNDS = 2


def _tag_number(value: float) -> str:
    return format(float(value), "g").replace(".", "p")


def scenario_name(*, alpha: float, sigma: float, smoke: bool = False) -> str:
    value = (
        f"alpha_{_tag_number(alpha)}_qs_lognormal_s{_tag_number(sigma)}"
    )
    return f"{value}_smoke" if smoke else value


def _is_complete(metrics_path: Path, *, expected_rounds: int) -> bool:
    if not metrics_path.is_file():
        return False
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rounds:
        return False
    try:
        rounds = [int(row["round"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return rounds == list(range(1, expected_rounds + 1))


def build_commands(
    *,
    profile: str = "full",
    output_root: str | Path | None = None,
    alpha: float = 0.3,
    sigma: float = 1.0,
    seeds: Iterable[int] = (1, 2, 3),
    skip_completed: bool = False,
) -> tuple[list[str], int]:
    """Return safe, unique commands and the expected design size."""

    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    selected_seeds = sorted(int(value) for value in seeds)
    if (
        not selected_seeds
        or selected_seeds != sorted(set(selected_seeds))
        or any(value < 0 for value in selected_seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers.")

    smoke = profile == "smoke"
    methods = SMOKE_METHODS if smoke else METHODS
    if smoke:
        selected_seeds = [selected_seeds[0]]
    expected_rounds = SMOKE_ROUNDS if smoke else FULL_ROUNDS
    root = Path(
        output_root
        or (
            "quantity_skew_smoke_outputs"
            if smoke
            else "quantity_skew_outputs"
        )
    )
    scenario = scenario_name(alpha=alpha, sigma=sigma, smoke=smoke)

    commands: list[str] = []
    expected = len(methods) * len(selected_seeds)
    for method in methods:
        for seed in selected_seeds:
            run_dir = (
                root
                / "default_credit"
                / method
                / scenario
                / f"seed_{seed}"
            )
            metrics_path = run_dir / "metrics.csv"
            if _is_complete(metrics_path, expected_rounds=expected_rounds):
                if skip_completed:
                    continue
                raise FileExistsError(
                    f"Completed quantity-skew run already exists: {run_dir}. "
                    "Use --skip-completed."
                )
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(
                    "Incomplete non-empty quantity-skew directory requires "
                    f"manual inspection before resubmission: {run_dir}"
                )

            overrides = [
                f"experiment.seed={seed}",
                f"experiment.output_dir={root}",
                f"experiment.scenario={scenario}",
                f"partition.dirichlet_alpha={format(alpha, 'g')}",
                "partition.quantity_skew.type=lognormal",
                f"partition.quantity_skew.sigma={format(sigma, 'g')}",
                "runtime.device=auto",
            ]
            if smoke:
                overrides.extend(
                    [
                        f"federated.rounds={SMOKE_ROUNDS}",
                        "evaluation.save_every=1",
                    ]
                )
            parts = [
                "python",
                "train.py",
                "--config",
                "configs/default_credit.yaml",
                "--algorithm",
                method,
                "--override",
                *overrides,
            ]
            commands.append(" ".join(shlex.quote(value) for value in parts))

    if len(commands) != len(set(commands)):
        raise RuntimeError("Generated quantity-skew matrix contains duplicates.")
    return commands, expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands, expected = build_commands(
        profile=args.profile,
        output_root=args.output_root,
        alpha=args.alpha,
        sigma=args.sigma,
        seeds=args.seeds,
        skip_completed=bool(args.skip_completed),
    )
    default_name = (
        "scripts/experiments_quantity_skew_smoke.txt"
        if args.profile == "smoke"
        else "scripts/experiments_quantity_skew_full.txt"
    )
    destination = Path(args.output or default_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(commands) + ("\n" if commands else ""),
        encoding="utf-8",
    )
    print(
        f"Wrote {len(commands)} of {expected} {args.profile} commands "
        f"to {destination}"
    )


if __name__ == "__main__":
    main()
