"""Fail-fast completeness check for quantity-skew smoke or full outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable

import yaml

from scripts.build_quantity_skew_matrix import (
    FULL_ROUNDS,
    METHODS,
    SMOKE_METHODS,
    SMOKE_ROUNDS,
    scenario_name,
)


def _valid_rounds(path: Path, expected_rounds: int) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        rounds = [int(row["round"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    return rounds == list(range(1, expected_rounds + 1))


def check_results(
    root: str | Path,
    *,
    profile: str,
    seeds: Iterable[int] = (1, 2, 3),
    alpha: float = 0.3,
    sigma: float = 1.0,
) -> dict[str, int]:
    """Validate expected files, rounds, config, and cross-method partitions."""

    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be 'smoke' or 'full'.")
    selected_seeds = sorted(int(value) for value in seeds)
    if not selected_seeds:
        raise ValueError("At least one seed is required.")
    smoke = profile == "smoke"
    if smoke:
        selected_seeds = [selected_seeds[0]]
    methods = SMOKE_METHODS if smoke else METHODS
    rounds = SMOKE_ROUNDS if smoke else FULL_ROUNDS
    scenario = scenario_name(alpha=alpha, sigma=sigma, smoke=smoke)
    source = Path(root)

    errors: list[str] = []
    hashes: dict[int, set[str]] = {}
    completed = 0
    for method in methods:
        for seed in selected_seeds:
            run_dir = (
                source
                / "default_credit"
                / method
                / scenario
                / f"seed_{seed}"
            )
            if not _valid_rounds(run_dir / "metrics.csv", rounds):
                errors.append(f"invalid/missing metrics: {run_dir}")
                continue
            missing_required = False
            for required in (
                "final_metrics.json",
                "config.yaml",
                "client_partitions.json",
            ):
                if not (run_dir / required).is_file():
                    errors.append(f"missing {required}: {run_dir}")
                    missing_required = True
            if missing_required:
                continue
            config = yaml.safe_load(
                (run_dir / "config.yaml").read_text(encoding="utf-8")
            )
            quantity = config["partition"]["quantity_skew"]
            if (
                quantity.get("type") != "lognormal"
                or float(quantity.get("sigma", 0.0)) != float(sigma)
            ):
                errors.append(f"wrong quantity-skew config: {run_dir}")
                continue
            digest = hashlib.sha256(
                (run_dir / "client_partitions.json").read_bytes()
            ).hexdigest()
            hashes.setdefault(seed, set()).add(digest)
            completed += 1

    for seed, values in hashes.items():
        if len(values) != 1:
            errors.append(
                f"methods use different client partitions for seed {seed}"
            )
    expected = len(methods) * len(selected_seeds)
    if errors:
        details = "\n".join(f"- {value}" for value in errors)
        raise RuntimeError(
            f"Quantity-skew {profile} check failed "
            f"({completed}/{expected} complete):\n{details}"
        )
    return {"completed": completed, "expected": expected}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_root = (
        "quantity_skew_smoke_outputs"
        if args.profile == "smoke"
        else "quantity_skew_outputs"
    )
    result = check_results(
        args.root or default_root,
        profile=args.profile,
        seeds=args.seeds,
        alpha=args.alpha,
        sigma=args.sigma,
    )
    print(
        f"QUANTITY-SKEW {args.profile.upper()} VALID: "
        f"{result['completed']}/{result['expected']} runs complete"
    )


if __name__ == "__main__":
    main()
