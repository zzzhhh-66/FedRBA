"""Create a compact quantity-skew analysis archive without checkpoints."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path
from typing import Iterable

from scripts.build_quantity_skew_matrix import METHODS, scenario_name
from scripts.check_quantity_skew_results import check_results


REQUIRED_RUN_FILES = (
    "metrics.csv",
    "final_metrics.json",
    "config.yaml",
    "client_partitions.json",
)


def build_archive(
    skew_root: str | Path,
    audit_root: str | Path,
    analysis_root: str | Path,
    output: str | Path,
    *,
    seeds: Iterable[int] = (1, 2, 3),
    alpha: float = 0.3,
    sigma: float = 1.0,
) -> Path:
    """Validate full outputs and package only paper-analysis artifacts."""

    selected_seeds = sorted(int(value) for value in seeds)
    check_results(
        skew_root,
        profile="full",
        seeds=selected_seeds,
        alpha=alpha,
        sigma=sigma,
    )
    source = Path(skew_root)
    audit = Path(audit_root)
    analysis = Path(analysis_root)
    for required_root in (audit, analysis):
        if not required_root.is_dir():
            raise FileNotFoundError(f"Missing analysis directory: {required_root}")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scenario = scenario_name(alpha=alpha, sigma=sigma)
    with tarfile.open(destination, mode="w") as archive:
        for method in METHODS:
            for seed in selected_seeds:
                run_dir = (
                    source
                    / "default_credit"
                    / method
                    / scenario
                    / f"seed_{seed}"
                )
                for name in REQUIRED_RUN_FILES:
                    path = run_dir / name
                    archive.add(
                        path,
                        arcname=(
                            Path(source.name) / path.relative_to(source)
                        ).as_posix(),
                    )
        for folder in (audit, analysis):
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.add(
                        path,
                        arcname=(
                            Path(folder.name) / path.relative_to(folder)
                        ).as_posix(),
                    )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skew-root", default="quantity_skew_outputs")
    parser.add_argument(
        "--audit-root", default="quantity_skew_partition_audit"
    )
    parser.add_argument("--analysis-root", default="quantity_skew_analysis")
    parser.add_argument(
        "--output", default="quantity_skew_results_for_analysis.tar"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_archive(
        args.skew_root,
        args.audit_root,
        args.analysis_root,
        args.output,
        seeds=args.seeds,
        alpha=args.alpha,
        sigma=args.sigma,
    )
    print(f"Wrote compact quantity-skew archive: {output}")


if __name__ == "__main__":
    main()
