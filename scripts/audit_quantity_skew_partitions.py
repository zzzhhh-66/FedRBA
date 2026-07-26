"""Audit realized client-size and label heterogeneity before full training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.config import load_config
from src.data import (
    apply_lognormal_quantity_skew,
    build_dataset,
    partition_labels_dirichlet,
)


def gini(values: Sequence[int] | np.ndarray) -> float:
    """Return the non-negative Gini coefficient of positive client sizes."""

    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.isfinite(array).all()
        or (array <= 0.0).any()
    ):
        raise ValueError("Gini values must be a non-empty positive vector.")
    ordered = np.sort(array)
    indexes = np.arange(1, len(ordered) + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(indexes * ordered))
        / (len(ordered) * np.sum(ordered))
        - (len(ordered) + 1.0) / len(ordered)
    )


def _validate_complete(
    partitions: Mapping[int, Sequence[int]], *, num_samples: int
) -> None:
    flattened = np.asarray(
        [index for client_id in sorted(partitions) for index in partitions[client_id]],
        dtype=np.int64,
    )
    if len(flattened) != num_samples:
        raise RuntimeError("Partition audit found missing or duplicated samples.")
    if not np.array_equal(np.sort(flattened), np.arange(num_samples)):
        raise RuntimeError("Partition indexes do not cover the dataset exactly once.")


def summarize_partitions(
    labels: Sequence[int] | np.ndarray,
    partitions: Mapping[int, Sequence[int]],
    *,
    seed: int,
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return one summary row and auditable per-client rows."""

    target = np.asarray(labels, dtype=np.int64)
    _validate_complete(partitions, num_samples=len(target))
    classes = np.unique(target)
    global_counts = np.asarray(
        [(target == value).sum() for value in classes], dtype=np.float64
    )
    global_distribution = global_counts / global_counts.sum()

    client_rows: list[dict[str, Any]] = []
    sizes: list[int] = []
    total_variations: list[float] = []
    minority_rates: list[float] = []
    minority_class = int(classes[np.argmin(global_counts)])
    for client_id in sorted(partitions):
        indexes = np.asarray(partitions[client_id], dtype=np.int64)
        client_target = target[indexes]
        counts = np.asarray(
            [(client_target == value).sum() for value in classes],
            dtype=np.float64,
        )
        distribution = counts / counts.sum()
        total_variation = float(
            0.5 * np.abs(distribution - global_distribution).sum()
        )
        minority_rate = float((client_target == minority_class).mean())
        sizes.append(len(indexes))
        total_variations.append(total_variation)
        minority_rates.append(minority_rate)
        row: dict[str, Any] = {
            "condition": condition,
            "seed": seed,
            "client_id": int(client_id),
            "num_samples": int(len(indexes)),
            "minority_class": minority_class,
            "minority_rate": minority_rate,
            "total_variation_from_global": total_variation,
        }
        for class_value, count in zip(classes, counts):
            row[f"class_{int(class_value)}_count"] = int(count)
        client_rows.append(row)

    size_array = np.asarray(sizes, dtype=np.float64)
    summary = {
        "condition": condition,
        "seed": seed,
        "num_clients": len(size_array),
        "num_samples": int(size_array.sum()),
        "size_min": int(size_array.min()),
        "size_max": int(size_array.max()),
        "size_mean": float(size_array.mean()),
        "size_std": float(size_array.std(ddof=0)),
        "size_cv": float(size_array.std(ddof=0) / size_array.mean()),
        "size_gini": gini(size_array),
        "size_max_min_ratio": float(size_array.max() / size_array.min()),
        "minority_rate_mean": float(np.mean(minority_rates)),
        "minority_rate_std": float(np.std(minority_rates, ddof=0)),
        "label_tv_mean": float(np.mean(total_variations)),
        "label_tv_std": float(np.std(total_variations, ddof=0)),
    }
    return summary, client_rows


def audit(
    config_path: str | Path,
    output_root: str | Path,
    *,
    seeds: Iterable[int] = (1, 2, 3),
    alpha: float = 0.3,
    sigma: float = 1.0,
) -> dict[str, Path]:
    """Build real dataset partitions and write an audit bundle."""

    selected_seeds = sorted(int(value) for value in seeds)
    if (
        not selected_seeds
        or selected_seeds != sorted(set(selected_seeds))
        or any(value < 0 for value in selected_seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers.")
    if alpha <= 0.0 or sigma <= 0.0:
        raise ValueError("alpha and sigma must be positive.")

    summary_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        config = load_config(
            config_path,
            [
                f"experiment.seed={seed}",
                f"partition.dirichlet_alpha={format(alpha, 'g')}",
            ],
        )
        bundle = build_dataset(config["dataset"], seed=seed)
        partition_config = config["partition"]
        base = partition_labels_dirichlet(
            bundle.train_targets,
            num_clients=int(partition_config["num_clients"]),
            alpha=float(partition_config["dirichlet_alpha"]),
            min_client_samples=int(partition_config["min_client_samples"]),
            seed=seed,
        )
        skewed = apply_lognormal_quantity_skew(
            base,
            min_client_samples=int(partition_config["min_client_samples"]),
            sigma=sigma,
            seed=seed + 31,
        )
        for condition, partitions in (
            ("none", base),
            (f"lognormal_sigma_{format(sigma, 'g')}", skewed),
        ):
            summary, clients = summarize_partitions(
                bundle.train_targets,
                partitions,
                seed=seed,
                condition=condition,
            )
            if summary["size_min"] < int(partition_config["min_client_samples"]):
                raise RuntimeError("A client violates min_client_samples.")
            summary_rows.append(summary)
            client_rows.extend(clients)

    none_cv = np.mean(
        [row["size_cv"] for row in summary_rows if row["condition"] == "none"]
    )
    skew_cv = np.mean(
        [
            row["size_cv"]
            for row in summary_rows
            if row["condition"].startswith("lognormal")
        ]
    )
    if skew_cv <= none_cv:
        raise RuntimeError(
            "Explicit log-normal skew did not increase mean realized size CV; "
            "choose a larger sigma before running the full matrix."
        )

    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": destination / "quantity_skew_partition_summary.csv",
        "clients": destination / "quantity_skew_client_details.csv",
        "audit": destination / "quantity_skew_partition_audit.json",
    }
    with outputs["summary"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with outputs["clients"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(client_rows[0]))
        writer.writeheader()
        writer.writerows(client_rows)
    outputs["audit"].write_text(
        json.dumps(
            {
                "config": str(config_path),
                "alpha": alpha,
                "sigma": sigma,
                "seeds": selected_seeds,
                "complete": True,
                "mean_size_cv_none": float(none_cv),
                "mean_size_cv_lognormal": float(skew_cv),
                "summary_rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default_credit.yaml")
    parser.add_argument(
        "--output-root", default="quantity_skew_partition_audit"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--sigma", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = audit(
        args.config,
        args.output_root,
        seeds=args.seeds,
        alpha=args.alpha,
        sigma=args.sigma,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
