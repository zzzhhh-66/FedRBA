"""Aggregate validation-selected credit operating points across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_TEST_FIELDS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "fpr",
    "f1",
    "predicted_positive_rate",
]


def _mean_std_rows(
    frame: pd.DataFrame,
    *,
    group_fields: list[str],
    value_fields: list[str],
    expected_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_fields, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        seeds = sorted(int(value) for value in group["seed"])
        if seeds != expected_seeds:
            raise ValueError(
                f"Expected seeds {expected_seeds} for {key_values}, got {seeds}."
            )
        row = dict(zip(group_fields, key_values))
        row["n_seeds"] = len(seeds)
        for field in value_fields:
            values = group[field].astype(float).to_numpy()
            row[f"{field}_mean"] = float(np.mean(values))
            row[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(
    root: str | Path,
    *,
    expected_seeds: list[int],
) -> dict[str, Path]:
    source = Path(root)
    result_paths = sorted(source.glob("**/operating_point_metrics.json"))
    expected_runs = 15 * len(expected_seeds)
    if len(result_paths) != expected_runs:
        raise FileNotFoundError(
            f"Expected {expected_runs} operating-point results, "
            f"found {len(result_paths)} under {source}."
        )

    primary_rows: list[dict[str, Any]] = []
    fpr_rows: list[dict[str, Any]] = []
    recall_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    reference_fpr_targets: tuple[float, ...] | None = None
    reference_recall_targets: tuple[float, ...] | None = None
    for path in result_paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        protocol = value["protocol"]
        if (
            protocol.get("test_selection_prohibited") is not True
            or protocol.get("test_evaluations_after_selection") != 1
        ):
            raise ValueError(f"Invalid no-test-selection audit in {path}.")
        run = value["run"]
        key = (
            str(run["dataset"]),
            str(run["scenario"]),
            str(run["method"]),
            int(run["seed"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate operating-point result for {key}.")
        seen.add(key)
        base = {
            "dataset": key[0],
            "scenario": key[1],
            "method": key[2],
            "seed": key[3],
        }
        split_audit = value["split_audit"]
        selected = value["selected_checkpoint"]
        balanced = value["balanced_accuracy_operating_point"]
        primary: dict[str, Any] = {
            **base,
            "validation_indices_sha256": split_audit[
                "validation_indices_sha256"
            ],
            "validation_targets_sha256": split_audit[
                "validation_targets_sha256"
            ],
            "test_targets_sha256": split_audit["test_targets_sha256"],
            "selected_checkpoint_round": selected["round"],
            "validation_checkpoint_pr_auc": selected["validation_pr_auc"],
            "validation_checkpoint_roc_auc": selected["validation_roc_auc"],
            "threshold_selected_on_validation": balanced[
                "threshold_selected_on_validation"
            ],
            "test_roc_auc": value["test_ranking_metrics"]["roc_auc"],
            "test_pr_auc": value["test_ranking_metrics"]["pr_auc"],
        }
        for field in PRIMARY_TEST_FIELDS:
            primary[f"validation_{field}"] = balanced["validation"][field]
            primary[f"test_{field}"] = balanced["test"][field]
        primary_rows.append(primary)

        current_fpr_targets = tuple(
            sorted(
                float(point["target_fpr"])
                for point in value["fixed_fpr_operating_points"]
            )
        )
        current_recall_targets = tuple(
            sorted(
                float(point["target_recall"])
                for point in value["fixed_recall_operating_points"]
            )
        )
        if reference_fpr_targets is None:
            reference_fpr_targets = current_fpr_targets
            reference_recall_targets = current_recall_targets
        elif (
            current_fpr_targets != reference_fpr_targets
            or current_recall_targets != reference_recall_targets
        ):
            raise ValueError(f"Inconsistent operating-point targets in {path}.")

        for point in value["fixed_fpr_operating_points"]:
            row = {
                **base,
                "target_fpr": point["target_fpr"],
                "threshold_selected_on_validation": point[
                    "threshold_selected_on_validation"
                ],
            }
            for split in ("validation", "test"):
                for field in ("fpr", "recall", "precision", "f1"):
                    row[f"{split}_{field}"] = point[split][field]
            fpr_rows.append(row)

        for point in value["fixed_recall_operating_points"]:
            row = {
                **base,
                "target_recall": point["target_recall"],
                "threshold_selected_on_validation": point[
                    "threshold_selected_on_validation"
                ],
            }
            for split in ("validation", "test"):
                for field in ("recall", "precision", "fpr", "f1"):
                    row[f"{split}_{field}"] = point[split][field]
            recall_rows.append(row)

    primary_frame = pd.DataFrame(primary_rows).sort_values(
        ["dataset", "scenario", "method", "seed"]
    )
    for audit_key, group in primary_frame.groupby(
        ["dataset", "scenario", "seed"], sort=True
    ):
        if len(group) != 5:
            raise ValueError(
                f"Expected five methods for split audit {audit_key}, got {len(group)}."
            )
        for field in (
            "validation_indices_sha256",
            "validation_targets_sha256",
            "test_targets_sha256",
        ):
            if group[field].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Methods do not share the same {field} for {audit_key}."
                )
    fpr_frame = pd.DataFrame(fpr_rows).sort_values(
        ["dataset", "scenario", "method", "target_fpr", "seed"]
    )
    recall_frame = pd.DataFrame(recall_rows).sort_values(
        ["dataset", "scenario", "method", "target_recall", "seed"]
    )

    primary_values = [
        "selected_checkpoint_round",
        "validation_checkpoint_pr_auc",
        "validation_checkpoint_roc_auc",
        "threshold_selected_on_validation",
        "test_roc_auc",
        "test_pr_auc",
        *(f"test_{field}" for field in PRIMARY_TEST_FIELDS),
    ]
    primary_aggregate = _mean_std_rows(
        primary_frame,
        group_fields=["dataset", "scenario", "method"],
        value_fields=primary_values,
        expected_seeds=expected_seeds,
    )
    fpr_aggregate = _mean_std_rows(
        fpr_frame,
        group_fields=["dataset", "scenario", "method", "target_fpr"],
        value_fields=["test_fpr", "test_recall", "test_precision", "test_f1"],
        expected_seeds=expected_seeds,
    )
    recall_aggregate = _mean_std_rows(
        recall_frame,
        group_fields=["dataset", "scenario", "method", "target_recall"],
        value_fields=["test_recall", "test_precision", "test_fpr", "test_f1"],
        expected_seeds=expected_seeds,
    )

    outputs = {
        "primary_per_run": source / "operating_point_per_run.csv",
        "primary_aggregate": source / "operating_point_aggregate.csv",
        "fixed_fpr_per_run": source / "fixed_fpr_per_run.csv",
        "fixed_fpr_aggregate": source / "fixed_fpr_aggregate.csv",
        "fixed_recall_per_run": source / "fixed_recall_per_run.csv",
        "fixed_recall_aggregate": source / "fixed_recall_aggregate.csv",
    }
    primary_frame.to_csv(outputs["primary_per_run"], index=False)
    primary_aggregate.to_csv(outputs["primary_aggregate"], index=False)
    fpr_frame.to_csv(outputs["fixed_fpr_per_run"], index=False)
    fpr_aggregate.to_csv(outputs["fixed_fpr_aggregate"], index=False)
    recall_frame.to_csv(outputs["fixed_recall_per_run"], index=False)
    recall_aggregate.to_csv(outputs["fixed_recall_aggregate"], index=False)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="operating_point_outputs")
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=[1, 2, 3]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = sorted(set(args.expected_seeds))
    if seeds != sorted(args.expected_seeds):
        raise SystemExit("--expected-seeds must not contain duplicates.")
    outputs = aggregate(args.root, expected_seeds=seeds)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
