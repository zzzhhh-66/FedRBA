"""Tests for audited aggregation of validation-selected operating points."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.aggregate_operating_points import aggregate


METHODS = [
    "fedavg",
    "fedavg_classweight",
    "fedprox",
    "fedrba",
    "fedrba_classagg",
]
SCENARIOS = [
    ("default_credit", "alpha_0p1"),
    ("default_credit", "alpha_0p3"),
    ("give_me_some_credit", "alpha_0p3"),
]


def _metrics(value: float) -> dict[str, float]:
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "precision": value,
        "recall": value,
        "fpr": 1.0 - value,
        "f1": value,
        "predicted_positive_rate": value,
    }


def _write_results(root: Path, *, corrupt_hash: bool = False) -> None:
    for dataset, scenario in SCENARIOS:
        for method_id, method in enumerate(METHODS):
            for seed in (1, 2, 3):
                value = 0.5 + 0.01 * method_id + 0.001 * seed
                validation_hash = f"{dataset}-{scenario}-{seed}-validation"
                if corrupt_hash and method == "fedrba" and seed == 1:
                    validation_hash += "-corrupt"
                payload = {
                    "protocol": {
                        "test_selection_prohibited": True,
                        "test_evaluations_after_selection": 1,
                    },
                    "run": {
                        "dataset": dataset,
                        "scenario": scenario,
                        "method": method,
                        "seed": seed,
                    },
                    "split_audit": {
                        "validation_indices_sha256": validation_hash,
                        "validation_targets_sha256": (
                            f"{dataset}-{scenario}-{seed}-validation-targets"
                        ),
                        "test_targets_sha256": f"{dataset}-{seed}-test-targets",
                    },
                    "selected_checkpoint": {
                        "round": 20,
                        "validation_pr_auc": value,
                        "validation_roc_auc": value,
                    },
                    "test_ranking_metrics": {
                        "roc_auc": value,
                        "pr_auc": value,
                    },
                    "balanced_accuracy_operating_point": {
                        "threshold_selected_on_validation": 0.4,
                        "validation": _metrics(value),
                        "test": _metrics(value),
                    },
                    "fixed_fpr_operating_points": [
                        {
                            "target_fpr": target,
                            "threshold_selected_on_validation": 0.4,
                            "validation": _metrics(value),
                            "test": _metrics(value),
                        }
                        for target in (0.01, 0.05, 0.10)
                    ],
                    "fixed_recall_operating_points": [
                        {
                            "target_recall": target,
                            "threshold_selected_on_validation": 0.4,
                            "validation": _metrics(value),
                            "test": _metrics(value),
                        }
                        for target in (0.50, 0.70, 0.80)
                    ],
                }
                path = (
                    root
                    / dataset
                    / method
                    / scenario
                    / f"seed_{seed}"
                    / "operating_point_metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_writes_six_audited_tables(tmp_path: Path) -> None:
    _write_results(tmp_path)
    outputs = aggregate(tmp_path, expected_seeds=[1, 2, 3])
    assert len(outputs) == 6
    assert all(path.is_file() for path in outputs.values())
    assert len(pd.read_csv(outputs["primary_per_run"])) == 45
    assert len(pd.read_csv(outputs["primary_aggregate"])) == 15
    assert len(pd.read_csv(outputs["fixed_fpr_aggregate"])) == 45
    assert len(pd.read_csv(outputs["fixed_recall_aggregate"])) == 45


def test_aggregate_rejects_different_validation_split(tmp_path: Path) -> None:
    _write_results(tmp_path, corrupt_hash=True)
    with pytest.raises(ValueError, match="same validation_indices_sha256"):
        aggregate(tmp_path, expected_seeds=[1, 2, 3])
