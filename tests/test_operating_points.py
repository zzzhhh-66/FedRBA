"""Tests for validation-only binary operating-point selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics.operating_points import (
    binary_metrics_at_threshold,
    select_balanced_accuracy_threshold,
    select_thresholds_at_fpr,
    select_thresholds_at_recall,
)


TARGETS = np.asarray([1, 0, 1, 0], dtype=np.int64)
SCORES = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)


def test_binary_metrics_at_threshold() -> None:
    metrics = binary_metrics_at_threshold(TARGETS, SCORES, 0.7)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert np.isclose(metrics["recall"], 1.0)
    assert np.isclose(metrics["precision"], 2 / 3)
    assert np.isclose(metrics["fpr"], 0.5)
    assert np.isclose(metrics["balanced_accuracy"], 0.75)


def test_balanced_accuracy_tie_prefers_lower_fpr() -> None:
    selected = select_balanced_accuracy_threshold(TARGETS, SCORES)
    assert np.isclose(selected["threshold"], 0.9)
    assert np.isclose(selected["balanced_accuracy"], 0.75)
    assert np.isclose(selected["fpr"], 0.0)


def test_fixed_operating_points_are_selected_on_validation() -> None:
    fpr = select_thresholds_at_fpr(TARGETS, SCORES, [0.5])
    recall = select_thresholds_at_recall(TARGETS, SCORES, [0.75])
    assert np.isclose(fpr[0]["threshold"], 0.7)
    assert np.isclose(fpr[0]["recall"], 1.0)
    assert np.isclose(recall[0]["threshold"], 0.7)
    assert np.isclose(recall[0]["precision"], 2 / 3)


def test_binary_selection_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        select_balanced_accuracy_threshold([0, 0], [0.2, 0.3])


def _write_credit_csv(path: Path) -> None:
    rows = 400
    rng = np.random.default_rng(72)
    targets = np.asarray([0] * 300 + [1] * 100, dtype=np.int64)
    feature = targets + rng.normal(0, 0.35, size=rows)
    frame = pd.DataFrame(
        {
            "SeriousDlqin2yrs": targets,
            "RevolvingUtilizationOfUnsecuredLines": feature,
            "age": rng.integers(20, 90, size=rows),
            "MonthlyIncome": rng.normal(5_000, 1_000, size=rows),
            "NumberOfDependents": rng.integers(0, 5, size=rows),
        }
    )
    frame.to_csv(path, index=False)


def test_credit_run_uses_validation_before_test(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from src.config import load_config, save_config
    from src.data import build_dataset
    from src.data.partition import (
        partition_labels_dirichlet,
        split_client_indices,
    )
    from src.evaluation.credit_operating_points import evaluate_credit_run
    from src.models import MLPClassifier

    data_path = tmp_path / "cs-training.csv"
    _write_credit_csv(data_path)
    run_dir = (
        tmp_path
        / "outputs"
        / "give_me_some_credit"
        / "fedavg"
        / "alpha_0p3"
        / "seed_7"
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    config = load_config("configs/give_me_some_credit.yaml")
    config["experiment"]["seed"] = 7
    config["dataset"]["path"] = str(data_path)
    config["partition"].update(num_clients=5, min_client_samples=20)
    config["model"].update(hidden_dims=[8, 4], dropout=0.0)
    config["runtime"].update(device="cpu", num_workers=0, pin_memory=False)
    save_config(config, run_dir / "config.yaml")

    bundle = build_dataset(config["dataset"], seed=7)
    partitions = partition_labels_dirichlet(
        bundle.train_targets,
        num_clients=5,
        alpha=0.3,
        min_client_samples=20,
        seed=7,
    )
    splits = split_client_indices(
        partitions, validation_fraction=0.2, seed=24
    )
    (run_dir / "client_partitions.json").write_text(
        json.dumps(
            {
                str(client_id): {
                    "validation": split["validation"],
                }
                for client_id, split in splits.items()
            }
        ),
        encoding="utf-8",
    )

    model = MLPClassifier(bundle.input_dim, [8, 4], 2, 0.0)
    for round_number in (10, 20):
        torch.save(
            {
                "round": round_number,
                "model_state": {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                },
            },
            checkpoint_dir / f"round_{round_number:04d}.pt",
        )

    result = evaluate_credit_run(
        run_dir,
        tmp_path / "operating_points",
        device_name="cpu",
        fpr_targets=[0.05],
        recall_targets=[0.5],
    )
    assert result["protocol"]["test_selection_prohibited"] is True
    assert result["protocol"]["test_evaluations_after_selection"] == 1
    assert result["selected_checkpoint"]["round"] == 10
    assert len(result["fixed_fpr_operating_points"]) == 1
    assert len(result["fixed_recall_operating_points"]) == 1
    output = (
        tmp_path
        / "operating_points"
        / "give_me_some_credit"
        / "fedavg"
        / "alpha_0p3"
        / "seed_7"
    )
    assert (output / "operating_point_metrics.json").is_file()
    assert (output / "checkpoint_validation_scores.csv").is_file()
