"""Offline tests for both tabular credit dataset loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.default_credit import load_default_credit
from src.data.give_me_some_credit import load_give_me_some_credit


def test_default_credit_local_csv_loader(tmp_path) -> None:
    rows = 200
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(
        {
            "ID": np.arange(rows),
            "LIMIT_BAL": rng.integers(10_000, 500_000, rows),
            "SEX": rng.integers(1, 3, rows),
            "EDUCATION": rng.integers(1, 5, rows),
            "MARRIAGE": rng.integers(1, 4, rows),
            "AGE": rng.integers(20, 70, rows),
            "PAY_0": rng.integers(-1, 5, rows),
            "BILL_AMT1": rng.normal(50_000, 10_000, rows),
            "default payment next month": np.asarray([0] * 150 + [1] * 50),
        }
    )
    path = tmp_path / "default.csv"
    frame.to_csv(path, index=False)
    bundle = load_default_credit(
        {
            "path": str(path),
            "target_column": "auto",
            "test_size": 0.2,
            "missing_strategy": "median",
            "numeric_scaling": "standard",
            "minority_class": 1,
        },
        seed=42,
    )
    assert len(bundle.train_dataset) == 160
    assert len(bundle.test_dataset) == 40
    assert bundle.input_dim > 7
    assert bundle.test_targets.sum() == 10
    assert bundle.feature_names is not None


def test_give_me_some_credit_uses_labeled_training_csv(tmp_path) -> None:
    rows = 200
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(rows),
            "SeriousDlqin2yrs": np.asarray([0] * 180 + [1] * 20),
            "RevolvingUtilizationOfUnsecuredLines": rng.random(rows),
            "age": rng.integers(20, 90, rows),
            "MonthlyIncome": rng.normal(5_000, 1_000, rows),
            "NumberOfDependents": rng.integers(0, 5, rows).astype(float),
        }
    )
    frame.loc[::9, "MonthlyIncome"] = np.nan
    path = tmp_path / "cs-training.csv"
    frame.to_csv(path, index=False)
    bundle = load_give_me_some_credit(
        {
            "path": str(path),
            "target_column": "SeriousDlqin2yrs",
            "test_size": 0.2,
            "missing_strategy": "median",
            "numeric_scaling": "standard",
            "minority_class": 1,
        },
        seed=42,
    )
    assert len(bundle.train_dataset) == 160
    assert len(bundle.test_dataset) == 40
    assert bundle.input_dim == 4
    assert bundle.test_targets.sum() == 4
    assert "cs-test.csv" in bundle.metadata["note"]

