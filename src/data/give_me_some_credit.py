"""Loader for Kaggle's Give Me Some Credit labeled training CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .base import ArrayDataset, DatasetBundle
from .preprocessing import prepare_tabular_data


def load_give_me_some_credit(
    config: Mapping[str, Any], *, seed: int
) -> DatasetBundle:
    """Create an independent stratified test split from ``cs-training.csv``."""

    path = Path(str(config.get("path", "data/give_me_some_credit/cs-training.csv")))
    if not path.is_file():
        raise FileNotFoundError(
            f"Give Me Some Credit training file not found: {path}. "
            "Download the Kaggle competition and point dataset.path to "
            "cs-training.csv."
        )
    frame = pd.read_csv(path)
    unnamed = [column for column in frame if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    target_column = str(config.get("target_column", "SeriousDlqin2yrs"))
    if target_column not in frame:
        raise ValueError(
            f"Target {target_column!r} not found. Available: {list(frame.columns)}"
        )
    targets = frame.pop(target_column)
    if frame.select_dtypes(exclude="number").columns.tolist():
        raise ValueError("Give Me Some Credit loader expects numeric features only.")
    split = prepare_tabular_data(
        frame,
        targets,
        categorical_columns=[],
        numeric_columns=list(frame.columns),
        test_size=float(config.get("test_size", 0.2)),
        seed=seed,
        missing_strategy=str(config.get("missing_strategy", "median")),
        numeric_scaling=str(config.get("numeric_scaling", "standard")),
    )
    return DatasetBundle(
        train_dataset=ArrayDataset(split.x_train, split.y_train),
        test_dataset=ArrayDataset(split.x_test, split.y_test),
        train_targets=split.y_train,
        test_targets=split.y_test,
        input_dim=int(split.x_train.shape[1]),
        num_classes=2,
        minority_class=int(config.get("minority_class", 1)),
        feature_names=split.feature_names,
        class_names=["non-serious-delinquency", "serious-delinquency"],
        metadata={
            "source": f"file:{path}",
            "raw_train_indices": split.train_indices,
            "raw_test_indices": split.test_indices,
            "note": "Kaggle cs-test.csv is intentionally not used because labels are hidden.",
        },
    )

