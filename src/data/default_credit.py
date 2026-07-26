"""Loader for UCI Default of Credit Card Clients (dataset id 350)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .base import ArrayDataset, DatasetBundle
from .preprocessing import prepare_tabular_data


_CATEGORICAL_ALIASES = {
    "X2",
    "X3",
    "X4",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
}


def _load_raw(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load UCI data from a local cache/path or the official API."""

    source = str(config.get("source", "ucimlrepo"))
    cache_dir = Path(str(config.get("cache_dir", "data/default_credit")))
    cache_file = cache_dir / "default_credit_raw.csv"
    explicit_path = config.get("path")

    if explicit_path:
        path = Path(str(explicit_path))
        if not path.is_file():
            raise FileNotFoundError(f"Default-credit file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".xls", ".xlsx"}:
            frame = pd.read_excel(path)
        else:
            frame = pd.read_csv(path)
        target_column = str(
            config.get("target_column", "default payment next month")
        )
        candidates = [
            target_column,
            "default payment next month",
            "default_payment_next_month",
            "Y",
        ]
        actual_target = next((name for name in candidates if name in frame), None)
        if actual_target is None:
            raise ValueError(
                f"Could not find target column; tried {candidates}. "
                f"Available columns: {list(frame.columns)}"
            )
        y = frame.pop(actual_target)
        return frame, y, f"file:{path}"

    if cache_file.is_file():
        frame = pd.read_csv(cache_file)
        y = frame.pop("__target__")
        return frame, y, f"cache:{cache_file}"
    if source != "ucimlrepo":
        raise ValueError("default_credit source must be 'ucimlrepo' or use path.")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Online UCI loading requires ucimlrepo; install requirements.txt."
        ) from exc

    dataset = fetch_ucirepo(id=int(config.get("uci_id", 350)))
    x = dataset.data.features.copy()
    targets = dataset.data.targets
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one target column, got {targets.shape[1]}.")
    y = targets.iloc[:, 0].copy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = x.copy()
    cached["__target__"] = y.to_numpy()
    cached.to_csv(cache_file, index=False)
    return x, y, "ucimlrepo:350"


def load_default_credit(
    config: Mapping[str, Any], *, seed: int
) -> DatasetBundle:
    """Load, split, and preprocess the 30,000-row UCI credit dataset."""

    features, targets, source = _load_raw(config)
    features.columns = [str(column).strip() for column in features.columns]
    for identifier in ("ID", "id"):
        if identifier in features:
            features = features.drop(columns=[identifier])
    categorical = [
        column for column in features.columns if column.upper() in _CATEGORICAL_ALIASES
    ]
    numeric = [column for column in features.columns if column not in categorical]
    split = prepare_tabular_data(
        features,
        targets,
        categorical_columns=categorical,
        numeric_columns=numeric,
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
        class_names=["non-default", "default"],
        metadata={
            "source": source,
            "raw_train_indices": split.train_indices,
            "raw_test_indices": split.test_indices,
        },
    )

