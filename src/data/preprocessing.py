"""Leakage-safe preprocessing for tabular credit-risk datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class TabularSplit:
    """Train/test arrays plus fitted preprocessing metadata."""

    x_train: NDArray[np.float32]
    x_test: NDArray[np.float32]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    feature_names: list[str]
    train_indices: list[int]
    test_indices: list[int]
    preprocessor: Any


def prepare_tabular_data(
    features: pd.DataFrame,
    targets: pd.Series | NDArray[np.integer],
    *,
    categorical_columns: Sequence[str],
    numeric_columns: Sequence[str],
    test_size: float,
    seed: int,
    missing_strategy: str = "median",
    numeric_scaling: str = "standard",
) -> TabularSplit:
    """Split first, then fit preprocessing only on the training rows."""

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Tabular preprocessing requires scikit-learn; install requirements.txt."
        ) from exc

    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame.")
    y = np.asarray(targets, dtype=np.int64).reshape(-1)
    if len(features) != len(y):
        raise ValueError("features and targets have different row counts.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must lie strictly in (0, 1).")
    if missing_strategy not in {"median", "mean"}:
        raise ValueError("missing_strategy must be 'median' or 'mean'.")
    if numeric_scaling not in {"standard", "none"}:
        raise ValueError("numeric_scaling must be 'standard' or 'none'.")

    categorical = list(categorical_columns)
    numeric = list(numeric_columns)
    missing = set(categorical + numeric) - set(features.columns)
    if missing:
        raise ValueError(f"Configured columns are missing: {sorted(missing)}")
    overlap = set(categorical) & set(numeric)
    if overlap:
        raise ValueError(f"Columns cannot be both categorical and numeric: {overlap}")
    unused = set(features.columns) - set(categorical) - set(numeric)
    if unused:
        raise ValueError(f"Columns were not assigned a preprocessing role: {unused}")

    all_indices = np.arange(len(features), dtype=np.int64)
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=float(test_size),
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    x_train_frame = features.iloc[train_idx].copy()
    x_test_frame = features.iloc[test_idx].copy()

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=missing_strategy))
    ]
    if numeric_scaling == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float32,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    x_train = np.asarray(
        preprocessor.fit_transform(x_train_frame), dtype=np.float32
    )
    x_test = np.asarray(preprocessor.transform(x_test_frame), dtype=np.float32)
    if not np.all(np.isfinite(x_train)) or not np.all(np.isfinite(x_test)):
        raise ValueError("Preprocessing produced NaN or Inf values.")
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return TabularSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y[train_idx],
        y_test=y[test_idx],
        feature_names=names,
        train_indices=train_idx.astype(int).tolist(),
        test_indices=test_idx.astype(int).tolist(),
        preprocessor=preprocessor,
    )

