"""Leakage-safe tabular preprocessing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import prepare_tabular_data


def test_tabular_preprocessing_is_finite_reproducible_and_stratified() -> None:
    features = pd.DataFrame(
        {
            "amount": [float(index) if index % 7 else np.nan for index in range(100)],
            "age": np.arange(100) % 40 + 20,
            "category": ["a", "b"] * 50,
        }
    )
    targets = np.asarray([0] * 80 + [1] * 20)
    kwargs = dict(
        categorical_columns=["category"],
        numeric_columns=["amount", "age"],
        test_size=0.2,
        seed=42,
        missing_strategy="median",
        numeric_scaling="standard",
    )
    first = prepare_tabular_data(features, targets, **kwargs)
    second = prepare_tabular_data(features, targets, **kwargs)
    np.testing.assert_array_equal(first.x_train, second.x_train)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    assert np.all(np.isfinite(first.x_train))
    assert np.all(np.isfinite(first.x_test))
    assert first.y_test.sum() == 4
    assert set(first.train_indices).isdisjoint(first.test_indices)

