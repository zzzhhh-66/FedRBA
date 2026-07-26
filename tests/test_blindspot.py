"""Blindspot normalization and monotonicity tests."""

from __future__ import annotations

import numpy as np

from src.federated.blindspot import (
    ClassStatistics,
    blindspot_class_weights,
    update_blindspot_weights,
)


def test_blindspot_is_normalized_finite_and_tracks_error() -> None:
    statistics = [
        ClassStatistics(
            count=np.asarray([100, 100, 100]),
            loss_sum=np.asarray([20.0, 60.0, 100.0]),
            correct=np.asarray([90, 50, 10]),
        )
    ]
    updated, details = update_blindspot_weights(
        np.full(3, 1 / 3),
        statistics,
        temperature=0.5,
        ema=0.0,
    )
    assert np.all(updated >= 0)
    assert np.all(np.isfinite(updated))
    assert np.isclose(updated.sum(), 1.0)
    assert updated[2] > updated[1] > updated[0]
    np.testing.assert_allclose(details["class_error"], [0.1, 0.5, 0.9])


def test_ema_and_local_class_weights_remain_normalized() -> None:
    statistics = [
        ClassStatistics(
            count=np.asarray([10, 10]),
            loss_sum=np.asarray([1.0, 5.0]),
            correct=np.asarray([9, 2]),
        )
    ]
    updated, _ = update_blindspot_weights(
        np.asarray([0.8, 0.2]),
        statistics,
        temperature=0.5,
        ema=0.8,
    )
    class_weights = blindspot_class_weights(updated, strength=0.8)
    assert np.isclose(updated.sum(), 1.0)
    assert np.isclose(class_weights.mean(), 1.0)
    assert class_weights[1] > 0
